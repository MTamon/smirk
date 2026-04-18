"""Extract FLAME parameters from a video using SMIRK and save them to disk.

This script is the storage-only counterpart to ``demo_video.py``. It does
NOT render meshes or produce a side-by-side video; it runs the SMIRK
encoder over every frame and writes the five FLARE-compatible parameter
tensors (``shape``, ``exp``, ``pose``, ``cam``, ``eyelid``) plus optional
MediaPipe-Tasks-derived ``eyes_pose`` / ``eyelids`` to a single ``.pt``
(or ``.npz``) file.

Output schema (``.pt``)::

    {
        "shape":      Tensor (T, 300),
        "exp":        Tensor (T,  50),
        "pose":       Tensor (T,   6),   # global rotation (3) + jaw (3)
        "cam":        Tensor (T,   3),
        "eyelid":     Tensor (T,   2),   # SMIRK native 2D eyelid
        "fps":        float,             # source video fps
        "num_frames": int,
        "valid_mask": BoolTensor (T,),   # False when face_detect failed
        "source":     str,

        # present only when --with_eye_pose is set:
        "eyes_pose":  Tensor (T, 12),    # MediaPipe-blendshape rot6d eyeballs
        "eyelids":    Tensor (T, 2),     # MediaPipe blendshape eyeBlink{L,R}

        # present only when --benchmark is set:
        "bench": {
            "total_seconds":      float,
            "encode_seconds":     float,
            "mediapipe_seconds":  float,
            "encode_fps":         float,   # SMIRK batch throughput only
            "end_to_end_fps":     float,   # frames / total_seconds
        },
    }

Examples::

    python demo_save_flame.py --input_path samples/example.mp4 --crop
    python demo_save_flame.py --input_path samples/example.mp4 \
        --with_eye_pose --benchmark --out_path output/example_flame.pt
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp

from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe, run_mediapipe_full
from utils.eye_pose import estimate_eye_pose_and_eyelid


PARAM_DIMS = {'shape': 300, 'exp': 50, 'pose': 6, 'cam': 3, 'eyelid': 2}


def crop_face(landmarks, scale=1.4, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([
        right - (right - left) / 2.0,
        bottom - (bottom - top) / 2.0,
    ])
    size = int(old_size * scale)

    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])
    dst_pts = np.array([
        [0, 0],
        [0, image_size - 1],
        [image_size - 1, 0],
    ])
    return estimate_transform('similarity', src_pts, dst_pts)


def detect_and_crop(frame_bgr, need_blendshapes, mp_delegate, image_size=224):
    """Return (rgb_224 uint8 or None, blendshapes or None, mp_elapsed).

    When need_blendshapes is False this uses the lighter ``run_mediapipe``
    path and the returned blendshapes dict is None.
    """
    t0 = time.perf_counter()
    if need_blendshapes:
        mp_result = run_mediapipe_full(frame_bgr, delegate=mp_delegate)
        t_mp = time.perf_counter() - t0
        if mp_result is None:
            return None, None, t_mp
        landmarks = mp_result['landmarks'][..., :2]
        blendshapes = mp_result['blendshapes']
    else:
        landmarks = run_mediapipe(frame_bgr, delegate=mp_delegate)
        t_mp = time.perf_counter() - t0
        if landmarks is None:
            return None, None, t_mp
        landmarks = landmarks[..., :2]
        blendshapes = None

    tform = crop_face(landmarks, scale=1.4, image_size=image_size)
    cropped = warp(
        frame_bgr,
        tform.inverse,
        output_shape=(image_size, image_size),
        preserve_range=True,
    ).astype(np.uint8)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped, blendshapes, t_mp


def resize_only(frame_bgr, image_size=224):
    cropped = cv2.resize(frame_bgr, (image_size, image_size))
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def to_tensor_batch(imgs_rgb, device):
    arr = np.stack(imgs_rgb, axis=0).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t.to(device, non_blocking=True)


def load_encoder(checkpoint_path, device):
    encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_state = {
        k.replace('smirk_encoder.', ''): v
        for k, v in checkpoint.items()
        if 'smirk_encoder' in k
    }
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder


def main():
    parser = argparse.ArgumentParser(
        description='Extract FLAME parameters from a video with SMIRK and '
                    'save them to disk (no rendering).'
    )
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input video.')
    parser.add_argument('--checkpoint', type=str,
                        default='pretrained_models/SMIRK_em1.pt',
                        help='Path to the SMIRK checkpoint (.pt).')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output file path. Suffix decides format: '
                             '.pt (torch.save, default) or .npz (numpy). '
                             'Defaults to output/<input_stem>_flame.pt.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on.')
    parser.add_argument('--crop', action='store_true',
                        help='Crop the face with mediapipe before encoding.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Frames per SMIRK forward pass.')
    parser.add_argument('--with_eye_pose', action='store_true',
                        help='Also estimate eyes_pose (12D rot6d) and '
                             'eyelids (2D) from MediaPipe Tasks blendshapes '
                             'and save them alongside the SMIRK outputs. '
                             'Implies --crop (mediapipe must run per-frame).')
    parser.add_argument('--benchmark', action='store_true',
                        help='Measure and report per-stage timings + FPS.')
    parser.add_argument('--mp_delegate', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='MediaPipe Tasks inference delegate. GPU '
                             'requires a MediaPipe build with OpenGL ES '
                             'support; falls back to CPU on init failure.')
    args = parser.parse_args()

    if args.with_eye_pose and not args.crop:
        # blendshape path requires per-frame mediapipe detection anyway.
        args.crop = True

    if args.out_path is None:
        stem = os.path.splitext(os.path.basename(args.input_path))[0]
        args.out_path = os.path.join('output', f'{stem}_flame.pt')
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or '.', exist_ok=True)

    encoder = load_encoder(args.checkpoint, args.device)

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.input_path}')
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    per_frame_out: dict[int, dict[str, torch.Tensor]] = {}
    per_frame_eye: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    valid_mask: list[bool] = []

    batch_imgs: list[np.ndarray] = []
    batch_indices: list[int] = []

    t_start = time.perf_counter()
    t_encode_total = 0.0
    t_mp_total = 0.0

    def flush():
        nonlocal t_encode_total
        if not batch_imgs:
            return
        imgs_tensor = to_tensor_batch(batch_imgs, args.device)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t_enc0 = time.perf_counter()
        with torch.no_grad():
            out = encoder(imgs_tensor)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t_encode_total += time.perf_counter() - t_enc0

        cpu_out = {k: out[k].detach().cpu() for k in PARAM_DIMS}
        for row, src_idx in enumerate(batch_indices):
            per_frame_out[src_idx] = {k: cpu_out[k][row] for k in PARAM_DIMS}
        batch_imgs.clear()
        batch_indices.clear()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.crop:
            rgb_224, blendshapes, t_mp = detect_and_crop(
                frame, need_blendshapes=args.with_eye_pose,
                mp_delegate=args.mp_delegate,
            )
            t_mp_total += t_mp
            if rgb_224 is None:
                valid_mask.append(False)
                frame_count += 1
                continue
        else:
            rgb_224 = resize_only(frame)
            blendshapes = None

        valid_mask.append(True)
        batch_imgs.append(rgb_224)
        batch_indices.append(frame_count)

        if args.with_eye_pose:
            eyes_pose, eyelids = estimate_eye_pose_and_eyelid(blendshapes)
            per_frame_eye[frame_count] = (eyes_pose.squeeze(0), eyelids.squeeze(0))

        frame_count += 1
        if len(batch_imgs) >= args.batch_size:
            flush()

        if total > 0 and frame_count % max(1, total // 20) == 0:
            print(f'[demo_save_flame] {frame_count}/{total} frames')

    flush()
    cap.release()
    t_total = time.perf_counter() - t_start

    # Assemble per-key stacked tensors in frame order; zero-fill invalid frames.
    stacked: dict[str, torch.Tensor] = {}
    for key, dim in PARAM_DIMS.items():
        rows = []
        for idx in range(frame_count):
            if idx in per_frame_out:
                rows.append(per_frame_out[idx][key])
            else:
                rows.append(torch.zeros(dim))
        stacked[key] = torch.stack(rows) if rows else torch.zeros(0, dim)

    result = {
        **stacked,
        'fps': float(src_fps) if src_fps else 0.0,
        'num_frames': frame_count,
        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
        'source': os.path.abspath(args.input_path),
    }

    if args.with_eye_pose:
        eyes_pose_rows = []
        eyelids_rows = []
        for idx in range(frame_count):
            if idx in per_frame_eye:
                ep, el = per_frame_eye[idx]
                eyes_pose_rows.append(ep)
                eyelids_rows.append(el)
            else:
                eyes_pose_rows.append(torch.zeros(12))
                eyelids_rows.append(torch.zeros(2))
        result['eyes_pose'] = torch.stack(eyes_pose_rows) if eyes_pose_rows else torch.zeros(0, 12)
        result['eyelids'] = torch.stack(eyelids_rows) if eyelids_rows else torch.zeros(0, 2)

    if args.benchmark:
        encode_fps = (len(per_frame_out) / t_encode_total) if t_encode_total > 0 else 0.0
        e2e_fps = (frame_count / t_total) if t_total > 0 else 0.0
        mp_fps = (frame_count / t_mp_total) if t_mp_total > 0 else 0.0
        result['bench'] = {
            'total_seconds': t_total,
            'encode_seconds': t_encode_total,
            'mediapipe_seconds': t_mp_total,
            'encode_fps': encode_fps,
            'mediapipe_fps': mp_fps,
            'end_to_end_fps': e2e_fps,
            'mp_delegate': args.mp_delegate,
            'encode_device': args.device,
        }
        print(f'[bench] frames={frame_count}  valid={int(sum(valid_mask))}  '
              f'mp_delegate={args.mp_delegate}  encode_device={args.device}')
        print(f'[bench] total={t_total:.2f}s  encode={t_encode_total:.2f}s  '
              f'mediapipe={t_mp_total:.2f}s')
        print(f'[bench] encode_fps={encode_fps:.1f}  mediapipe_fps={mp_fps:.1f}  '
              f'end_to_end_fps={e2e_fps:.1f}')

    ext = os.path.splitext(args.out_path)[1].lower()
    if ext == '.npz':
        npz: dict[str, np.ndarray] = {
            k: result[k].numpy() if isinstance(result[k], torch.Tensor) else np.array(result[k])
            for k in result
            if k != 'bench'
        }
        if 'bench' in result:
            for bk, bv in result['bench'].items():
                npz[f'bench_{bk}'] = np.array(bv)
        np.savez(args.out_path, **npz)
    else:
        torch.save(result, args.out_path)

    print(f'[demo_save_flame] wrote {frame_count} frames to {args.out_path}')
    print(f'  shape:{tuple(result["shape"].shape)}  '
          f'exp:{tuple(result["exp"].shape)}  '
          f'pose:{tuple(result["pose"].shape)}  '
          f'cam:{tuple(result["cam"].shape)}  '
          f'eyelid:{tuple(result["eyelid"].shape)}')
    if args.with_eye_pose:
        print(f'  eyes_pose:{tuple(result["eyes_pose"].shape)}  '
              f'eyelids:{tuple(result["eyelids"].shape)}')
    print(f'  valid frames: {int(result["valid_mask"].sum())}/{frame_count}')


if __name__ == '__main__':
    main()
