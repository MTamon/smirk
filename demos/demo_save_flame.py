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

        # present only when --with_oral_features is set:
        "mediapipe_landmarks_2d":         Tensor (T, 478, 2),   # crop-space pixel coords
        "mediapipe_landmarks_inner_mouth_2d": Tensor (T, 20, 2),# inner-lip subset of the above
        "mediapipe_inner_mouth_indices":  LongTensor (20,),     # MediaPipe 478 indices (metadata)
        "mouth_blendshapes":              Tensor (T, 27),       # ARKit jaw*/mouth* scores
        "mouth_blendshape_names":         list[str] (27,),      # names aligned with last axis
        "mediapipe_transform":            Tensor (T, 4, 4),     # facial transform matrix
        "flame_landmarks_mp_3d":          Tensor (T, 105, 3),   # SMIRK-driven FLAME MP landmarks
        "flame_landmarks_inner_mouth_3d": Tensor (T, 20, 3),    # inner-lip subset in FLAME space
        "flame_mp_inner_mouth_positions": LongTensor (20,),     # positions within the 105-array

        # present only when --benchmark is set:
        "bench": {
            "total_seconds":      float,
            "encode_seconds":     float,
            "mediapipe_seconds":  float,   # == detect + warp (legacy alias)
            "decode_seconds":     float,   # cv2.VideoCapture.read()
            "detect_seconds":     float,   # mediapipe face_landmarker only
            "warp_seconds":       float,   # cv2.warpAffine crop
            "to_device_seconds":  float,   # numpy -> tensor -> GPU
            "encode_fps":         float,   # SMIRK batch throughput only
            "mediapipe_fps":      float,   # frames / mediapipe_seconds
            "end_to_end_fps":     float,   # frames / total_seconds
            "mp_delegate":        str,     # "cpu" | "gpu"
            "encode_device":      str,
            "warmup_frames":      int,     # frames excluded from bench
            "bench_frames":       int,     # frames counted toward bench
        },
    }

Examples::

    python demo_save_flame.py --input_path samples/example.mp4 --crop
    python demo_save_flame.py --input_path samples/example.mp4 \
        --with_eye_pose --benchmark --out_path output/example_flame.pt
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch

from src.smirk_encoder import SmirkEncoder
from utils.face_crop import fast_crop_face_bgr
from utils.mediapipe_utils import run_mediapipe, run_mediapipe_full
from utils.eye_pose import estimate_eye_pose_and_eyelid
from utils.oral_features import (
    ARKIT_MOUTH_KEYS,
    FLAME_MP_INNER_MOUTH,
    MP_INNER_MOUTH,
    apply_affine_2x3,
    extract_inner_mouth_landmarks_2d,
    extract_inner_mouth_landmarks_3d,
    extract_mouth_blendshapes,
)


PARAM_DIMS = {'shape': 300, 'exp': 50, 'pose': 6, 'cam': 3, 'eyelid': 2}


def detect_and_crop(frame_bgr, need_blendshapes, mp_delegate, image_size=224,
                    return_extras=False):
    """Return (rgb_224 uint8 or None, blendshapes or None, t_detect, t_warp).

    When ``need_blendshapes`` is False this uses the lighter
    ``run_mediapipe`` path and the returned blendshapes dict is None.
    Detection time and warp time are reported separately so per-stage
    benchmarks can tell them apart.

    When ``return_extras`` is True the return is a 5-tuple instead:
    ``(rgb_224, blendshapes, t_detect, t_warp, extras)`` where ``extras``
    is a dict carrying ``'landmarks_full'`` (the raw (478, 2) array in
    source-image pixel space), the full-res ``'transform'`` (4x4 or None),
    and the 2x3 crop affine ``'crop_affine'``. On detection failure
    ``extras`` is None.
    """
    t0 = time.perf_counter()
    if need_blendshapes:
        mp_result = run_mediapipe_full(frame_bgr, delegate=mp_delegate)
        t_detect = time.perf_counter() - t0
        if mp_result is None:
            if return_extras:
                return None, None, t_detect, 0.0, None
            return None, None, t_detect, 0.0
        landmarks = mp_result['landmarks'][..., :2]
        blendshapes = mp_result['blendshapes']
        transform = mp_result.get('transform')
    else:
        landmarks = run_mediapipe(frame_bgr, delegate=mp_delegate)
        t_detect = time.perf_counter() - t0
        if landmarks is None:
            if return_extras:
                return None, None, t_detect, 0.0, None
            return None, None, t_detect, 0.0
        landmarks = landmarks[..., :2]
        blendshapes = None
        transform = None

    t1 = time.perf_counter()
    crop_matrix: list = []
    cropped_bgr = fast_crop_face_bgr(
        frame_bgr, landmarks, scale=1.4, image_size=image_size,
        matrix_out=crop_matrix if return_extras else None,
    )
    cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    t_warp = time.perf_counter() - t1
    if return_extras:
        extras = {
            'landmarks_full': landmarks,
            'transform': transform,
            'crop_affine': crop_matrix[0] if crop_matrix else None,
        }
        return cropped, blendshapes, t_detect, t_warp, extras
    return cropped, blendshapes, t_detect, t_warp


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
    parser.add_argument('--with_oral_features', action='store_true',
                        help='Also save mouth-interior signals so downstream '
                             'avatars (FlashAvatar etc.) can drive teeth / '
                             'tongue: raw MediaPipe 478 landmarks in crop '
                             'space, the inner-lip subset (20 pts), ARKit '
                             'jaw/mouth blendshapes (27 values), the 4x4 '
                             'MediaPipe facial transform, and the SMIRK-driven '
                             '3D FLAME MediaPipe landmarks (105 pts + 20-pt '
                             'inner-mouth subset). Implies --crop and loads '
                             'FLAME on the selected device.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Measure and report per-stage timings + FPS.')
    parser.add_argument('--warmup', type=int, default=0,
                        help='Frames to process before resetting the bench '
                             'timers. Excludes one-off cold-start costs '
                             '(MediaPipe GPU delegate shader JIT, cuDNN '
                             'autotune, first malloc) from the reported FPS. '
                             'The warmup frames still land in the .pt output; '
                             'only the timing numbers skip them. Recommended '
                             'when using --mp_delegate gpu: try --warmup 10.')
    parser.add_argument('--mp_delegate', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='MediaPipe Tasks inference delegate. GPU '
                             'requires a MediaPipe build with OpenGL ES '
                             'support; falls back to CPU on init failure.')
    args = parser.parse_args()

    if args.with_eye_pose and not args.crop:
        # blendshape path requires per-frame mediapipe detection anyway.
        args.crop = True
    if args.with_oral_features and not args.crop:
        args.crop = True

    if args.out_path is None:
        stem = os.path.splitext(os.path.basename(args.input_path))[0]
        args.out_path = os.path.join('output', f'{stem}_flame.pt')
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or '.', exist_ok=True)

    if args.device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True

    encoder = load_encoder(args.checkpoint, args.device)

    # FLAME is only needed when --with_oral_features is requested (we call
    # it to get 3D MediaPipe-embedded landmarks driven by SMIRK parameters).
    flame = None
    if args.with_oral_features:
        from src.FLAME.FLAME import FLAME
        flame = FLAME().to(args.device)
        flame.eval()

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.input_path}')
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    per_frame_out: dict[int, dict[str, torch.Tensor]] = {}
    per_frame_eye: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    per_frame_oral: dict[int, dict[str, np.ndarray]] = {}
    valid_mask: list[bool] = []

    batch_imgs: list[np.ndarray] = []
    batch_indices: list[int] = []

    t_start = time.perf_counter()
    t_encode_total = 0.0
    t_detect_total = 0.0
    t_warp_total = 0.0
    t_decode_total = 0.0
    t_to_device_total = 0.0
    warmup_done = args.warmup <= 0
    per_frame_out_size_at_reset = 0

    def flush():
        nonlocal t_encode_total, t_to_device_total
        if not batch_imgs:
            return
        t_td0 = time.perf_counter()
        imgs_tensor = to_tensor_batch(batch_imgs, args.device)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t_to_device_total += time.perf_counter() - t_td0

        t_enc0 = time.perf_counter()
        with torch.no_grad():
            out = encoder(imgs_tensor)
            # Run FLAME inside the same no_grad block so the 3D MediaPipe
            # landmarks land on CPU for saving without autograd overhead.
            flame_mp_3d = None
            if flame is not None:
                flame_out = flame.forward(out)
                flame_mp_3d = flame_out['landmarks_mp'].detach().cpu().numpy()
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t_encode_total += time.perf_counter() - t_enc0

        cpu_out = {k: out[k].detach().cpu() for k in PARAM_DIMS}
        for row, src_idx in enumerate(batch_indices):
            per_frame_out[src_idx] = {k: cpu_out[k][row] for k in PARAM_DIMS}
            if flame_mp_3d is not None and src_idx in per_frame_oral:
                per_frame_oral[src_idx]['flame_landmarks_mp_3d'] = flame_mp_3d[row]
        batch_imgs.clear()
        batch_indices.clear()

    frame_count = 0
    while True:
        if not warmup_done and frame_count >= args.warmup:
            # Flush the pending batch so warmup-frame encode time is
            # attributed to the warmup window, then zero all per-stage
            # timers so the reported bench numbers reflect steady state.
            flush()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            per_frame_out_size_at_reset = len(per_frame_out)
            t_start = time.perf_counter()
            t_encode_total = 0.0
            t_detect_total = 0.0
            t_warp_total = 0.0
            t_decode_total = 0.0
            t_to_device_total = 0.0
            warmup_done = True

        t_dec0 = time.perf_counter()
        ret, frame = cap.read()
        t_decode_total += time.perf_counter() - t_dec0
        if not ret:
            break

        if args.crop:
            need_blendshapes = args.with_eye_pose or args.with_oral_features
            if args.with_oral_features:
                rgb_224, blendshapes, t_detect, t_warp, extras = detect_and_crop(
                    frame, need_blendshapes=need_blendshapes,
                    mp_delegate=args.mp_delegate,
                    return_extras=True,
                )
            else:
                rgb_224, blendshapes, t_detect, t_warp = detect_and_crop(
                    frame, need_blendshapes=need_blendshapes,
                    mp_delegate=args.mp_delegate,
                )
                extras = None
            t_detect_total += t_detect
            t_warp_total += t_warp
            if rgb_224 is None:
                valid_mask.append(False)
                frame_count += 1
                continue
        else:
            rgb_224 = resize_only(frame)
            blendshapes = None
            extras = None

        valid_mask.append(True)
        batch_imgs.append(rgb_224)
        batch_indices.append(frame_count)

        if args.with_eye_pose:
            eyes_pose, eyelids = estimate_eye_pose_and_eyelid(blendshapes)
            per_frame_eye[frame_count] = (eyes_pose.squeeze(0), eyelids.squeeze(0))

        if args.with_oral_features and extras is not None:
            # Transform raw MP landmarks into crop space so 2D coords align
            # with the 224x224 image SMIRK is encoding. The affine is the
            # same one fast_crop_face_bgr applied to the pixels, so points
            # and pixels stay in lockstep.
            lmks_full = extras['landmarks_full']  # (478, 2), source-image px
            crop_M = extras['crop_affine']        # (2, 3) or None
            if crop_M is not None:
                lmks_crop = apply_affine_2x3(lmks_full, crop_M).astype(np.float32)
            else:
                lmks_crop = lmks_full.astype(np.float32)
            inner_mouth_2d = extract_inner_mouth_landmarks_2d(lmks_crop)
            mouth_bs = extract_mouth_blendshapes(blendshapes)
            transform = extras['transform']
            if transform is None:
                transform = np.eye(4, dtype=np.float32)
            per_frame_oral[frame_count] = {
                'mediapipe_landmarks_2d': lmks_crop,
                'mediapipe_landmarks_inner_mouth_2d': inner_mouth_2d.astype(np.float32),
                'mouth_blendshapes': mouth_bs,
                'mediapipe_transform': transform.astype(np.float32),
            }

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

    if args.with_oral_features:
        oral_keys_shape = {
            'mediapipe_landmarks_2d': (478, 2),
            'mediapipe_landmarks_inner_mouth_2d': (len(MP_INNER_MOUTH), 2),
            'mouth_blendshapes': (len(ARKIT_MOUTH_KEYS),),
            'mediapipe_transform': (4, 4),
            'flame_landmarks_mp_3d': (105, 3),
        }
        for key, shape in oral_keys_shape.items():
            rows = []
            for idx in range(frame_count):
                rec = per_frame_oral.get(idx)
                if rec is not None and key in rec:
                    rows.append(torch.from_numpy(np.asarray(rec[key], dtype=np.float32)))
                else:
                    rows.append(torch.zeros(*shape, dtype=torch.float32))
            result[key] = torch.stack(rows) if rows else torch.zeros(0, *shape)
        # Inner-mouth 3D subset is sliced from the FLAME 105-point tensor so
        # the layout matches the 2D subset ordering exactly.
        if result['flame_landmarks_mp_3d'].numel() > 0:
            result['flame_landmarks_inner_mouth_3d'] = result['flame_landmarks_mp_3d'][
                :, list(FLAME_MP_INNER_MOUTH), :
            ].contiguous()
        else:
            result['flame_landmarks_inner_mouth_3d'] = torch.zeros(0, len(MP_INNER_MOUTH), 3)
        # Index / metadata arrays (constant across frames; saved once so
        # downstream code doesn't need to re-derive them).
        result['mediapipe_inner_mouth_indices'] = torch.tensor(
            list(MP_INNER_MOUTH), dtype=torch.long,
        )
        result['flame_mp_inner_mouth_positions'] = torch.tensor(
            list(FLAME_MP_INNER_MOUTH), dtype=torch.long,
        )
        result['mouth_blendshape_names'] = list(ARKIT_MOUTH_KEYS)

    if args.benchmark:
        warmup_frames = args.warmup if warmup_done and args.warmup > 0 else 0
        bench_frames = max(0, frame_count - warmup_frames)
        bench_encoded = max(0, len(per_frame_out) - per_frame_out_size_at_reset)
        t_mp_total = t_detect_total + t_warp_total
        encode_fps = (bench_encoded / t_encode_total) if t_encode_total > 0 else 0.0
        e2e_fps = (bench_frames / t_total) if t_total > 0 else 0.0
        mp_fps = (bench_frames / t_mp_total) if t_mp_total > 0 else 0.0
        result['bench'] = {
            'total_seconds': t_total,
            'encode_seconds': t_encode_total,
            'mediapipe_seconds': t_mp_total,
            'decode_seconds': t_decode_total,
            'detect_seconds': t_detect_total,
            'warp_seconds': t_warp_total,
            'to_device_seconds': t_to_device_total,
            'encode_fps': encode_fps,
            'mediapipe_fps': mp_fps,
            'end_to_end_fps': e2e_fps,
            'mp_delegate': args.mp_delegate,
            'encode_device': args.device,
            'warmup_frames': warmup_frames,
            'bench_frames': bench_frames,
        }
        print(f'[bench] frames={frame_count}  valid={int(sum(valid_mask))}  '
              f'warmup={warmup_frames}  bench_frames={bench_frames}  '
              f'mp_delegate={args.mp_delegate}  encode_device={args.device}')
        print(f'[bench] total={t_total:.2f}s  decode={t_decode_total:.2f}s  '
              f'detect={t_detect_total:.2f}s  warp={t_warp_total:.2f}s  '
              f'to_device={t_to_device_total:.2f}s  encode={t_encode_total:.2f}s')
        print(f'[bench] encode_fps={encode_fps:.1f}  mediapipe_fps={mp_fps:.1f}  '
              f'end_to_end_fps={e2e_fps:.1f}')

    ext = os.path.splitext(args.out_path)[1].lower()
    if ext == '.npz':
        npz: dict[str, np.ndarray] = {}
        for k, v in result.items():
            if k == 'bench':
                continue
            if isinstance(v, torch.Tensor):
                npz[k] = v.numpy()
            elif isinstance(v, list) and v and isinstance(v[0], str):
                # String arrays (e.g. mouth_blendshape_names) need dtype=object
                # so numpy doesn't pad them to a fixed-width ascii dtype.
                npz[k] = np.array(v, dtype=object)
            else:
                npz[k] = np.array(v)
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
    if args.with_oral_features:
        print(f'  mp_landmarks_2d:{tuple(result["mediapipe_landmarks_2d"].shape)}  '
              f'inner_mouth_2d:{tuple(result["mediapipe_landmarks_inner_mouth_2d"].shape)}  '
              f'mouth_bs:{tuple(result["mouth_blendshapes"].shape)}  '
              f'mp_transform:{tuple(result["mediapipe_transform"].shape)}  '
              f'flame_mp_3d:{tuple(result["flame_landmarks_mp_3d"].shape)}  '
              f'inner_mouth_3d:{tuple(result["flame_landmarks_inner_mouth_3d"].shape)}')
    print(f'  valid frames: {int(result["valid_mask"].sum())}/{frame_count}')


if __name__ == '__main__':
    main()
