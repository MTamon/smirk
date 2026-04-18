"""Extract FLAME parameters from a video using SMIRK and save them to disk.

This script is the storage-only counterpart to ``demo_video.py``. It does
NOT render meshes or produce a side-by-side video; it just runs the SMIRK
encoder over every frame and writes the five FLARE-compatible parameter
tensors (``shape``, ``exp``, ``pose``, ``cam``, ``eyelid``) to a single
``.pt`` (or ``.npz``) file so that downstream consumers (FLARE /
FlashAvatar / DECA pipelines) can load them directly.

Output schema (``.pt``)::

    {
        "shape":  Tensor (T, 300),
        "exp":    Tensor (T,  50),
        "pose":   Tensor (T,   6),   # global rotation (3) + jaw (3)
        "cam":    Tensor (T,   3),
        "eyelid": Tensor (T,   2),
        "fps":    float,
        "num_frames": int,
        "valid_mask": BoolTensor (T,),  # False for frames where mediapipe
                                        # found no landmarks and --crop was
                                        # requested (params are zeros then).
        "source":   str,                # input video path
    }

Example::

    python demo_save_flame.py \
        --input_path samples/example.mp4 \
        --checkpoint pretrained_models/SMIRK_em1.pt \
        --out_path output/example_flame.pt \
        --crop
"""

import argparse
import os

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp

from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe


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


def preprocess_frame(frame_bgr, crop, device, image_size=224):
    """Return a (1,3,H,W) float tensor in [0,1] on device, or None.

    None is only returned when ``crop`` is True and mediapipe fails to
    find landmarks; the caller should record that frame as invalid.
    """
    if crop:
        kpt = run_mediapipe(frame_bgr)
        if kpt is None:
            return None
        kpt = kpt[..., :2]
        tform = crop_face(kpt, scale=1.4, image_size=image_size)
        cropped = warp(
            frame_bgr,
            tform.inverse,
            output_shape=(image_size, image_size),
            preserve_range=True,
        ).astype(np.uint8)
    else:
        cropped = cv2.resize(frame_bgr, (image_size, image_size))

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


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
        description='Extract FLAME parameters from a video with SMIRK '
                    'and save them to disk (no rendering).'
    )
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input video.')
    parser.add_argument('--checkpoint', type=str,
                        default='pretrained_models/SMIRK_em1.pt',
                        help='Path to the SMIRK checkpoint (.pt).')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output file path. Suffix decides the format: '
                             '.pt (default, torch.save) or .npz (numpy). '
                             'If omitted, defaults to '
                             'output/<input_stem>_flame.pt.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on.')
    parser.add_argument('--crop', action='store_true',
                        help='Crop the face with mediapipe before encoding.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of frames to encode per forward pass.')
    args = parser.parse_args()

    if args.out_path is None:
        stem = os.path.splitext(os.path.basename(args.input_path))[0]
        args.out_path = os.path.join('output', f'{stem}_flame.pt')
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or '.', exist_ok=True)

    encoder = load_encoder(args.checkpoint, args.device)

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.input_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Collect per-frame validity and, for valid frames, the preprocessed tensor.
    # Encode in batches; store the encoder outputs indexed by source frame.
    frame_tensors = []       # list[Tensor (1,3,H,W)] for valid frames only
    frame_indices = []       # source frame index for each entry in frame_tensors
    valid_mask = []          # one bool per source frame

    per_frame_out = {}       # {frame_idx: {'shape': t, 'exp': t, ...}}

    def flush():
        if not frame_tensors:
            return
        imgs = torch.cat(frame_tensors, dim=0)
        with torch.no_grad():
            out = encoder(imgs)
        cpu_out = {k: out[k].detach().cpu() for k in PARAM_DIMS}
        for row, src_idx in enumerate(frame_indices):
            per_frame_out[src_idx] = {k: cpu_out[k][row] for k in PARAM_DIMS}
        frame_tensors.clear()
        frame_indices.clear()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tensor = preprocess_frame(frame, args.crop, args.device)
        if tensor is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            frame_tensors.append(tensor)
            frame_indices.append(frame_count)

        frame_count += 1
        if len(frame_tensors) >= args.batch_size:
            flush()

        if total > 0 and frame_count % max(1, total // 20) == 0:
            print(f'[demo_save_flame] {frame_count}/{total} frames')

    flush()
    cap.release()

    # Assemble per-key stacked tensors in frame order; zero-fill invalid frames.
    stacked = {}
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
        'fps': float(fps) if fps else 0.0,
        'num_frames': frame_count,
        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
        'source': os.path.abspath(args.input_path),
    }

    ext = os.path.splitext(args.out_path)[1].lower()
    if ext == '.npz':
        np.savez(
            args.out_path,
            shape=result['shape'].numpy(),
            exp=result['exp'].numpy(),
            pose=result['pose'].numpy(),
            cam=result['cam'].numpy(),
            eyelid=result['eyelid'].numpy(),
            fps=np.array(result['fps']),
            num_frames=np.array(result['num_frames']),
            valid_mask=result['valid_mask'].numpy(),
            source=np.array(result['source']),
        )
    else:
        torch.save(result, args.out_path)

    print(f'[demo_save_flame] wrote {frame_count} frames to {args.out_path}')
    print(f'  shape:{tuple(result["shape"].shape)}  '
          f'exp:{tuple(result["exp"].shape)}  '
          f'pose:{tuple(result["pose"].shape)}  '
          f'cam:{tuple(result["cam"].shape)}  '
          f'eyelid:{tuple(result["eyelid"].shape)}')
    print(f'  valid frames: {int(result["valid_mask"].sum())}/{frame_count}')


if __name__ == '__main__':
    main()
