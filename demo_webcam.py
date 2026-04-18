"""Real-time webcam FLAME extraction with SMIRK + live mesh overlay.

Opens a webcam (or a pre-recorded video in "ideal-source" mode), runs
per-frame face detection (MediaPipe Tasks API), encodes with SMIRK,
renders the reconstructed FLAME mesh with the existing Renderer, and
displays the webcam frame and mesh side-by-side with an FPS overlay.

The ideal-source mode is enabled by passing a video file path to
``--source`` instead of a camera index: it reads frames as fast as
``cv2.VideoCapture`` can decode them (no hardware-limited frame rate)
so the end-to-end pipeline throughput can be measured when the camera
is not the bottleneck. **The pipeline itself is identical** between
real-webcam and ideal-source modes — we do NOT pre-transfer frames to
the GPU or batch across frames — only the frame source differs. The
only other difference is the mirror flip, which is applied only for
live webcams (it would be incorrect for a recorded clip).

Optional features:
    --with_eye_pose : additionally extract rot6d eyes_pose (12D) and
                      blendshape eyelids (2D) per frame; printed to
                      stdout at a throttled rate and (if --save_path is
                      given) appended to an NDJSON log.
    --save_path     : append SMIRK (and optional eye-pose) parameters
                      per frame as JSON Lines to this file.
    --no_render     : skip the mesh render (SMIRK-only benchmark mode).

Controls:
    q : quit
    s : save a snapshot PNG to <out_dir>/snapshot_<ts>.png

Examples::

    # basic webcam demo with mesh overlay
    python demo_webcam.py --checkpoint pretrained_models/SMIRK_em1.pt

    # CPU-only benchmark (no rendering)
    python demo_webcam.py --device cpu --no_render

    # log SMIRK + eye-pose to JSONL while showing mesh
    python demo_webcam.py --with_eye_pose --save_path output/webcam_log.jsonl

    # ideal-source mode: read frames from a file as fast as possible.
    # Same pipeline as the live webcam, just no hardware FPS ceiling.
    python demo_webcam.py --source samples/dafoe.mp4 --no_render
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from src.smirk_encoder import SmirkEncoder
from utils.face_crop import fast_crop_face_bgr
from utils.mediapipe_utils import run_mediapipe, run_mediapipe_full
from utils.eye_pose import estimate_eye_pose_and_eyelid


def load_encoder(checkpoint_path, device):
    encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {
        k.replace('smirk_encoder.', ''): v
        for k, v in checkpoint.items()
        if 'smirk_encoder' in k
    }
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


def draw_overlay(canvas_bgr, fps, mp_ms, enc_ms, ren_ms, face_ok,
                 mp_delegate='cpu', enc_device='cpu', extras=None):
    h, w = canvas_bgr.shape[:2]
    pad = 8
    lines = [
        f'FPS: {fps:5.1f}',
        f'mp:{mp_ms:5.1f}ms  enc:{enc_ms:5.1f}ms  ren:{ren_ms:5.1f}ms',
        f'mp_delegate:{mp_delegate}  enc_device:{enc_device}',
        f'face: {"OK" if face_ok else "---"}',
    ]
    if extras:
        lines.extend(extras)
    for i, line in enumerate(lines):
        y = pad + 18 * (i + 1)
        cv2.putText(canvas_bgr, line, (pad + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas_bgr, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 240, 20), 1, cv2.LINE_AA)
    return canvas_bgr


def render_mesh(flame, renderer, outputs, device):
    """Run FLAME + Renderer and return a (224,224,3) uint8 BGR image."""
    flame_out = flame.forward(outputs)
    ren_out = renderer.forward(
        flame_out['vertices'], outputs['cam'],
        landmarks_fan=flame_out['landmarks_fan'],
        landmarks_mp=flame_out['landmarks_mp'],
    )
    img = ren_out['rendered_img']  # (1,3,H,W) in [0,1]
    img = (img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Real-time webcam SMIRK FLAME extraction with mesh overlay.',
    )
    parser.add_argument('--checkpoint', type=str,
                        default='pretrained_models/SMIRK_em1.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--source', type=str, default='0',
                        help='Frame source. Integer string → webcam index. '
                             'Otherwise a path to a video file (ideal-source '
                             'mode: reads as fast as VideoCapture decodes, '
                             'no hardware FPS ceiling). Pipeline is otherwise '
                             'identical between the two modes.')
    parser.add_argument('--camera', type=int, default=None,
                        help='(deprecated) synonym for --source <int>.')
    parser.add_argument('--width', type=int, default=1280,
                        help='Requested webcam capture width (webcam only).')
    parser.add_argument('--height', type=int, default=720,
                        help='Requested webcam capture height (webcam only).')
    parser.add_argument('--with_eye_pose', action='store_true',
                        help='Also estimate rot6d eyes_pose + blendshape '
                             'eyelids per frame via MediaPipe Tasks.')
    parser.add_argument('--no_render', action='store_true',
                        help='Skip the FLAME mesh render (benchmark only).')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Append per-frame params as NDJSON to this path.')
    parser.add_argument('--snapshot_dir', type=str, default='output',
                        help='Directory used for snapshot PNGs (key: s).')
    parser.add_argument('--window', type=str, default='SMIRK webcam',
                        help='OpenCV window title.')
    parser.add_argument('--mp_delegate', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='MediaPipe Tasks inference delegate. GPU '
                             'requires a MediaPipe build with OpenGL ES '
                             'support; falls back to CPU on init failure.')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if str(device) != args.device:
        print(f'[demo_webcam] requested {args.device} but using {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    encoder = load_encoder(args.checkpoint, device)

    flame = None
    renderer = None
    if not args.no_render:
        # Lazy import so benchmark mode works without FLAME assets.
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer
        flame = FLAME().to(device)
        renderer = Renderer().to(device)

    # Resolve --source: integer → webcam, anything else → video file path.
    source_arg = args.source if args.camera is None else str(args.camera)
    try:
        source_handle: object = int(source_arg)
        is_webcam = True
    except ValueError:
        source_handle = source_arg
        is_webcam = False

    cap = cv2.VideoCapture(source_handle)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open source: {source_handle!r}')
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f'[demo_webcam] source={source_handle!r} mode={"webcam" if is_webcam else "ideal-video"}')

    save_fp = None
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or '.', exist_ok=True)
        save_fp = open(args.save_path, 'a', buffering=1)  # line-buffered

    os.makedirs(args.snapshot_dir, exist_ok=True)

    fps_window = deque(maxlen=30)
    frame_idx = 0
    # Aggregate (sum over all frames) for the final summary print.
    agg_mp_s = 0.0
    agg_enc_s = 0.0
    agg_ren_s = 0.0
    agg_face_ok = 0
    t_run0 = time.perf_counter()
    print('[demo_webcam] q: quit, s: snapshot')

    try:
        while True:
            t_frame0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print('[demo_webcam] frame read failed / EOF')
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)  # mirror only for live selfie view
            orig_h, orig_w = frame.shape[:2]

            # --- MediaPipe ---
            t_mp0 = time.perf_counter()
            if args.with_eye_pose:
                mp_result = run_mediapipe_full(frame, delegate=args.mp_delegate)
                if mp_result is not None:
                    landmarks = mp_result['landmarks'][..., :2]
                    blendshapes = mp_result['blendshapes']
                else:
                    landmarks, blendshapes = None, None
            else:
                landmarks_full = run_mediapipe(frame, delegate=args.mp_delegate)
                landmarks = landmarks_full[..., :2] if landmarks_full is not None else None
                blendshapes = None
            t_mp = (time.perf_counter() - t_mp0) * 1000.0

            face_ok = landmarks is not None

            enc_ms = 0.0
            ren_ms = 0.0
            mesh_img = None
            outputs = None
            eyes_pose_np = None
            eyelids_np = None

            if face_ok:
                cropped_bgr = fast_crop_face_bgr(frame, landmarks, scale=1.4, image_size=224)
                cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                tensor = tensor.to(device, non_blocking=True)

                # --- SMIRK encode ---
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_enc0 = time.perf_counter()
                with torch.no_grad():
                    outputs = encoder(tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                enc_ms = (time.perf_counter() - t_enc0) * 1000.0

                # --- Optional eye-pose supplementation ---
                if args.with_eye_pose:
                    ep, el = estimate_eye_pose_and_eyelid(blendshapes, device='cpu')
                    eyes_pose_np = ep.squeeze(0).numpy()
                    eyelids_np = el.squeeze(0).numpy()

                # --- Render (optional) ---
                if not args.no_render:
                    t_ren0 = time.perf_counter()
                    mesh_img = render_mesh(flame, renderer, outputs, device)
                    ren_ms = (time.perf_counter() - t_ren0) * 1000.0

                # --- Log ---
                if save_fp is not None:
                    record = {
                        'frame': frame_idx,
                        'wall_time': time.time(),
                        'shape': outputs['shape'].detach().cpu().numpy().squeeze(0).tolist(),
                        'exp': outputs['exp'].detach().cpu().numpy().squeeze(0).tolist(),
                        'pose': outputs['pose'].detach().cpu().numpy().squeeze(0).tolist(),
                        'cam': outputs['cam'].detach().cpu().numpy().squeeze(0).tolist(),
                        'eyelid': outputs['eyelid'].detach().cpu().numpy().squeeze(0).tolist(),
                    }
                    if args.with_eye_pose and eyes_pose_np is not None:
                        record['eyes_pose'] = eyes_pose_np.tolist()
                        record['eyelids'] = eyelids_np.tolist()
                    save_fp.write(json.dumps(record) + '\n')

            # --- Display ---
            display_h = 480
            scale = display_h / orig_h
            webcam_disp = cv2.resize(frame, (int(orig_w * scale), display_h))
            if mesh_img is not None:
                mesh_disp = cv2.resize(mesh_img, (display_h, display_h))
                canvas = np.hstack([webcam_disp, mesh_disp])
            else:
                canvas = webcam_disp

            t_frame = time.perf_counter() - t_frame0
            fps_window.append(t_frame)
            avg_fps = len(fps_window) / max(sum(fps_window), 1e-6)

            agg_mp_s += t_mp / 1000.0
            agg_enc_s += enc_ms / 1000.0
            agg_ren_s += ren_ms / 1000.0
            if face_ok:
                agg_face_ok += 1

            extras = []
            if args.with_eye_pose and eyes_pose_np is not None:
                extras.append(
                    f'eyes_pose|L|≈{np.linalg.norm(eyes_pose_np[:6]):.2f}  '
                    f'blink L/R: {eyelids_np[0]:.2f}/{eyelids_np[1]:.2f}'
                )
            draw_overlay(
                canvas, avg_fps, t_mp, enc_ms, ren_ms, face_ok,
                mp_delegate=args.mp_delegate, enc_device=str(device),
                extras=extras,
            )

            cv2.imshow(args.window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                path = os.path.join(args.snapshot_dir, f'snapshot_{ts}.png')
                cv2.imwrite(path, canvas)
                print(f'[demo_webcam] saved {path}')

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_fp is not None:
            save_fp.close()
        t_run = time.perf_counter() - t_run0
        e2e_fps = (frame_idx / t_run) if t_run > 0 else 0.0
        def _avg_ms(total_s): return (1000.0 * total_s / frame_idx) if frame_idx else 0.0
        print(f'[demo_webcam] processed {frame_idx} frames in {t_run:.2f}s '
              f'(valid={agg_face_ok}, end_to_end_fps={e2e_fps:.1f})')
        print(f'[demo_webcam] avg-per-frame  mp={_avg_ms(agg_mp_s):.1f}ms  '
              f'enc={_avg_ms(agg_enc_s):.1f}ms  ren={_avg_ms(agg_ren_s):.1f}ms  '
              f'mp_delegate={args.mp_delegate}  enc_device={str(device)}  '
              f'mode={"webcam" if is_webcam else "ideal-video"}')


if __name__ == '__main__':
    main()
