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
    --capture_only  : skip MediaPipe + SMIRK + render entirely. Measures
                      the I/O ceiling (cap.read + imshow + waitKey) so
                      "is the camera the bottleneck?" can be answered
                      independently of inference.
    --fourcc MJPG   : pixel format requested from a UVC webcam. Defaults
                      to MJPG — without this most USB 2.0 UVC cameras
                      fall back to YUYV and cap cap.read() at ~5 fps
                      even though the host can render far faster. The
                      missing ~170ms/frame at 5 FPS is entirely the
                      USB frame-arrival wait, not host-side processing.

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
import sys
import threading
import time
from collections import deque

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch

from src.smirk_encoder import SmirkEncoder
from utils.bbox_tracker import OnlineBBoxTracker
from utils.face_crop import fast_crop_face_bgr
from utils.mediapipe_utils import run_mediapipe, run_mediapipe_full
from utils.eye_pose import estimate_eye_pose_and_eyelid
from utils.oral_features import (
    ARKIT_MOUTH_KEYS,
    FLAME_MP_INNER_MOUTH,
    MP_INNER_MOUTH,
    apply_affine_2x3,
    extract_inner_mouth_landmarks_2d,
    extract_mouth_blendshapes,
)
from utils.vertex_viz import (
    add_vertex_viz_args,
    draw_vertex_points_bgr,
    ndc_to_crop_pixels,
)


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


def draw_overlay(canvas_bgr, fps, timings, face_ok,
                 mp_delegate='cpu', enc_device='cpu', extras=None):
    """Overlay per-stage timings on the display canvas.

    ``timings`` is a dict of stage-name -> milliseconds for the most
    recent frame (``cap``, ``mp``, ``pre``, ``enc``, ``ren``, ``disp``,
    ``gui`` are expected; missing keys render as ``---``).
    """
    pad = 8

    def t(k):
        v = timings.get(k)
        return '  --' if v is None else f'{v:5.1f}'

    total = sum(v for v in timings.values() if v is not None)
    lines = [
        f'FPS: {fps:5.1f}  total:{total:5.1f}ms',
        f'cap:{t("cap")}  mp:{t("mp")}  pre:{t("pre")}',
        f'enc:{t("enc")}  ren:{t("ren")}',
        f'disp:{t("disp")}  gui:{t("gui")}',
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


class CaptureThread:
    """Background thread that pulls frames from cv2.VideoCapture.

    On some OpenCV V4L2 builds the main-loop ``cap.read()`` halves the
    effective FPS because MJPG CPU decode + imshow/waitKey overlap with
    the camera's ISO-cadence; the driver keeps the next frame queued
    while cv2 is busy. Running cap.read() in its own thread lets the
    driver drain continuously, and the main loop just grabs whatever
    the latest decoded frame is. DECA's webcam demo benefits from the
    same pattern implicitly because its per-frame work is so small.
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stop = False
        self._last_seq_read = 0
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._error = 'cap.read() returned False'
                break
            with self._lock:
                self._latest = frame
                self._seq += 1

    def read(self, timeout=2.0):
        deadline = time.perf_counter() + timeout
        while True:
            with self._lock:
                if self._error is not None:
                    return False, None
                if self._latest is not None and self._seq != self._last_seq_read:
                    self._last_seq_read = self._seq
                    return True, self._latest
            if time.perf_counter() > deadline:
                return False, None
            time.sleep(0.001)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)


def render_mesh(flame, renderer, outputs, device):
    """Run FLAME + Renderer.

    Returns ``(mesh_bgr, ren_out)`` where ``mesh_bgr`` is a (224, 224, 3)
    uint8 BGR image and ``ren_out`` is the raw renderer dict (needed for
    ``transformed_vertices`` when ``--show_vertices`` is on).
    """
    flame_out = flame.forward(outputs)
    ren_out = renderer.forward(
        flame_out['vertices'], outputs['cam'],
        landmarks_fan=flame_out['landmarks_fan'],
        landmarks_mp=flame_out['landmarks_mp'],
    )
    img = ren_out['rendered_img']  # (1,3,H,W) in [0,1]
    img = (img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, ren_out


def _alpha_blend_mesh_over_crop_bgr(crop_bgr, mesh_bgr, alpha):
    """Alpha-blend ``mesh_bgr`` over ``crop_bgr`` wherever the mesh is non-black.

    Both are (H, W, 3) uint8 BGR; returns a new uint8 BGR array. Channel
    order does not matter for the mask or the blend math, so this mirrors
    ``utils.vertex_viz.alpha_blend_mesh_over_input`` without the torch
    round-trip.
    """
    mesh_mask = (mesh_bgr.astype(np.int32).sum(axis=2, keepdims=True) > 3).astype(np.float32)
    eff_alpha = mesh_mask * float(alpha)
    blended = crop_bgr.astype(np.float32) * (1.0 - eff_alpha) + mesh_bgr.astype(np.float32) * eff_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


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
    parser.add_argument('--fourcc', type=str, default='MJPG',
                        help='Webcam pixel-format FOURCC. Most UVC cameras '
                             'fall back to ~5 fps at 720p under the default '
                             'YUYV because of USB 2.0 bandwidth; MJPG enables '
                             'in-camera JPEG and restores ~30 fps. Use YUYV '
                             'only if the camera does not support MJPG or you '
                             'need uncompressed frames. Set to empty string '
                             '("") to leave the driver default untouched.')
    parser.add_argument('--auto_exposure', type=str, default='auto',
                        choices=['auto', 'manual'],
                        help='Toggle UVC auto-exposure. When FPS is capped at '
                             '15 even with MJPG, the camera is usually in AE '
                             'mode and stretching exposure time in dim light, '
                             'which upper-bounds fps as 1/exposure_seconds. '
                             'Switch to manual and pair with --exposure to '
                             'remove that ceiling (image will be darker).')
    parser.add_argument('--exposure', type=float, default=None,
                        help='Manual exposure value passed to '
                             'cv2.CAP_PROP_EXPOSURE. On Linux V4L2 this is in '
                             '100us units (e.g. 200 -> 20ms exposure -> 50 '
                             'fps ceiling). On Windows MSMF/DSHOW it is '
                             'log2-scaled (try -6 .. -4). Ignored unless '
                             '--auto_exposure manual. Note: some UVC cams '
                             '(e.g. Sunplus FHD) will accept the value '
                             '(echoed back by exposure_ctrl readback) yet '
                             'still stretch the actual integration time in '
                             'dim scenes, so FPS can still drop. Check the '
                             'startup log and confirm by shining a light.')
    parser.add_argument('--camera_fps', type=float, default=None,
                        help='Request a target frame rate from the camera via '
                             'cv2.CAP_PROP_FPS. Most UVC drivers honor it if '
                             'the (fourcc, width, height, fps) tuple exists in '
                             'the camera descriptor; silently ignored '
                             'otherwise. Use when the startup log shows a '
                             'reported_fps lower than the camera spec and you '
                             'want to force negotiation to a higher mode.')
    parser.add_argument('--buffersize', type=int, default=None,
                        help='cv2.CAP_PROP_BUFFERSIZE. Default is to leave the '
                             'driver default (usually 4). Setting 1 is '
                             'tempting (latest frame only) but on several V4L2 '
                             'builds it drops the effective FPS by ~30 percent '
                             '(measured 30 -> 20 fps on Sunplus FHD) because '
                             'cap.read() waits for a buffer flip every call. '
                             'Combined with AUTO_EXPOSURE re-assert it drops '
                             'to 15 fps (the classic halving). Prefer '
                             '--capture_thread for low-latency instead.')
    parser.add_argument('--capture_thread', action='store_true',
                        help='Run cap.read() in a dedicated background thread '
                             'so the main loop always receives the most recent '
                             'decoded frame without blocking. Fixes the classic '
                             "cv2 V4L2 + MJPG halving (camera delivers 30 fps "
                             'but cv2 in-loop cap.read() retrieves every other '
                             'frame because MJPG CPU decode + imshow/waitKey '
                             'overlap with the camera cadence).')
    parser.add_argument('--minimal_cap', action='store_true',
                        help='Skip every cap.set() call (FOURCC, W/H, '
                             'BUFFERSIZE, AUTO_EXPOSURE, FPS). Opens the '
                             'camera exactly like DECA/demo_webcam does, with '
                             'driver defaults. Useful to bisect which cap.set '
                             'call is hurting a given camera. Overrides '
                             '--fourcc, --width, --height, --camera_fps, '
                             '--buffersize, --auto_exposure, --exposure.')
    parser.add_argument('--with_eye_pose', action='store_true',
                        help='Also estimate rot6d eyes_pose + blendshape '
                             'eyelids per frame via MediaPipe Tasks.')
    parser.add_argument('--with_oral_features', action='store_true',
                        help='Also log mouth-interior signals to the NDJSON '
                             'save stream (only meaningful together with '
                             '--save_path): raw MediaPipe 478 landmarks in '
                             'crop space (float16-safe lists), the 20-pt '
                             'inner-lip subset, 27 ARKit jaw/mouth '
                             'blendshapes, the 4x4 MediaPipe facial '
                             'transform, and SMIRK-driven 3D FLAME '
                             'MediaPipe landmarks (105 pts + 20-pt subset). '
                             'Uses run_mediapipe_full under the hood so it '
                             'is slightly more expensive than the default '
                             'landmark-only path.')
    parser.add_argument('--no_render', action='store_true',
                        help='Skip the FLAME mesh render (benchmark only).')
    parser.add_argument('--capture_only', action='store_true',
                        help='Skip MediaPipe + SMIRK + render entirely and '
                             'only capture -> (optional resize) -> display. '
                             'Used to measure the I/O ceiling (cap.read + '
                             'imshow + waitKey) independent of inference.')
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

    # ---- visualization (overlay / show_vertices) ---- #
    add_vertex_viz_args(parser)

    # ---- bbox stabilization (online only; offline is not meaningful live) ---- #
    parser.add_argument('--bbox_mode', type=str, default='online',
                        choices=['legacy', 'online'],
                        help='How to derive the per-frame face bbox. '
                             '"legacy" = min/max over every MediaPipe landmark '
                             'with no temporal filtering (same as the original '
                             'webcam demo). "online" = stable-landmark subset + '
                             'One-Euro low-pass on bbox size (O(1) state). '
                             'Default "online".')
    parser.add_argument('--bbox_scale', type=float, default=1.4,
                        help='Padding multiplier around the landmark bbox. '
                             'Same semantics as demo_video.py. Default 1.4.')
    parser.add_argument('--bbox_all_landmarks', action='store_true',
                        help='Use every MediaPipe landmark for the bbox '
                             'instead of the speech/blink-invariant subset. '
                             'Applies only to --bbox_mode online.')
    parser.add_argument('--bbox_size_calibration', type=float, default=None,
                        help='Multiplier applied to the stable-subset size to '
                             'match legacy crop extent. None = module default '
                             '(~1.55). Pass 1.0 to disable. Ignored under '
                             '--bbox_all_landmarks or --bbox_mode legacy.')
    parser.add_argument('--online_size_min_cutoff', type=float, default=1.0,
                        help='One-Euro min_cutoff (Hz) for bbox size filter. '
                             'Default 1.0.')
    parser.add_argument('--online_size_beta', type=float, default=0.02,
                        help='One-Euro beta (speed sensitivity) for bbox size '
                             'filter. Default 0.02.')
    parser.add_argument('--online_center_cutoff', type=float, default=None,
                        help='If set, also One-Euro-filter the bbox center with '
                             'this min_cutoff (Hz). Usually leave unset so head '
                             'translation tracks fast motion faithfully.')
    parser.add_argument('--online_center_beta', type=float, default=0.02,
                        help='One-Euro beta for bbox center filter (only used '
                             'when --online_center_cutoff is set). Default 0.02.')

    # ---- identity freeze (online warm-up only for live webcam) ---- #
    parser.add_argument('--freeze_shape', action='store_true',
                        help='Lock the predicted FLAME shape (identity) '
                             'parameters after a warm-up window: collect '
                             'shape_params for --freeze_shape_warmup_frames '
                             'frames, take their median, then override all '
                             'subsequent predictions with it. Useful when the '
                             'subject stays in frame long enough to calibrate.')
    parser.add_argument('--freeze_shape_warmup_frames', type=int, default=45,
                        help='Warm-up length (frames) for --freeze_shape. '
                             'Default 45 (~1.5 s at 30 fps).')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if str(device) != args.device:
        print(f'[demo_webcam] requested {args.device} but using {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    encoder = None
    if not args.capture_only:
        encoder = load_encoder(args.checkpoint, device)

    flame = None
    renderer = None
    need_flame = (not args.no_render and not args.capture_only) or (
        args.with_oral_features and not args.capture_only
    )
    if need_flame:
        # Lazy import so pure-benchmark mode works without FLAME assets.
        from src.FLAME.FLAME import FLAME
        flame = FLAME().to(device)
    if not args.no_render and not args.capture_only:
        from src.renderer.renderer import Renderer
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
    if is_webcam and not args.minimal_cap:
        # FOURCC must be set BEFORE width/height — V4L2 picks the format
        # first, then negotiates resolution within what the format supports.
        # Without MJPG, most UVC cams fall back to YUYV and clamp 720p to
        # ~5 fps because USB 2.0 bandwidth cannot carry raw 1280x720@30.
        if args.fourcc:
            fourcc_val = cv2.VideoWriter_fourcc(*args.fourcc.upper())
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.camera_fps is not None:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        # Only touch BUFFERSIZE when the user explicitly asks: setting 1
        # looks sensible (latest frame only) but on several V4L2 builds
        # it forces cap.read() to wait for a buffer flip every call and
        # halves the effective FPS (camera delivers 30fps, loop sees 15).
        if args.buffersize is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
        # Auto-exposure handling. V4L2 magic numbers: 1 = manual,
        # 3 = aperture priority (auto). We only call cap.set() for
        # manual mode; re-asserting auto mode is a no-op at best and
        # on some drivers re-negotiates the UVC control stream and
        # drops the reported frame rate, so leave the camera default
        # untouched when --auto_exposure auto.
        if args.auto_exposure == 'manual':
            ok1 = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            if args.exposure is not None:
                ok2 = cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
            else:
                ok2 = True
            if not (ok1 and ok2):
                print('[demo_webcam] WARN: cap.set for exposure returned False; '
                      'driver may have ignored it.')
    elif is_webcam and args.minimal_cap:
        print('[demo_webcam] --minimal_cap: skipping every cap.set() call '
              '(driver defaults only).')
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)) if actual_fourcc else ''
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'[demo_webcam] source={source_handle!r} mode={"webcam" if is_webcam else "ideal-video"}  '
          f'fourcc={fourcc_str or "?"}  size={actual_w}x{actual_h}  reported_fps={actual_fps:.1f}')
    if is_webcam:
        # Echo back the exposure controls the driver is actually using, so
        # a silently-ignored cap.set() or stuck AE is visible at a glance.
        ae_val = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        exp_val = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f'[demo_webcam] auto_exposure_ctrl={ae_val}  exposure_ctrl={exp_val}  '
              f'(requested: auto_exposure={args.auto_exposure}, exposure={args.exposure})')

    capture_thread = None
    if is_webcam and args.capture_thread:
        capture_thread = CaptureThread(cap).start()
        print('[demo_webcam] --capture_thread: background thread draining cap.read()')

    save_fp = None
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or '.', exist_ok=True)
        save_fp = open(args.save_path, 'a', buffering=1)  # line-buffered
        if args.with_oral_features:
            # One-shot metadata line so consumers can align the per-frame
            # inner-mouth subsets with the canonical index arrays without
            # re-deriving them. Tagged with ``kind: meta`` so parsers that
            # expect per-frame records can ignore it.
            meta = {
                'kind': 'meta',
                'mediapipe_inner_mouth_indices': list(MP_INNER_MOUTH),
                'flame_mp_inner_mouth_positions': list(FLAME_MP_INNER_MOUTH),
                'mouth_blendshape_names': list(ARKIT_MOUTH_KEYS),
            }
            save_fp.write(json.dumps(meta) + '\n')

    os.makedirs(args.snapshot_dir, exist_ok=True)

    # ---- bbox stabilization state ---- #
    # "legacy" keeps fast_crop_face_bgr (raw all-landmark min/max, no filter).
    # "online" wires OnlineBBoxTracker. ``actual_fps`` is the camera's reported
    # rate (0 for ideal-video mode); we fall back to 30 Hz when unknown so the
    # One-Euro filter has a sensible freq estimate.
    bbox_tracker = None
    if not args.capture_only and args.bbox_mode == 'online':
        tracker_fps = actual_fps if actual_fps and actual_fps > 0 else 30.0
        bbox_tracker = OnlineBBoxTracker(
            fps=tracker_fps, image_size=224, scale=args.bbox_scale,
            use_stable_subset=not args.bbox_all_landmarks,
            size_calibration=args.bbox_size_calibration,
            size_min_cutoff=args.online_size_min_cutoff,
            size_beta=args.online_size_beta,
            center_min_cutoff=args.online_center_cutoff,
            center_beta=args.online_center_beta,
        )

    # ---- identity freeze state ---- #
    frozen_shape: torch.Tensor | None = None
    shape_buffer: list[torch.Tensor] = []

    fps_window = deque(maxlen=30)
    frame_idx = 0
    # Aggregate per-stage sums (seconds) for the final summary print.
    agg = {'cap': 0.0, 'mp': 0.0, 'pre': 0.0, 'enc': 0.0,
           'ren': 0.0, 'disp': 0.0, 'gui': 0.0}
    agg_face_ok = 0
    first_frame_seconds = None
    t_run0 = time.perf_counter()
    print('[demo_webcam] q: quit, s: snapshot')

    try:
        while True:
            t_frame0 = time.perf_counter()
            timings: dict[str, float] = {}

            # --- Capture (cap.read + optional flip) ---
            t_cap0 = time.perf_counter()
            if capture_thread is not None:
                ret, frame = capture_thread.read()
            else:
                ret, frame = cap.read()
            if not ret:
                print('[demo_webcam] frame read failed / EOF')
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)
            timings['cap'] = (time.perf_counter() - t_cap0) * 1000.0
            orig_h, orig_w = frame.shape[:2]

            face_ok = False
            mesh_img = None
            outputs = None
            ren_out = None
            cropped_bgr = None
            eyes_pose_np = None
            eyelids_np = None

            if not args.capture_only:
                # --- MediaPipe ---
                t_mp0 = time.perf_counter()
                mp_transform = None
                need_full_mp = args.with_eye_pose or args.with_oral_features
                if need_full_mp:
                    mp_result = run_mediapipe_full(frame, delegate=args.mp_delegate)
                    if mp_result is not None:
                        landmarks = mp_result['landmarks'][..., :2]
                        blendshapes = mp_result['blendshapes']
                        mp_transform = mp_result.get('transform')
                    else:
                        landmarks, blendshapes = None, None
                else:
                    landmarks_full = run_mediapipe(frame, delegate=args.mp_delegate)
                    landmarks = landmarks_full[..., :2] if landmarks_full is not None else None
                    blendshapes = None
                timings['mp'] = (time.perf_counter() - t_mp0) * 1000.0
                face_ok = landmarks is not None

                if face_ok:
                    # --- Preprocess (warp + cvtColor + to-device transfer) ---
                    t_pre0 = time.perf_counter()
                    crop_affine = None
                    if bbox_tracker is not None:
                        # OnlineBBoxTracker returns a skimage similarity tform;
                        # slice its forward 3x3 .params to a 2x3 float32 matrix
                        # so cv2.warpAffine can consume it (same fast path as
                        # fast_crop_face_bgr's internal call).
                        tform, _, _ = bbox_tracker.update(landmarks)
                        M = tform.params[:2, :].astype(np.float32)
                        crop_affine = M
                        cropped_bgr = cv2.warpAffine(
                            frame, M, (224, 224),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0),
                        )
                    else:
                        _crop_M: list = []
                        cropped_bgr = fast_crop_face_bgr(
                            frame, landmarks,
                            scale=args.bbox_scale, image_size=224,
                            matrix_out=_crop_M if args.with_oral_features else None,
                        )
                        if _crop_M:
                            crop_affine = _crop_M[0]
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    tensor = tensor.to(device, non_blocking=True)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    timings['pre'] = (time.perf_counter() - t_pre0) * 1000.0

                    # --- SMIRK encode ---
                    t_enc0 = time.perf_counter()
                    with torch.no_grad():
                        outputs = encoder(tensor)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    timings['enc'] = (time.perf_counter() - t_enc0) * 1000.0

                    # --- Shape freeze (online warm-up median) ---
                    # Update both ``shape_params`` (consumed by FLAME.forward)
                    # and ``shape`` (the FLARE-compatible alias read by the
                    # save path) so the NDJSON log matches the frozen mesh.
                    if args.freeze_shape:
                        if frozen_shape is not None:
                            outputs['shape_params'] = frozen_shape
                            outputs['shape'] = frozen_shape
                        else:
                            shape_buffer.append(outputs['shape_params'].detach())
                            if len(shape_buffer) >= args.freeze_shape_warmup_frames:
                                stacked = torch.cat(shape_buffer, dim=0)
                                frozen_shape = torch.median(stacked, dim=0).values.unsqueeze(0).to(device)
                                outputs['shape_params'] = frozen_shape
                                outputs['shape'] = frozen_shape
                                shape_buffer = []
                                print(f'[demo_webcam] --freeze_shape: identity locked after '
                                      f'{args.freeze_shape_warmup_frames} frames')

                    if args.with_eye_pose:
                        ep, el = estimate_eye_pose_and_eyelid(blendshapes, device='cpu')
                        eyes_pose_np = ep.squeeze(0).numpy()
                        eyelids_np = el.squeeze(0).numpy()

                    # --- Render (optional) ---
                    flame_mp_3d_np = None
                    if not args.no_render:
                        t_ren0 = time.perf_counter()
                        mesh_img, ren_out = render_mesh(flame, renderer, outputs, device)
                        timings['ren'] = (time.perf_counter() - t_ren0) * 1000.0

                    if args.with_oral_features and flame is not None:
                        # Need 3D FLAME-driven MediaPipe landmarks for the
                        # save record. The renderer also runs flame.forward
                        # internally, but we keep this separate so --no_render
                        # still produces the feature.
                        with torch.no_grad():
                            flame_out_oral = flame.forward(outputs)
                        flame_mp_3d_np = (
                            flame_out_oral['landmarks_mp'].detach().cpu().numpy().squeeze(0)
                        )

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
                        if args.with_oral_features:
                            if crop_affine is not None:
                                lmks_crop = apply_affine_2x3(landmarks, crop_affine)
                            else:
                                lmks_crop = np.asarray(landmarks, dtype=np.float32)
                            inner_mouth_2d = extract_inner_mouth_landmarks_2d(lmks_crop)
                            mouth_bs = extract_mouth_blendshapes(blendshapes)
                            transform = (
                                mp_transform if mp_transform is not None
                                else np.eye(4, dtype=np.float32)
                            )
                            record['mediapipe_landmarks_2d'] = lmks_crop.astype(np.float32).tolist()
                            record['mediapipe_landmarks_inner_mouth_2d'] = (
                                inner_mouth_2d.astype(np.float32).tolist()
                            )
                            record['mouth_blendshapes'] = mouth_bs.tolist()
                            record['mediapipe_transform'] = transform.astype(np.float32).tolist()
                            if flame_mp_3d_np is not None:
                                record['flame_landmarks_mp_3d'] = flame_mp_3d_np.tolist()
                                record['flame_landmarks_inner_mouth_3d'] = (
                                    flame_mp_3d_np[list(FLAME_MP_INNER_MOUTH), :].tolist()
                                )
                        save_fp.write(json.dumps(record) + '\n')

            # --- Display prep (resize + hstack + overlay) ---
            t_disp0 = time.perf_counter()
            display_h = 480
            scale = display_h / orig_h
            webcam_disp = cv2.resize(frame, (int(orig_w * scale), display_h))
            if mesh_img is not None:
                # Build the right panel at 224x224 in BGR, then resize once
                # for display. This matches demo_video.py's non-render_orig
                # behavior: overlay / vertex-scatter happens in crop space
                # (224x224) so the user sees the same visualization surface
                # regardless of webcam resolution.
                if args.overlay and cropped_bgr is not None:
                    right_panel_bgr = _alpha_blend_mesh_over_crop_bgr(
                        cropped_bgr, mesh_img, args.overlay_alpha,
                    )
                else:
                    right_panel_bgr = mesh_img.copy() if args.show_vertices else mesh_img
                if args.show_vertices and ren_out is not None:
                    crop_pixels = ndc_to_crop_pixels(
                        ren_out['transformed_vertices'], 224,
                    )
                    # BGR cyan = (255, 255, 0) — matches the RGB (0, 255, 255)
                    # used by demo_video.py after its RGB->BGR cvtColor step.
                    draw_vertex_points_bgr(
                        right_panel_bgr, crop_pixels,
                        color_bgr=(255, 255, 0),
                        radius=args.vertex_radius,
                        radius_rel=args.vertex_radius_rel,
                        stride=args.vertex_stride,
                    )
                mesh_disp = cv2.resize(right_panel_bgr, (display_h, display_h))
                canvas = np.hstack([webcam_disp, mesh_disp])
            else:
                canvas = webcam_disp

            t_frame_pre_gui = time.perf_counter() - t_frame0
            # FPS uses frame-over-frame wall time, so take it after gui
            # below; here we still need an estimate for the overlay.
            fps_estimate = len(fps_window) / max(sum(fps_window), 1e-6) if fps_window else 0.0

            extras = []
            if args.with_eye_pose and eyes_pose_np is not None:
                extras.append(
                    f'eyes_pose|L|≈{np.linalg.norm(eyes_pose_np[:6]):.2f}  '
                    f'blink L/R: {eyelids_np[0]:.2f}/{eyelids_np[1]:.2f}'
                )
            draw_overlay(
                canvas, fps_estimate, timings, face_ok,
                mp_delegate=args.mp_delegate, enc_device=str(device),
                extras=extras,
            )
            timings['disp'] = (time.perf_counter() - t_disp0) * 1000.0

            # --- GUI (imshow + waitKey) ---
            t_gui0 = time.perf_counter()
            cv2.imshow(args.window, canvas)
            key = cv2.waitKey(1) & 0xFF
            timings['gui'] = (time.perf_counter() - t_gui0) * 1000.0

            t_frame = time.perf_counter() - t_frame0
            fps_window.append(t_frame)

            if first_frame_seconds is None:
                first_frame_seconds = t_frame
            else:
                for k, v in timings.items():
                    agg[k] += v / 1000.0
            if face_ok:
                agg_face_ok += 1

            if key == ord('q'):
                break
            if key == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                path = os.path.join(args.snapshot_dir, f'snapshot_{ts}.png')
                cv2.imwrite(path, canvas)
                print(f'[demo_webcam] saved {path}')

            frame_idx += 1
    finally:
        if capture_thread is not None:
            capture_thread.stop()
        cap.release()
        cv2.destroyAllWindows()
        if save_fp is not None:
            save_fp.close()
        t_run = time.perf_counter() - t_run0
        e2e_fps = (frame_idx / t_run) if t_run > 0 else 0.0
        # Averages exclude the first frame (cuDNN autotune, MediaPipe JIT,
        # webcam auto-exposure ramp, first FLAME/Renderer allocation).
        bench_frames = max(0, frame_idx - 1)

        def _avg_ms(key):
            return (1000.0 * agg[key] / bench_frames) if bench_frames else 0.0

        stages = ('cap', 'mp', 'pre', 'enc', 'ren', 'disp', 'gui')
        measured_sum_ms = sum(_avg_ms(k) for k in stages)
        first_ms = (first_frame_seconds * 1000.0) if first_frame_seconds is not None else 0.0
        print(f'[demo_webcam] processed {frame_idx} frames in {t_run:.2f}s '
              f'(valid={agg_face_ok}, end_to_end_fps={e2e_fps:.1f})')
        print(f'[demo_webcam] first-frame cost (excluded from averages): {first_ms:.1f}ms')
        print('[demo_webcam] avg-per-frame  ' + '  '.join(
            f'{k}={_avg_ms(k):.1f}ms' for k in stages
        ) + f'  (sum={measured_sum_ms:.1f}ms)')
        print(f'[demo_webcam] mp_delegate={args.mp_delegate}  enc_device={str(device)}  '
              f'mode={"webcam" if is_webcam else "ideal-video"}  '
              f'capture_only={args.capture_only}')


if __name__ == '__main__':
    main()
