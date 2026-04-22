import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from utils.bbox_tracker import (
    OnlineBBoxTracker,
    build_similarity_tform,
    extract_bbox_center_size,
    fir_lowpass_offline,
)
from utils.vertex_viz import (
    add_vertex_viz_args,
    alpha_blend_mesh_over_input,
    crop_pixels_to_full_pixels,
    draw_vertex_points_tensor,
    ndc_to_crop_pixels,
)
from datasets.base_dataset import create_mask
import torch.nn.functional as F


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')
    add_vertex_viz_args(parser)

    parser.add_argument('--bbox_mode', type=str, default='online',
                        choices=['legacy', 'online', 'offline'],
                        help='How to derive the per-frame face bbox. '
                             '"legacy" reproduces the original min/max-over-all-landmarks '
                             'behavior (kept for A/B comparisons). "online" uses the '
                             'stable-landmark subset and One-Euro-filters the bbox size '
                             '(O(1) state, suitable for webcam/real-time). "offline" '
                             'runs a pre-pass over the whole video, applies a zero-phase '
                             'FIR low-pass to the raw size series, then processes frames '
                             'with the smoothed bbox. Default "online".')
    parser.add_argument('--bbox_scale', type=float, default=1.4,
                        help='Padding multiplier around the landmark bbox (same as '
                             'the legacy scale=1.4). Default 1.4.')
    parser.add_argument('--bbox_all_landmarks', action='store_true',
                        help='Use every MediaPipe landmark to derive the bbox (legacy '
                             'behavior). By default only a speech/blink-invariant subset '
                             '(eye corners, nose bridge, temples) is used so the bbox '
                             'size does not grow when the mouth opens.')
    parser.add_argument('--bbox_size_calibration', type=float, default=None,
                        help='Multiplier applied to the stable-subset size to compensate '
                             'for its smaller vertical extent (nose-only instead of '
                             'forehead-to-chin). Default is the value in '
                             'utils.bbox_tracker.STABLE_LANDMARK_SIZE_CALIBRATION '
                             '(~1.55) which makes the average crop match '
                             '--bbox_mode legacy. Pass 1.0 to disable. Ignored when '
                             '--bbox_all_landmarks or --bbox_mode legacy is set.')
    parser.add_argument('--online_size_min_cutoff', type=float, default=1.0,
                        help='One-Euro min_cutoff (Hz) for the bbox-size filter in '
                             '--bbox_mode online. Default 1.0.')
    parser.add_argument('--online_size_beta', type=float, default=0.02,
                        help='One-Euro beta (speed sensitivity) for the bbox-size '
                             'filter in --bbox_mode online. Default 0.02.')
    parser.add_argument('--online_center_cutoff', type=float, default=None,
                        help='If set, also One-Euro-filter the bbox center with this '
                             'min_cutoff (Hz) in --bbox_mode online. Leave unset to '
                             'pass center through unsmoothed (recommended; head '
                             'translation should follow fast motion faithfully).')
    parser.add_argument('--online_center_beta', type=float, default=0.02,
                        help='One-Euro beta for the bbox-center filter (only used when '
                             '--online_center_cutoff is set). Default 0.02.')
    parser.add_argument('--offline_size_cutoff', type=float, default=2.5,
                        help='Zero-phase FIR low-pass cutoff (Hz) for the bbox-size '
                             'series in --bbox_mode offline. The size signal carries '
                             'only camera-distance changes once the stable-landmark '
                             'subset is used, so 2-3 Hz is usually safe. Default 2.5.')
    parser.add_argument('--offline_size_taps', type=int, default=61,
                        help='Number of FIR taps for the offline size low-pass. Must be '
                             'shorter than the video. Default 61 (≈1 s group delay at '
                             '30 fps before edge compensation).')
    parser.add_argument('--offline_center_cutoff', type=float, default=None,
                        help='If set, also FIR-low-pass the bbox center in '
                             '--bbox_mode offline with this cutoff (Hz). Leave unset '
                             'to preserve raw center tracking.')
    parser.add_argument('--freeze_shape', action='store_true',
                        help='Lock the predicted FLAME shape (identity) parameters. '
                             'With --bbox_mode online: collect shape_params for '
                             '--freeze_shape_warmup_frames frames, take a median, then '
                             'override all subsequent predictions with it. With '
                             '--bbox_mode offline: run an extra pre-pass over the whole '
                             'video and use the global median. Opt-in — useful when '
                             'the subject is stationary enough to calibrate.')
    parser.add_argument('--freeze_shape_warmup_frames', type=int, default=45,
                        help='Warm-up length (frames) for --freeze_shape in online '
                             'mode. Default 45 (≈1.5 s at 30 fps).')

    args = parser.parse_args()

    input_image_size = 224
    

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

        # load also triangle probabilities for sampling points on the image
        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  


    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)


    cap = cv2.VideoCapture(args.input_path)

    if not cap.isOpened():
        print('Error opening video file')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # calculate size of output video
    if args.render_orig:
        out_width = video_width
        out_height = video_height
    else:
        out_width = input_image_size
        out_height = input_image_size

    if args.use_smirk_generator:
        out_width *= 3
    else:
        out_width *= 2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    cap_out = cv2.VideoWriter(f"{args.out_path}/{args.input_path.split('/')[-1].split('.')[0]}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (out_width, out_height))

    # ------------------------- bbox stabilization setup ------------------------- #
    # ``precomputed_tforms`` is populated by the offline pre-pass and indexed by
    # frame number in the main loop. ``bbox_tracker`` is the stateful One-Euro
    # wrapper used for --bbox_mode online. Only one of them is active at a time;
    # --bbox_mode legacy leaves both unset and calls ``crop_face`` directly.
    precomputed_tforms = None
    bbox_tracker = None
    use_stable_subset = not args.bbox_all_landmarks

    if args.crop and args.bbox_mode == 'offline':
        # Pass 1: read every frame, detect landmarks, collect raw (center, size).
        raw_centers = []
        raw_sizes = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kpt = run_mediapipe(frame)
            if kpt is None:
                print('Could not find landmarks during offline pass 1. Exiting...')
                exit()
            center_i, size_i = extract_bbox_center_size(
                kpt[..., :2], use_stable_subset=use_stable_subset,
                size_calibration=args.bbox_size_calibration,
            )
            raw_centers.append(center_i)
            raw_sizes.append(size_i)
        cap.release()

        raw_centers = np.stack(raw_centers, axis=0)
        raw_sizes = np.asarray(raw_sizes, dtype=np.float64)

        smoothed_sizes = fir_lowpass_offline(
            raw_sizes, fps=video_fps,
            cutoff_hz=args.offline_size_cutoff, taps=args.offline_size_taps,
        )
        smoothed_centers = raw_centers.copy()
        if args.offline_center_cutoff is not None and args.offline_center_cutoff > 0:
            smoothed_centers[:, 0] = fir_lowpass_offline(
                raw_centers[:, 0], fps=video_fps,
                cutoff_hz=args.offline_center_cutoff, taps=args.offline_size_taps,
            )
            smoothed_centers[:, 1] = fir_lowpass_offline(
                raw_centers[:, 1], fps=video_fps,
                cutoff_hz=args.offline_center_cutoff, taps=args.offline_size_taps,
            )

        precomputed_tforms = [
            build_similarity_tform(
                smoothed_centers[i], float(smoothed_sizes[i]),
                scale=args.bbox_scale, image_size=input_image_size,
            )
            for i in range(len(raw_sizes))
        ]

        # Reopen the capture for the main processing loop.
        cap = cv2.VideoCapture(args.input_path)

    elif args.crop and args.bbox_mode == 'online':
        bbox_tracker = OnlineBBoxTracker(
            fps=video_fps, image_size=input_image_size, scale=args.bbox_scale,
            use_stable_subset=use_stable_subset,
            size_calibration=args.bbox_size_calibration,
            size_min_cutoff=args.online_size_min_cutoff,
            size_beta=args.online_size_beta,
            center_min_cutoff=args.online_center_cutoff,
            center_beta=args.online_center_beta,
        )

    # --------------------------- shape freeze setup ----------------------------- #
    # frozen_shape holds a (1, 300) tensor that overrides the per-frame prediction.
    # shape_buffer accumulates shape_params during the online warm-up window.
    frozen_shape = None
    shape_buffer: list = []

    if args.crop and args.freeze_shape and args.bbox_mode == 'offline':
        # Pass 2 (offline + freeze): run SMIRK encoder on every smoothed crop to
        # collect shape_params across the whole video, then take the median.
        cap_shape = cv2.VideoCapture(args.input_path)
        shape_idx = 0
        with torch.no_grad():
            while True:
                ret, frame = cap_shape.read()
                if not ret:
                    break
                tform_i = precomputed_tforms[shape_idx]
                crop_i = warp(
                    frame, tform_i.inverse, output_shape=(input_image_size, input_image_size),
                    preserve_range=True,
                ).astype(np.uint8)
                crop_i = cv2.cvtColor(crop_i, cv2.COLOR_BGR2RGB)
                crop_t = torch.from_numpy(crop_i).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                crop_t = crop_t.to(args.device)
                out_i = smirk_encoder(crop_t)
                shape_buffer.append(out_i['shape_params'].detach().cpu())
                shape_idx += 1
        cap_shape.release()
        if shape_buffer:
            stacked = torch.cat(shape_buffer, dim=0)  # (N, 300)
            frozen_shape = torch.median(stacked, dim=0).values.unsqueeze(0).to(args.device)
            shape_buffer = []

    frame_idx = 0

    while True:
        ret, image = cap.read()

        if not ret:
            break
    
        kpt_mediapipe = run_mediapipe(image)

        # crop face if needed
        if args.crop:
            if (kpt_mediapipe is None):
                print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                exit()
            
            kpt_mediapipe = kpt_mediapipe[..., :2]

            if args.bbox_mode == 'offline':
                tform = precomputed_tforms[frame_idx]
            elif args.bbox_mode == 'online':
                tform, _, _ = bbox_tracker.update(kpt_mediapipe)
            else:  # 'legacy'
                tform = crop_face(image, kpt_mediapipe,
                                  scale=args.bbox_scale, image_size=input_image_size)

            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

            cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
            cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
        else:
            cropped_image = image
            cropped_kpt_mediapipe = kpt_mediapipe

        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(args.device)

        outputs = smirk_encoder(cropped_image)

        if args.freeze_shape:
            if frozen_shape is not None:
                outputs['shape_params'] = frozen_shape
            else:
                # Online warm-up median. Keep per-frame shape during the window,
                # switch to the frozen median once enough frames are buffered.
                shape_buffer.append(outputs['shape_params'].detach())
                if len(shape_buffer) >= args.freeze_shape_warmup_frames:
                    stacked = torch.cat(shape_buffer, dim=0)  # (N, 300)
                    frozen_shape = torch.median(stacked, dim=0).values.unsqueeze(0).to(args.device)
                    outputs['shape_params'] = frozen_shape
                    shape_buffer = []

        flame_output = flame.forward(outputs)
        renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                            landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])

        rendered_img = renderer_output['rendered_img']

        if args.render_orig:
            if args.crop:
                rendered_img_numpy = (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)
                rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                # back to pytorch to concatenate with full_image
                rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
            else:
                rendered_img_orig = F.interpolate(rendered_img, (video_height, video_width), mode='bilinear').cpu()

            full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
            right_panel = (
                alpha_blend_mesh_over_input(full_image, rendered_img_orig, args.overlay_alpha)
                if args.overlay else rendered_img_orig
            )
            if args.show_vertices:
                crop_pixels = ndc_to_crop_pixels(renderer_output['transformed_vertices'], input_image_size)
                if args.crop:
                    pixels_full = crop_pixels_to_full_pixels(crop_pixels, tform)
                else:
                    scale_x = video_width / float(input_image_size)
                    scale_y = video_height / float(input_image_size)
                    pixels_full = crop_pixels * np.array([scale_x, scale_y])
                right_panel = draw_vertex_points_tensor(
                    right_panel, pixels_full,
                    radius=args.vertex_radius,
                    radius_rel=args.vertex_radius_rel,
                    stride=args.vertex_stride,
                )
            grid = torch.cat([full_image, right_panel], dim=3)
        else:
            if args.overlay:
                right_panel = alpha_blend_mesh_over_input(
                    cropped_image, rendered_img, args.overlay_alpha
                )
            else:
                right_panel = rendered_img
            if args.show_vertices:
                crop_pixels = ndc_to_crop_pixels(renderer_output['transformed_vertices'], input_image_size)
                right_panel = draw_vertex_points_tensor(
                    right_panel, crop_pixels,
                    radius=args.vertex_radius,
                    radius_rel=args.vertex_radius_rel,
                    stride=args.vertex_stride,
                )
            grid = torch.cat([cropped_image, right_panel], dim=3)

        # ---- create the neural renderer reconstructed img ---- #
        if args.use_smirk_generator:
            if (kpt_mediapipe is None):
                print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting...')
                exit()

            mask_ratio_mul = 5
            mask_ratio = 0.01
            mask_dilation_radius = 10

            hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            tmask_ratio = mask_ratio * mask_ratio_mul # upper bound on the number of points to sample
            
            npoints, _ = masking_utils.mesh_based_mask_uniform_faces(renderer_output['transformed_vertices'], # sample uniformly from the mesh
                                                                    flame_faces=flame.faces_tensor,
                                                                    face_probabilities=face_probabilities,
                                                                    mask_ratio=tmask_ratio)
            
            pmask = torch.zeros_like(rendered_mask)                
            rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
            rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
            rbound =(npoints.size(1) * (1/mask_ratio_mul) * (rscale ** rsing)).long()

            for bi in range(npoints.size(0)):
                pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1
            
            hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32).unsqueeze(0).to(args.device)

            extra_points = cropped_image * pmask
            masked_img = masking_utils.masking(cropped_image, hull_mask, extra_points, mask_dilation_radius, rendered_mask=rendered_mask)

            smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)

            reconstructed_img = smirk_generator(smirk_generator_input)

            if args.render_orig:
                if args.crop:
                    reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
                    reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                    # back to pytorch to concatenate with full_image
                    reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
                else:
                    reconstructed_img_orig = F.interpolate(reconstructed_img, (video_height, video_width), mode='bilinear').cpu()

                grid = torch.cat([grid, reconstructed_img_orig], dim=3)
            else:
                grid = torch.cat([grid, reconstructed_img], dim=3)

        grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
        cap_out.write(grid_numpy)

        frame_idx += 1

    cap.release()
    cap_out.release()


