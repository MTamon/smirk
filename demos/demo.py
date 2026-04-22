import os
import sys

# Allow `from src.*` / `from utils.*` to resolve when the script is
# invoked from anywhere (e.g. via demos/run_demo.sh from the repo root).
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
from datasets.base_dataset import create_mask, mediapipe_indices
import torch.nn.functional as F


def _shift_image_2d(img_chw_tensor, dx_pixels, dy_pixels):
    """Translate a (1, C, H, W) torch image by (dx, dy) pixels in screen space.

    Used by --align_to_input to cancel the per-frame screen-space drift that
    appears because SMIRK's weak-perspective `cam` only shifts in 2D while
    FLAME LBS rotates the mesh around the neck joint J_root. The encoder's
    saved (s, tx, ty) is left untouched; only the rasterized image is shifted
    so the mesh overlays the input face during preview.
    """
    _, _, h, w = img_chw_tensor.shape
    img_np = (img_chw_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    M = np.float32([[1.0, 0.0, float(dx_pixels)], [0.0, 1.0, float(dy_pixels)]])
    shifted = cv2.warpAffine(
        img_np, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return (
        torch.from_numpy(shifted).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ).to(img_chw_tensor.device)


def _alignment_shift_pixels(rendered_landmarks_mp_ndc, input_landmarks_mp_pixel, image_size):
    """Compute (dx, dy) in pixels to align rendered MP landmarks centroid to input."""
    rendered = rendered_landmarks_mp_ndc.squeeze(0).detach().cpu().numpy()
    rendered_pixel = (rendered + 1.0) * 0.5 * image_size
    input_subset = input_landmarks_mp_pixel[mediapipe_indices, :2]
    return input_subset.mean(axis=0) - rendered_pixel.mean(axis=0)


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
    parser.add_argument('--align_to_input', action='store_true',
                        help='Visualization-only 2D shift of the rendered mesh so its '
                             'MediaPipe-landmark centroid matches the input crop. Cancels '
                             'screen-space drift from SMIRK rotating the FLAME mesh around '
                             'J_root under a weak-perspective camera. Saved FLAME parameters '
                             'are unaffected; only the displayed mesh moves. Requires --crop.')

    args = parser.parse_args()

    image_size = 224
    

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

    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    # check if input is an image or a video or webcam or directory
    

    
    image = cv2.imread(args.input_path)
    orig_image_height, orig_image_width, _ = image.shape

    kpt_mediapipe = run_mediapipe(image)

    # crop face if needed
    if args.crop:
        if (kpt_mediapipe is None):
            print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
            exit()
        
        kpt_mediapipe = kpt_mediapipe[..., :2]

        tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=image_size)
        
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


    flame_output = flame.forward(outputs)
    renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                        landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
    
    rendered_img = renderer_output['rendered_img']

    # Visualization-only translation. Keep the un-shifted ``rendered_img``
    # for the smirk_generator path so its 6-channel input stays consistent
    # with how the network was trained.
    if args.align_to_input:
        if not args.crop:
            raise ValueError('--align_to_input requires --crop (input MediaPipe '
                             'landmarks must live in the same 224 crop frame as '
                             'the rendered mesh).')
        shift = _alignment_shift_pixels(
            renderer_output['landmarks_mp'],
            cropped_kpt_mediapipe,
            image_size,
        )
        rendered_img_display = _shift_image_2d(rendered_img, shift[0], shift[1])
    else:
        rendered_img_display = rendered_img

    if args.render_orig:
        if args.crop:
            rendered_img_numpy = (rendered_img_display.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)
            rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
            # back to pytorch to concatenate with full_image
            rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
        else:
            rendered_img_orig = F.interpolate(rendered_img_display, (orig_image_height, orig_image_width), mode='bilinear').cpu()

        full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
        grid = torch.cat([full_image, rendered_img_orig], dim=3)
    else:
        grid = torch.cat([cropped_image, rendered_img_display], dim=3)


    # ---- create the neural renderer reconstructed img ---- #
    if args.use_smirk_generator:
        if (kpt_mediapipe is None):
            print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting...')
            exit()

        mask_ratio_mul = 5
        mask_ratio = 0.01
        mask_dilation_radius = 10

        hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  

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
                reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
                # back to pytorch to concatenate with full_image
                reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
            else:
                reconstructed_img_orig = F.interpolate(reconstructed_img, (orig_image_height, orig_image_width), mode='bilinear').cpu()

            grid = torch.cat([grid, reconstructed_img_orig], dim=3)
        else:
            grid = torch.cat([grid, reconstructed_img], dim=3)

    grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
    grid_numpy = grid_numpy.astype(np.uint8)
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    image_name = args.input_path.split('/')[-1]

    cv2.imwrite(f"{args.out_path}/{image_name}", grid_numpy)

