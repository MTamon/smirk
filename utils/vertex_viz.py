"""Shared helpers for the mesh-overlay / vertex-scatter visualizations used by
all demos (``demo.py``, ``demo_video.py``, ``demo_webcam.py``).

Previously each demo carried a near-identical copy of these functions. Keeping
them in one module means a fix (e.g. tuning the default vertex-dot size so
5023 dots do not swamp a 224x224 panel) lands in every demo at once.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


def alpha_blend_mesh_over_input(input_img_chw, rendered_img_chw, alpha):
    """Alpha-blend ``rendered_img_chw`` on top of ``input_img_chw`` wherever
    the mesh is non-black; the rest of the input is untouched.

    Both tensors are (1, 3, H, W) in [0, 1]. Returns a tensor of the same
    shape. The renderer writes pure black (0, 0, 0) outside the rasterized
    mesh, so the mesh mask is ``rendered.sum(dim=1) > 0``.
    """
    mesh_mask = (rendered_img_chw.sum(dim=1, keepdim=True) > 1e-3).float()
    effective_alpha = mesh_mask * float(alpha)
    return input_img_chw * (1.0 - effective_alpha) + rendered_img_chw * effective_alpha


def ndc_to_crop_pixels(transformed_vertices_ndc, image_size):
    """(1, N, 2+) NDC → (N, 2) pixel coords in the ``image_size`` crop."""
    verts = transformed_vertices_ndc.squeeze(0).detach().cpu().numpy()[:, :2]
    return (verts + 1.0) * 0.5 * image_size


def crop_pixels_to_full_pixels(crop_pixels, tform):
    """Lift crop-space pixel coords to original-image pixel coords via
    ``tform.inverse`` (the same similarity transform used when cropping)."""
    homog = np.hstack([crop_pixels, np.ones((crop_pixels.shape[0], 1))])
    return np.dot(tform.inverse.params, homog.T).T[:, :2]


def crop_pixels_to_full_pixels_affine(crop_pixels, affine_2x3):
    """Lift crop-space pixel coords to original-image pixel coords when the
    crop was produced with a ``cv2.getAffineTransform`` 2x3 matrix (as
    ``utils.face_crop.fast_crop_face_bgr`` does).

    The forward affine maps source -> crop. To go from crop -> source we
    invert the 2x3 as a 3x3 (with [0,0,1] bottom row), drop the last row
    again, and apply to the homogeneous crop pixels.
    """
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = affine_2x3
    Minv = np.linalg.inv(M)[:2, :]
    homog = np.hstack([crop_pixels, np.ones((crop_pixels.shape[0], 1))])
    return (Minv @ homog.T).T


def draw_vertex_points_tensor(
    img_chw_tensor, pixels_xy,
    color_rgb=(0, 255, 255), radius=0, radius_rel=None, stride=1,
):
    """Draw points at ``pixels_xy`` on an (1, 3, H, W) tensor in [0, 1].

    Useful for inspecting head regions the default face-only rasterizer
    culls (ears, scalp, neck), since it projects the full 5023-vertex
    shape at the same 2D position as the mesh.

    ``color_rgb`` is interpreted in the channel order of ``img_chw_tensor``.
    The video demos keep an RGB intermediate and swap to BGR only at the
    VideoWriter step, so (0, 255, 255) renders as cyan in the final mp4.

    ``radius`` is the absolute radius in pixels. ``radius=0`` (the default)
    writes a single pixel directly — no ``cv2.circle`` / no LINE_AA — which
    is the crispest possible dot. ``cv2.circle(..., radius=1, LINE_AA)``
    bleeds across a ~3x3 neighbourhood which swamps a 224x224 panel when
    5023 dots are drawn, hence the default of 0.

    ``radius_rel``, if given, overrides ``radius`` and is interpreted as a
    fraction of ``min(H, W)``. Values that round down to 0 produce
    single-pixel dots.
    """
    _, _, h, w = img_chw_tensor.shape
    r = _resolve_radius(radius, radius_rel, h, w)

    img_np = (
        img_chw_tensor.squeeze(0).permute(1, 2, 0)
        .detach().cpu().numpy() * 255.0
    ).astype(np.uint8).copy()

    _scatter(img_np, pixels_xy, color_rgb, r, stride)

    return (
        torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ).to(img_chw_tensor.device)


def draw_vertex_points_bgr(
    img_bgr, pixels_xy,
    color_bgr=(255, 255, 0), radius=0, radius_rel=None, stride=1,
):
    """Draw points on a raw (H, W, 3) uint8 BGR image, in place-ish.

    Real-time demos keep a BGR numpy image all the way to ``cv2.imshow`` /
    ``cv2.VideoWriter`` and never round-trip through a torch tensor. This
    skips that round-trip. Returns the modified array (which may be the
    input itself).

    ``color_bgr`` defaults to cyan in BGR. Semantics of ``radius`` /
    ``radius_rel`` / ``stride`` match :func:`draw_vertex_points_tensor`.
    """
    if img_bgr.dtype != np.uint8:
        raise TypeError('draw_vertex_points_bgr expects uint8 BGR image')
    h, w = img_bgr.shape[:2]
    r = _resolve_radius(radius, radius_rel, h, w)
    _scatter(img_bgr, pixels_xy, color_bgr, r, stride)
    return img_bgr


def _resolve_radius(radius, radius_rel, h, w):
    if radius_rel is not None:
        r = int(round(float(radius_rel) * min(h, w)))
    else:
        r = int(radius)
    return max(0, r)


def _scatter(img_np, pixels_xy, color, radius, stride):
    h, w = img_np.shape[:2]
    pts = pixels_xy
    if stride > 1:
        pts = pts[::stride]

    if radius == 0:
        colour_arr = np.array([int(v) for v in color], dtype=np.uint8)
        for x, y in pts:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                img_np[yi, xi] = colour_arr
    else:
        colour = tuple(int(v) for v in color)
        for x, y in pts:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(img_np, (xi, yi), radius, colour, -1,
                           lineType=cv2.LINE_AA)


def add_vertex_viz_args(parser, default_radius: int = 0):
    """Attach the common ``--overlay`` / ``--show_vertices`` / vertex-dot
    argparse flags to ``parser``. Centralized so demos stay in sync.

    ``default_radius=0`` (single-pixel direct write) is the new default to
    keep 5023 dots from visually swamping a 224x224 panel. Pass a larger
    value only if your demo is output at a noticeably higher resolution.
    """
    parser.add_argument(
        '--overlay', action='store_true',
        help='Draw the rendered mesh alpha-blended on top of the input '
             'frame instead of side-by-side. Lets you judge fit quality '
             'directly.',
    )
    parser.add_argument(
        '--overlay_alpha', type=float, default=0.55,
        help='Alpha for the mesh in --overlay mode (0=input only, '
             '1=mesh only). Default 0.55.',
    )
    parser.add_argument(
        '--show_vertices', action='store_true',
        help='Draw all FLAME vertices as colored dots on the right / mesh '
             'panel. Lets you see head regions the default face-only '
             'rasterizer omits (ears, scalp, neck). Combine with --overlay '
             'to inspect the full predicted shape against the real face.',
    )
    parser.add_argument(
        '--vertex_radius', type=int, default=default_radius,
        help='Absolute radius (px) for each vertex dot. Default '
             f'{default_radius}. 0 writes a true single-pixel dot (no '
             'anti-aliasing), which is the crispest option on 224x224 or '
             'otherwise low-resolution panels where LINE_AA circles with '
             'radius=1 look like ~3x3 blobs. Larger values are useful when '
             'rendering at full video resolution via --render_orig. Ignored '
             'when --vertex_radius_rel is set.',
    )
    parser.add_argument(
        '--vertex_radius_rel', type=float, default=None,
        help='Radius as a fraction of min(panel_h, panel_w). E.g. 0.001 on '
             'a 1080p panel → ~1 px; 0.0005 → single-pixel dot. Overrides '
             '--vertex_radius when set. Recommended when comparing across '
             'resolutions.',
    )
    parser.add_argument(
        '--vertex_stride', type=int, default=1,
        help='Stride when sampling the ~5023 FLAME vertices '
             '(1 = every vertex; 2 = every other; etc.). Default 1.',
    )
