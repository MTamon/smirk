"""Fast face-crop utilities using ``cv2.warpAffine``.

The existing demos (``demo.py`` / ``demo_video.py``) use
``skimage.transform.estimate_transform`` + ``skimage.transform.warp`` which
is a pure-Python / NumPy implementation and takes 5-15 ms per 224x224
crop. The equivalent ``cv2.getAffineTransform`` + ``cv2.warpAffine`` path
runs in <1 ms (OpenCV's C++ core dispatches to SIMD + IPP where available),
so swapping it in the performance-critical demos (``demo_save_flame.py``
and ``demo_webcam.py``) gives a large end-to-end FPS win.

The source/destination point layout matches ``demo.py::crop_face`` exactly
so the numerics are identical to what SMIRK's pretrained encoder was
trained on — only the library that applies the transform differs.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def build_affine_matrix(
    landmarks: np.ndarray,
    scale: float = 1.4,
    image_size: int = 224,
) -> np.ndarray:
    """Return a 2x3 float32 affine matrix that maps `landmarks` bbox to a
    centred square of side ``image_size`` with the given padding scale.

    Args:
        landmarks: (N, 2+) array of mediapipe landmarks in pixel coordinates.
        scale: padding multiplier around the landmark bbox (same meaning as
            ``demo.py::crop_face``).
        image_size: output crop side length.

    Returns:
        ndarray shape (2, 3), dtype float32. Suitable for ``cv2.warpAffine``.
    """
    left = float(np.min(landmarks[:, 0]))
    right = float(np.max(landmarks[:, 0]))
    top = float(np.min(landmarks[:, 1]))
    bottom = float(np.max(landmarks[:, 1]))

    old_size = (right - left + bottom - top) / 2.0
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0
    size = int(old_size * scale)

    src_pts = np.array([
        [center_x - size / 2, center_y - size / 2],
        [center_x - size / 2, center_y + size / 2],
        [center_x + size / 2, center_y - size / 2],
    ], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [0, image_size - 1],
        [image_size - 1, 0],
    ], dtype=np.float32)

    return cv2.getAffineTransform(src_pts, dst_pts)


def fast_crop_face_bgr(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    scale: float = 1.4,
    image_size: int = 224,
    matrix_out: Optional[list] = None,
) -> np.ndarray:
    """Crop ``frame_bgr`` to a ``(image_size, image_size, 3)`` BGR image.

    Args:
        frame_bgr: HxWx3 uint8 BGR input (as returned by ``cv2.VideoCapture``).
        landmarks: mediapipe 2D landmarks for the face.
        scale: bbox padding multiplier.
        image_size: output crop side.
        matrix_out: optional list; when provided the computed 2x3 affine
            matrix is appended so the caller can reuse it (e.g. to warp
            the rendered mesh back into the original image later).

    Returns:
        HxWx3 uint8 BGR crop. Use ``cv2.cvtColor(..., cv2.COLOR_BGR2RGB)``
        if your model expects RGB.
    """
    M = build_affine_matrix(landmarks, scale=scale, image_size=image_size)
    if matrix_out is not None:
        matrix_out.append(M)
    return cv2.warpAffine(
        frame_bgr, M, (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
