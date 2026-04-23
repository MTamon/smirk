"""Oral / mouth-interior feature extraction for downstream avatars.

SMIRK's native outputs (shape / expression / pose / jaw / eyelid / cam)
cover the face surface but say nothing about the mouth interior. Avatars
like FlashAvatar, GeoAvatar, 3DRealHead, etc. need extra signal to
drive mouth-closure faces, teeth, and tongue -- the symptoms of missing
that signal are blurred / missing teeth in rendered output.

This module is the producer side of that extra signal. It is a
pure-Python helper (no torch import) that slices three things out of
what SMIRK's pipeline already computes every frame:

  * ``MP_INNER_MOUTH``:          inner-lip 2D subset of raw MediaPipe 478
                                 landmarks (crop-space; image-plane
                                 supervision for teeth / tongue regions).
  * ``ARKIT_MOUTH_KEYS``:        the jaw / mouth ARKit blendshapes from
                                 ``run_mediapipe_full`` (conditioning
                                 signal for renderer-side deformation).
  * ``FLAME_MP_INNER_MOUTH``:    positions within FLAME's 105-point
                                 ``landmarks_mp`` output that correspond
                                 to the same inner-mouth MediaPipe
                                 indices -- i.e. SMIRK-driven 3D anchors
                                 for inner-mouth geometry.

Downstream consumers slice their own tensors with these constants and
get a stable, canonical layout across checkpoints and releases.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Upper inner-lip contour, MediaPipe 478 indices (ordered, shared endpoints
# 78 and 308 are the mouth commissures).
MP_INNER_LIP_UPPER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)

# Lower inner-lip contour, MediaPipe 478 indices.
MP_INNER_LIP_LOWER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)

# Union of the two contours, with the shared commissures (78, 308) kept
# only once so the array has 20 unique entries. ``dict.fromkeys`` preserves
# insertion order on CPython 3.7+.
MP_INNER_MOUTH = tuple(dict.fromkeys(MP_INNER_LIP_UPPER + MP_INNER_LIP_LOWER))


# The 105 MediaPipe indices that FLAME2020 ships a barycentric embedding
# for (same list as ``datasets.base_dataset.mediapipe_indices``). FLAME's
# ``forward`` returns ``landmarks_mp`` with shape ``(B, 105, 3)`` in this
# order, so indexing ``landmarks_mp[..., FLAME_MP_INNER_MOUTH, :]`` gives
# the SMIRK-driven 3D inner-mouth landmarks directly.
_FLAME_MP_INDICES_105 = (
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336,
     46,  52,  53,  55,  63,  65,  66,  70, 105, 107,
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385,
    386, 387, 388, 390, 398, 466,   7,  33, 133, 144,
    145, 153, 154, 155, 157, 158, 159, 160, 161, 163,
    173, 246, 168,   6, 197, 195,   5,   4, 129,  98,
     97,   2, 326, 327, 358,   0,  13,  14,  17,  37,
     39,  40,  61,  78,  80,  81,  82,  84,  87,  88,
     91,  95, 146, 178, 181, 185, 191, 267, 269, 270,
    291, 308, 310, 311, 312, 314, 317, 318, 321, 324,
    375, 402, 405, 409, 415,
)
assert len(_FLAME_MP_INDICES_105) == 105

_MP_TO_FLAME_POS = {mp_idx: pos for pos, mp_idx in enumerate(_FLAME_MP_INDICES_105)}

# Positions of ``MP_INNER_MOUTH`` within the 105-entry list. Computed at
# import time so a typo in either list would surface immediately.
FLAME_MP_INNER_MOUTH = tuple(_MP_TO_FLAME_POS[i] for i in MP_INNER_MOUTH)


# ARKit blendshape category names produced by MediaPipe Tasks'
# FaceLandmarker that are relevant to the mouth / jaw. MediaPipe reports
# 52 ARKit blendshapes + a ``_neutral`` entry; this is the jaw* and mouth*
# subset in a canonical order so saved arrays have a stable layout.
#
# Note: Apple ARKit also ships ``tongueOut`` but MediaPipe's FaceLandmarker
# does not; it is intentionally omitted here. ``_neutral`` is also omitted
# because it is a summary signal, not a mouth shape.
ARKIT_MOUTH_KEYS = (
    'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
    'mouthClose', 'mouthFunnel', 'mouthPucker',
    'mouthLeft', 'mouthRight',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper',
    'mouthPressLeft', 'mouthPressRight',
    'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
)


def extract_mouth_blendshapes(blendshapes: Optional[dict]) -> np.ndarray:
    """Return an ordered ``(len(ARKIT_MOUTH_KEYS),)`` float32 array.

    Missing or ``None`` ``blendshapes`` yields a zero array so callers
    always get a fixed-size, well-typed result.
    """
    bs = blendshapes or {}
    return np.asarray(
        [float(bs.get(k, 0.0)) for k in ARKIT_MOUTH_KEYS],
        dtype=np.float32,
    )


def extract_inner_mouth_landmarks_2d(landmarks_478: np.ndarray) -> np.ndarray:
    """Slice the inner-mouth subset from raw MediaPipe 478 landmarks.

    Accepts shape ``(478, C)`` or ``(B, 478, C)`` for any trailing
    dimension C (typically 2 for pixel coords, 3 if z is included).
    Returns the subset in ``MP_INNER_MOUTH`` order.
    """
    arr = np.asarray(landmarks_478)
    idx = np.asarray(MP_INNER_MOUTH, dtype=np.int64)
    return arr[..., idx, :]


def extract_inner_mouth_landmarks_3d(landmarks_mp_105: np.ndarray) -> np.ndarray:
    """Slice the inner-mouth subset from FLAME's 105-point MediaPipe-embedded
    landmarks (driven by SMIRK's predicted FLAME parameters).

    Accepts shape ``(105, 3)`` or ``(B, 105, 3)``. Returns the subset in
    ``MP_INNER_MOUTH`` order (same order as the 2D slice for easy pairing).
    """
    arr = np.asarray(landmarks_mp_105)
    idx = np.asarray(FLAME_MP_INNER_MOUTH, dtype=np.int64)
    return arr[..., idx, :]


def apply_affine_2x3(points: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    """Apply a ``(2, 3)`` affine matrix to 2D points.

    Points accepted as ``(N, 2)``, ``(N, 3)``, or ``(..., 2+)``. Only the
    first two channels are transformed; any trailing channels (e.g. the
    MediaPipe z) are dropped because the affine is 2D-only.
    """
    pts = np.asarray(points, dtype=np.float32)
    xy = pts[..., :2]
    ones = np.ones(xy.shape[:-1] + (1,), dtype=np.float32)
    homo = np.concatenate([xy, ones], axis=-1)
    return homo @ matrix_2x3.T.astype(np.float32)
