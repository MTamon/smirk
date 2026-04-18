"""Eye-pose and eyelid estimator from MediaPipe ARKit blendshapes.

This module is SMIRK's standalone counterpart to FLARE's
``flare/utils/face_detect.py::FaceDetector.detect_eye_pose`` — it derives
the 12D ``eyes_pose`` (6D rotation representation for left + right eyeball)
and 2D ``eyelids`` that FlashAvatar's FLAME ``forward`` expects but SMIRK's
``SmirkEncoder`` does not produce.

The mapping follows the standard ARKit → FLAME eye convention:

    left eye pitch  = eyeLookDownLeft  - eyeLookUpLeft
    left eye yaw    = eyeLookInLeft    - eyeLookOutLeft
    right eye pitch = eyeLookDownRight - eyeLookUpRight
    right eye yaw   = eyeLookOutRight  - eyeLookInRight   # mirrored

Each (pitch, yaw, 0) axis-angle is converted to a 3×3 rotation matrix via
Rodrigues and the first two columns are flattened to the 6D representation
(Zhou et al. 2019). Left (6) and right (6) are concatenated into a (12,)
``eyes_pose`` vector. Eyelids come from ``eyeBlinkLeft`` / ``eyeBlinkRight``.

All outputs are returned as torch tensors with a leading batch dim so they
can be stacked across frames without reshaping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


ARKIT_EYE_KEYS = (
    'eyeLookUpLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft',
    'eyeLookUpRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight',
    'eyeBlinkLeft', 'eyeBlinkRight',
)
"""ARKit blendshape category names consumed by this estimator."""

_MAX_PITCH = 0.6  # radians (~34deg); ARKit scores are already in [0,1].
_MAX_YAW = 0.6


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues formula. axis_angle: (3,) float. Returns (3,3) float."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = axis_angle / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float32)
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def _matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    """First two columns, column-major flatten. (3,3) -> (6,)."""
    return mat[:, :2].T.reshape(-1).astype(np.float32)


def estimate_eye_pose_and_eyelid(
    blendshapes: Optional[dict],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (eyes_pose (1,12), eyelids (1,2)) from an ARKit blendshape dict.

    Args:
        blendshapes: dict[str, float] of ARKit blendshape scores as returned
            by ``utils.mediapipe_utils.run_mediapipe_full``. May be None or
            missing some keys — absent keys are treated as 0.
        device: torch device to place outputs on. Defaults to CPU.

    Returns:
        eyes_pose: (1, 12) float tensor — left 6D ++ right 6D rot6d.
        eyelids:   (1, 2)  float tensor — [blink_left, blink_right] in [0, 1].

    When ``blendshapes`` is None the function returns identity 6D rotations
    (both eyes looking forward) and zero eyelids so the caller always gets
    well-typed tensors.
    """
    device = device or torch.device('cpu')
    bs = blendshapes or {}

    def get(key: str) -> float:
        return float(bs.get(key, 0.0))

    left_pitch = (get('eyeLookDownLeft') - get('eyeLookUpLeft')) * _MAX_PITCH
    left_yaw = (get('eyeLookInLeft') - get('eyeLookOutLeft')) * _MAX_YAW
    right_pitch = (get('eyeLookDownRight') - get('eyeLookUpRight')) * _MAX_PITCH
    # Right eye yaw is mirrored relative to left so both eyes converge
    # towards the screen centre when the subject looks inward.
    right_yaw = (get('eyeLookOutRight') - get('eyeLookInRight')) * _MAX_YAW

    left_aa = np.array([left_pitch, left_yaw, 0.0], dtype=np.float32)
    right_aa = np.array([right_pitch, right_yaw, 0.0], dtype=np.float32)

    left_6d = _matrix_to_rotation_6d(_axis_angle_to_matrix(left_aa))
    right_6d = _matrix_to_rotation_6d(_axis_angle_to_matrix(right_aa))
    eyes_pose = np.concatenate([left_6d, right_6d], axis=0)[None, :]  # (1,12)

    eyelids = np.array([[get('eyeBlinkLeft'), get('eyeBlinkRight')]], dtype=np.float32)

    return (
        torch.from_numpy(eyes_pose).to(device),
        torch.from_numpy(eyelids).to(device),
    )


def identity_eye_pose(device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback when no face is detected: identity rotations + zero eyelids."""
    return estimate_eye_pose_and_eyelid(None, device=device)
