"""Temporal bbox stabilization for the FLAME face-crop pipeline.

The default ``crop_face`` in ``demos/demo_video.py`` derives the face bbox
from the ``min/max`` of every MediaPipe landmark, which means mouth opening,
blinks, and per-frame detection noise all leak into the crop ``size``. That
in turn drives the orthographic camera predicted by SMIRK, so the reprojected
FLAME mesh visibly "breathes" at the ears / scalp.

This module provides:

* ``extract_bbox_center_size`` — same center/size formula but optionally
  restricted to a stable landmark subset (eye corners, nose bridge, temples)
  that does not move with speech or blinking.
* ``OneEuroFilter1D`` — adaptive low-pass filter (CHI 2012) used for the
  real-time / webcam path. Strong smoothing when stationary, low lag when
  the subject moves quickly.
* ``fir_lowpass_offline`` — zero-phase symmetric FIR (window-designed via
  ``scipy.signal.firwin``) for the offline 2-pass path. Constant group delay
  is cancelled by a centred ``valid``-mode convolution with edge padding.
* ``OnlineBBoxTracker`` — per-frame wrapper that feeds (center, size) through
  One-Euro filters and returns the same ``skimage`` similarity transform the
  rest of the pipeline already consumes.
* ``build_similarity_tform`` — rebuild the ``crop_face`` similarity transform
  from a (center, size) pair so callers can drop in precomputed smoothed
  series without re-running the original formula.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from skimage.transform import estimate_transform


# MediaPipe FaceMesh landmark indices that do not move with speech / blink.
# Chosen to cover the face's horizontal and vertical extent using only points
# anchored to skull geometry:
#
#   33, 133   : right eye outer / inner corner
#   362, 263  : left eye inner / outer corner
#   1, 4, 5   : nose tip and bottom-of-nose
#   6, 168    : glabella (between brows, on bone) and nose root
#   195, 197  : nose bridge
#   234, 454  : right / left temple
#   127, 356  : just below the temples on the hairline
#
# Mouth, jaw outline, eyebrows, and eyelids are intentionally excluded.
STABLE_LANDMARK_INDICES = np.array(
    [33, 133, 362, 263, 1, 4, 5, 6, 168, 195, 197, 234, 454, 127, 356],
    dtype=np.int64,
)


def extract_bbox_center_size(
    landmarks: np.ndarray,
    use_stable_subset: bool = True,
) -> Tuple[np.ndarray, float]:
    """Return ``(center_xy, size)`` with the same formula as ``crop_face``.

    ``size`` is the average of bbox width and height (matches the legacy
    ``old_size = (right - left + bottom - top) / 2`` convention). ``center``
    is the bbox midpoint. When ``use_stable_subset`` is True, only the
    speech/blink-invariant landmarks are used; the resulting size no longer
    grows when the subject opens their mouth.
    """
    pts = landmarks[STABLE_LANDMARK_INDICES] if use_stable_subset else landmarks
    xs = pts[:, 0]
    ys = pts[:, 1]
    left = float(np.min(xs))
    right = float(np.max(xs))
    top = float(np.min(ys))
    bottom = float(np.max(ys))
    size = (right - left + bottom - top) / 2.0
    center = np.array(
        [(left + right) / 2.0, (top + bottom) / 2.0], dtype=np.float64,
    )
    return center, float(size)


def build_similarity_tform(
    center: np.ndarray,
    size: float,
    scale: float = 1.4,
    image_size: int = 224,
):
    """Build the ``skimage`` similarity transform used by ``crop_face``.

    Given a (center, size) pair this reproduces the three source / destination
    control points of the original ``crop_face`` so the returned ``tform``
    is drop-in compatible with existing call sites (``tform.inverse``,
    ``tform.params``).
    """
    padded = int(float(size) * float(scale))
    cx, cy = float(center[0]), float(center[1])
    half = padded / 2.0
    src_pts = np.array(
        [
            [cx - half, cy - half],
            [cx - half, cy + half],
            [cx + half, cy - half],
        ]
    )
    dst_pts = np.array(
        [
            [0, 0],
            [0, image_size - 1],
            [image_size - 1, 0],
        ]
    )
    return estimate_transform('similarity', src_pts, dst_pts)


class OneEuroFilter1D:
    """Scalar One-Euro filter (Casiez, Roussel, Vogel; CHI 2012).

    The cutoff frequency adapts to the instantaneous absolute speed of the
    signal: low cutoff when the subject is still (kills detector jitter) and
    high cutoff when the subject is moving quickly (keeps lag low).
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0


def fir_lowpass_offline(
    series: np.ndarray,
    fps: float,
    cutoff_hz: float,
    taps: int = 61,
    window: str = 'hamming',
) -> np.ndarray:
    """Zero-phase symmetric FIR low-pass for a 1-D series.

    The filter is designed with ``scipy.signal.firwin`` (linear phase, no
    ringing with Hamming / Hann windows). ``taps`` is forced odd so the
    group delay is an integer; the input is edge-padded by ``(taps-1)//2``
    samples on both sides and convolved in ``valid`` mode, which recovers
    the original length with zero net phase.

    If the series is shorter than the filter, the input is returned
    unchanged (no-op).
    """
    from scipy import signal

    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError('series must be 1-D')
    if s.size == 0:
        return s.copy()

    nyq = float(fps) / 2.0
    if cutoff_hz >= nyq or cutoff_hz <= 0.0:
        return s.copy()

    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    if taps >= s.size:
        return s.copy()

    coef = signal.firwin(taps, cutoff_hz / nyq, window=window)
    half = (taps - 1) // 2
    padded = np.concatenate(
        [np.full(half, s[0]), s, np.full(half, s[-1])]
    )
    return np.convolve(padded, coef, mode='valid')


class OnlineBBoxTracker:
    """Stateful per-frame bbox tracker for the real-time / webcam path.

    Pipeline:

        1. Extract raw ``(center, size)`` via ``extract_bbox_center_size``.
        2. Smooth ``size`` with a One-Euro filter (always on in online mode).
        3. Optionally smooth ``center`` with two independent One-Euro filters.
        4. Return the same similarity transform ``crop_face`` would have
           produced for the smoothed values.

    The tracker keeps only O(1) state (previous filtered value and derivative
    per filter), so it is safe to use inside a live capture loop.
    """

    def __init__(
        self,
        fps: float,
        image_size: int = 224,
        scale: float = 1.4,
        use_stable_subset: bool = True,
        size_min_cutoff: float = 1.0,
        size_beta: float = 0.02,
        center_min_cutoff: Optional[float] = None,
        center_beta: float = 0.02,
    ):
        self.fps = float(fps)
        self.image_size = int(image_size)
        self.scale = float(scale)
        self.use_stable_subset = bool(use_stable_subset)

        self._size_filter = OneEuroFilter1D(
            freq=self.fps, min_cutoff=size_min_cutoff, beta=size_beta,
        )
        self._smooth_center = center_min_cutoff is not None
        if self._smooth_center:
            self._cx_filter = OneEuroFilter1D(
                freq=self.fps, min_cutoff=center_min_cutoff, beta=center_beta,
            )
            self._cy_filter = OneEuroFilter1D(
                freq=self.fps, min_cutoff=center_min_cutoff, beta=center_beta,
            )

    def update(self, landmarks: np.ndarray):
        """Consume one frame's landmarks and return ``(tform, center, size)``."""
        center, size = extract_bbox_center_size(
            landmarks, use_stable_subset=self.use_stable_subset,
        )
        size_s = self._size_filter(size)
        if self._smooth_center:
            cx = self._cx_filter(float(center[0]))
            cy = self._cy_filter(float(center[1]))
            center_s = np.array([cx, cy], dtype=np.float64)
        else:
            center_s = center
        tform = build_similarity_tform(
            center_s, size_s, scale=self.scale, image_size=self.image_size,
        )
        return tform, center_s, size_s
