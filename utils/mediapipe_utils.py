"""MediaPipe FaceLandmarker wrappers with CPU/GPU delegate selection.

The FaceLandmarker is expensive to create (it loads the TFLite graph and
possibly an OpenGL context for GPU), so instances are cached per delegate
by ``get_detector``. A module-level CPU ``detector`` is kept for backwards
compatibility with callers that imported it directly.

Delegate selection:

    cpu  : BaseOptions.Delegate.CPU (default; always works)
    gpu  : BaseOptions.Delegate.GPU — requires a MediaPipe build compiled
           with GPU support. On Ubuntu desktop the pip-installed
           ``mediapipe`` wheel includes the OpenGL ES delegate, but actual
           GPU acceleration depends on the system EGL / GL drivers. When
           construction fails the factory falls back to CPU and emits a
           warning so the caller still gets a functional detector.
"""

import sys
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


_MODEL_ASSET_PATH = 'assets/face_landmarker.task'

_Delegate = python.BaseOptions.Delegate
_DELEGATE_MAP = {
    'cpu': _Delegate.CPU,
    'gpu': _Delegate.GPU,
}

_detector_cache: dict[str, vision.FaceLandmarker] = {}


def _build_detector(delegate: str) -> vision.FaceLandmarker:
    if delegate not in _DELEGATE_MAP:
        raise ValueError(f'Unknown delegate: {delegate!r}. Use "cpu" or "gpu".')
    base_options = python.BaseOptions(
        model_asset_path=_MODEL_ASSET_PATH,
        delegate=_DELEGATE_MAP[delegate],
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_detector(delegate: str = 'cpu') -> vision.FaceLandmarker:
    """Return a cached FaceLandmarker for the requested delegate.

    Falls back to CPU on GPU build failure and prints a one-line warning
    to stderr so the caller sees why the GPU path was not taken.
    """
    delegate = delegate.lower()
    if delegate in _detector_cache:
        return _detector_cache[delegate]
    try:
        instance = _build_detector(delegate)
    except Exception as e:
        if delegate == 'gpu':
            print(
                f'[mediapipe_utils] GPU delegate unavailable ({e.__class__.__name__}: {e}); '
                'falling back to CPU.',
                file=sys.stderr,
            )
            instance = _detector_cache.get('cpu') or _build_detector('cpu')
            _detector_cache['gpu'] = instance  # cache the fallback too
            _detector_cache.setdefault('cpu', instance)
            return instance
        raise
    _detector_cache[delegate] = instance
    return instance


# Backwards-compatible module-level CPU detector. Imported directly by older
# call sites (e.g. demo.py / demo_video.py / smirk_trainer.py).
detector = get_detector('cpu')


def run_mediapipe(image, delegate: Optional[str] = None):
    """Detect face landmarks. Returns (478, 3) ndarray in pixel space or None."""
    det = get_detector(delegate) if delegate is not None else detector

    image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)
    detection_result = det.detect(mp_image)

    if len(detection_result.face_landmarks) == 0:
        print('No face detected')
        return None

    face_landmarks = detection_result.face_landmarks[0]
    face_landmarks_numpy = np.zeros((478, 3))
    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [
            landmark.x * mp_image.width,
            landmark.y * mp_image.height,
            landmark.z,
        ]
    return face_landmarks_numpy


def run_mediapipe_full(image, delegate: Optional[str] = None):
    """Run MediaPipe FaceLandmarker and return landmarks + blendshapes + transform.

    Returns None if no face is detected. On success returns a dict:

        {
            "landmarks":   np.ndarray (478, 3),           # pixel-space (x,y) and z
            "blendshapes": dict[str, float],              # ARKit 52 + _neutral
            "transform":   np.ndarray (4, 4) or None,     # facial transformation matrix
            "width":       int,
            "height":      int,
        }
    """
    det = get_detector(delegate) if delegate is not None else detector

    image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)

    detection = det.detect(mp_image)
    if len(detection.face_landmarks) == 0:
        return None

    face_landmarks = detection.face_landmarks[0]
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i, lm in enumerate(face_landmarks):
        landmarks[i] = [lm.x * mp_image.width, lm.y * mp_image.height, lm.z]

    blendshapes: dict = {}
    if detection.face_blendshapes:
        for bs in detection.face_blendshapes[0]:
            blendshapes[bs.category_name] = float(bs.score)

    transform = None
    if detection.facial_transformation_matrixes:
        transform = np.asarray(detection.facial_transformation_matrixes[0], dtype=np.float32)

    return {
        'landmarks': landmarks,
        'blendshapes': blendshapes,
        'transform': transform,
        'width': mp_image.width,
        'height': mp_image.height,
    }
