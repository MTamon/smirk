import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1,
                                    min_face_detection_confidence=0.1,
                                    min_face_presence_confidence=0.1
                                    )
detector = vision.FaceLandmarker.create_from_options(options)


def run_mediapipe(image):
    # print(image.shape)
    image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)


    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    if len (detection_result.face_landmarks) == 0:
        print('No face detected')
        return None

    face_landmarks = detection_result.face_landmarks[0]

    face_landmarks_numpy = np.zeros((478, 3))

    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

    return face_landmarks_numpy


def run_mediapipe_full(image):
    """Run MediaPipe FaceLandmarker and return landmarks + blendshapes + transform.

    Returns None if no face is detected. On success returns a dict:

        {
            "landmarks":   np.ndarray (478, 3),           # pixel-space (x,y) and z
            "blendshapes": dict[str, float],              # ARKit 52 + _neutral
            "transform":   np.ndarray (4, 4) or None,     # facial transformation matrix
            "width":       int,
            "height":      int,
        }

    Kept as a non-breaking addition next to ``run_mediapipe`` which returns
    landmarks only. Use this when eye-pose / blendshape-derived parameters
    are needed (e.g. SMIRK → FlashAvatar supplementation).
    """
    image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)

    detection = detector.detect(mp_image)
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
