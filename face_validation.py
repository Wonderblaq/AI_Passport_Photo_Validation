import math
import os
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
# Override with the FACE_MODEL_PATH env var if the model lives elsewhere.
MODEL_PATH = os.getenv(
    "FACE_MODEL_PATH",
    str(Path(__file__).parent / "assets" / "models" / "face_landmarker.task"),
)

MAX_TILT_DEG = 2                      # max allowed head roll (degrees)
EYE_FACE_RATIO_RANGE = (0.40, 0.52)   # eye gap / face width; outside => likely turned
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.18, 0.60    # face box area / image area
MIN_WIDTH_RATIO, MAX_WIDTH_RATIO = 0.40, 0.70  # face box width / image width
MAX_DETECT_SIDE = 1600                # downscale bigger images before detection (speed)

LEFT_IRIS, RIGHT_IRIS = 468, 473      # image-left and image-right iris centres

# ---------------- DETECTOR (loaded once, reused) ----------------
_detector = None
_lock = threading.Lock()  # MediaPipe detector is not guaranteed thread-safe


def load_detector():
    """Create the FaceLandmarker once. Call at app startup to fail fast."""
    global _detector
    with _lock:
        if _detector is None:
            options = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
            )
            _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


def close_detector():
    """Free the model. Call at app shutdown."""
    global _detector
    with _lock:
        if _detector is not None:
            _detector.close()
            _detector = None


# ---------------- HELPERS ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _result(valid, code, message, issues=None, metrics=None):
    return {
        "valid": valid,
        "code": code,
        "message": message,
        "issues": issues or [],     # every failed rule: [{"code", "message"}]
        "metrics": metrics or {},   # measurements, handy for tuning thresholds
    }


# ---------------- MAIN VALIDATION ----------------
def validate_face(image: np.ndarray) -> dict:
    """
    Validate a BGR image (as returned by cv2.imread / cv2.imdecode)
    using MediaPipe Face Landmarker. Returns a JSON-serialisable dict.
    """

    # ---------- CHECK IMAGE ----------
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return _result(
            False, "INVALID_IMAGE",
            "The uploaded file could not be read as a colour image.",
        )

    # Shrink very large images; landmarks are normalised so ratios are unaffected.
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > MAX_DETECT_SIDE:
        s = MAX_DETECT_SIDE / longest
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # OpenCV is BGR, MediaPipe expects RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # ---------- DETECT ----------
    detector = load_detector()
    with _lock:
        result = detector.detect(mp_image)

    # ---------- FACE COUNT ----------
    face_count = len(result.face_landmarks)
    if face_count == 0:
        return _result(False, "NO_FACE_DETECTED",
                       "No face detected. Please upload a clear photo.")
    if face_count > 1:
        return _result(False, "MULTIPLE_FACES_DETECTED",
                       "Multiple faces detected. Please upload a photo containing one person.")

    landmarks = result.face_landmarks[0]
    if len(landmarks) <= RIGHT_IRIS:
        return _result(False, "IRIS_LANDMARKS_MISSING",
                       "The model did not return iris landmarks (478 points expected).")

    img_h, img_w, _ = image_rgb.shape
    image_area = img_h * img_w

    # ---------- FACE BOX (clamped to image) ----------
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min_px = clamp(int(min(xs) * img_w), 0, img_w - 1)
    x_max_px = clamp(int(max(xs) * img_w), 0, img_w - 1)
    y_min_px = clamp(int(min(ys) * img_h), 0, img_h - 1)
    y_max_px = clamp(int(max(ys) * img_h), 0, img_h - 1)

    face_w = x_max_px - x_min_px
    face_h = y_max_px - y_min_px
    if face_w <= 0 or face_h <= 0:
        return _result(False, "INVALID_FACE_BOX",
                       "The detected face box is invalid. Please upload a clearer photo.")

    area_ratio = (face_w * face_h) / image_area
    vertical_ratio = face_h / img_h
    horizontal_ratio = face_w / img_w

    # ---------- EYES: tilt + turn ----------
    left_eye = landmarks[LEFT_IRIS]
    right_eye = landmarks[RIGHT_IRIS]

    delta_x = (right_eye.x - left_eye.x) * img_w
    delta_y = (right_eye.y - left_eye.y) * img_h
    # negative => right eye higher, positive => left eye higher, 0 => level
    eye_angle = math.degrees(math.atan2(delta_y, delta_x))
    eye_dist_px = math.hypot(delta_x, delta_y)
    eye_face_ratio = eye_dist_px / face_w

    metrics = {
        "image_width": img_w,
        "image_height": img_h,
        "area_ratio": round(area_ratio, 3),
        "vertical_ratio": round(vertical_ratio, 3),
        "horizontal_ratio": round(horizontal_ratio, 3),
        "eye_angle_deg": round(eye_angle, 2),
        "eye_face_ratio": round(eye_face_ratio, 3),
        "face_box": {"x_min": x_min_px, "y_min": y_min_px,
                     "x_max": x_max_px, "y_max": y_max_px},
    }

    # ---------- RULES (collect every failure) ----------
    issues = []

    if abs(eye_angle) > MAX_TILT_DEG:
        issues.append({"code": "HEAD_TILTED",
                       "message": "Head is tilted. Keep your head level."})

    lo, hi = EYE_FACE_RATIO_RANGE
    if not (lo <= eye_face_ratio <= hi):
        issues.append({"code": "FACE_NOT_FRONTAL",
                       "message": "Face appears turned away. Look straight at the camera."})

    if area_ratio < MIN_AREA_RATIO or horizontal_ratio < MIN_WIDTH_RATIO:
        issues.append({"code": "FACE_TOO_SMALL",
                       "message": "Face is too small. Move closer or crop tighter."})
    if area_ratio > MAX_AREA_RATIO or horizontal_ratio > MAX_WIDTH_RATIO:
        issues.append({"code": "FACE_TOO_LARGE",
                       "message": "Face is too large. Move back or leave more space."})

    if issues:
        first = issues[0]
        return _result(False, first["code"], first["message"], issues, metrics)

    return _result(True, "FACE_VALID", "Photo passed all checks.", [], metrics)