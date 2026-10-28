import math
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision  # noqa: F401 (registers vision tasks)

# CONFIG
MODEL_PATH = "assets/models/face_landmarker.task"
IMAGE_PATH = r"assets/images/brown_girl.jpg"
# IMAGE_PATH = r"C:\Users\User\Downloads\passport_profile.jpg"

# pip list | findstr opencv

MAX_TILT_DEG = 2            # max allowed head roll (degrees)
EYE_FACE_RATIO_RANGE = (0.40, 0.52)  # IPD / face width; outside => likely turned
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.18, 0.60   # face box area / image area
MIN_WIDTH_RATIO, MAX_WIDTH_RATIO = 0.40, 0.70  # face box width / image width
PAD = 15

LEFT_IRIS, RIGHT_IRIS = 468, 473  # image-left and image-right iris centers

# activate v-env
# .\.venv\Scripts\Activate.ps1


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    # ---------- MODEL ----------
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=4,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # ---------- LOAD IMAGE ----------
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        sys.exit(f"No image found at: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    detector.close()

    # ---------- FACE COUNT ----------
    face_count = len(result.face_landmarks)
    print(f"Faces detected: {face_count}")
    if face_count == 0:
        sys.exit("Rejected: No face detected")
    if face_count > 1:
        sys.exit("Rejected: More than 1 face detected")

    landmarks = result.face_landmarks[0]
    if len(landmarks) <= RIGHT_IRIS:
        sys.exit("Rejected: Model did not return iris landmarks (need 478 points)")

    img_h, img_w, _ = image_rgb.shape
    image_area = img_h * img_w
    print(f"Image size: {image_area}")

    # ---------- FACE BOX (clamped to image) ----------
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min_px = clamp(int(min(xs) * img_w), 0, img_w - 1)
    x_max_px = clamp(int(max(xs) * img_w), 0, img_w - 1)
    y_min_px = clamp(int(min(ys) * img_h), 0, img_h - 1)
    y_max_px = clamp(int(max(ys) * img_h), 0, img_h - 1)

    face_w = x_max_px - x_min_px
    face_h = y_max_px - y_min_px
    face_size = face_w * face_h
    print(f"Face size : {face_size}")
    if face_w <= 0 or face_h <= 0:
        sys.exit("Rejected: Invalid face box")

    face_image_ratio = (face_size) / image_area
    vertical_ratio = face_h / img_h
    horizontal_ratio = face_w / img_w
    print(f"area ratio: {face_image_ratio:.3f}")
    print(f"vertical ratio: {vertical_ratio:.3f}")
    print(f"horizontal ratio: {horizontal_ratio:.3f}")

    # ---------- EYES: tilt + turn ----------
    left_eye = landmarks[LEFT_IRIS]
    right_eye = landmarks[RIGHT_IRIS]

    delta_x = (right_eye.x - left_eye.x) * img_w
    delta_y = (right_eye.y - left_eye.y) * img_h
    # negative => right eye higher, positive => left eye higher, 0 => level
    eye_angle = math.degrees(math.atan2(delta_y, delta_x))
    eye_dist_px = math.hypot(delta_x, delta_y)
    eye_face_ratio = eye_dist_px / face_w
    print(f"eye angle: {eye_angle:.2f} deg")
    print(f"eye gap: {eye_dist_px:.2f}px")
    print(f"eye to face width ratio: {eye_face_ratio:.2f}")

    # ---------- CHECKS (collect all failures) ----------
    failures = []

    if abs(eye_angle) > MAX_TILT_DEG:
        failures.append("Head is tilted")

    lo, hi = EYE_FACE_RATIO_RANGE
    if not (lo <= eye_face_ratio <= hi):
        failures.append("Face likely turned away from camera (or measurement error)")

    if face_image_ratio< MIN_AREA_RATIO or horizontal_ratio < MIN_WIDTH_RATIO:
        failures.append("Face is too small")
    if face_image_ratio > MAX_AREA_RATIO or horizontal_ratio > MAX_WIDTH_RATIO:
        failures.append("Face is too large")

    passed = not failures
    if passed:
        print("Face passed")
    else:
        for f in failures:
            print(f"Rejected: {f}")

    # ---------- DISPLAY ----------
    color = (0, 200, 0) if passed else (0, 0, 255)
    cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, 2)
    for eye in (left_eye, right_eye):
        cv2.circle(image, (int(eye.x * img_w), int(eye.y * img_h)), 3, color, -1)
    cv2.putText(
        image,
        "PASSED" if passed else "REJECTED",
        (x_min_px, max(y_min_px - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    # shrink large images so the window fits on screen
    max_side = 900
    scale = max_side / max(img_h, img_w)
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cv2.imshow("Passport Validation Check", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()