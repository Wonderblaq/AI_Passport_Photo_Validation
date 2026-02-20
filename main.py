from math import atan2

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import  vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions
import math


# MODEL CONFIGURATION
model_path = "assets/models/face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisonRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options= BaseOptions(model_asset_path = model_path),
    running_mode= VisonRunningMode.IMAGE,
    num_faces= 4,
)
detector = FaceLandmarker.create_from_options(options)

# ---LOADING IMAGE WITH OPENCV---
image_path = "assets/images/brown_girl.jpg"
image = cv2.imread(image_path)
if image is None:
    raise ValueError("No image found")
    exit()

# convert image to rgb and then to mp image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(
    image_format= mp.ImageFormat.SRGB,
data = image_rgb)


result = detector.detect(mp_image)
print(len(result.face_landmarks))

face_count = len(result.face_landmarks)
if face_count != 1:
    print(" Rejected , More than 1 face detected!")
    exit()
print("Face Detected ")

landmarks = result.face_landmarks[0] # Gets first and only detected face
img_h, img_w, _ = image_rgb.shape  # get image size and ignore the channel
image_area = img_h * img_w
print(img_w, img_h)

# Extract all X and Y values in the face detected and their min and max
x_values = [lm.x for lm in landmarks]
y_values = [lm.y for lm in landmarks]

x_min, y_min= min(x_values), min(y_values)
x_max, y_max = max(x_values), max(y_values)

# convert the normalized points to pixels
x_min_px = int(x_min * img_w)
x_max_px = int (x_max * img_w)
y_min_px = int(y_min * img_h)
y_max_px = int(y_max * img_h)

# compute for face width and height and then compute for ratio of face area to image area
face_w = int((x_max_px - x_min_px))
face_h = int((y_max_px - y_min_px))
# print(f"face size : {face_w, face_h}")

 # Using face - image  to compute ratio
face_area = (face_w * face_h)
ratio = face_area / image_area
print(face_area, image_area)
print(f"ratio : {ratio}")

# using vertical height to calculate ratio
vertical_ratio = face_h / img_h
print(f"vertical ratio : {vertical_ratio}")
horizontal_ratio = (face_w / img_w)
print(f"horizontal ratio : {horizontal_ratio}")  

# Find midpoint of image and compare face position and distance for further decision 
face_center_x = x_min_px + ((x_max_px - x_min_px) / 2) # center of face at x-axis 
face_center_y = y_min_px + ((y_max_px - y_min_px) / 2)       # center of face at y-axis

# Get eye centers (using MediaPipe iris landmarks)
left_eye = landmarks[468]  # gets normalized points of eye
right_eye = landmarks[473]

# compute angle between eyes
delta_y = (right_eye.y - left_eye.y) * img_h # convert to pixels
delta_x = (right_eye.x - left_eye.x) * img_w
eye_angle = math.degrees(math.atan2(delta_y, delta_x))  # convert from rad to deg

# check InterPupilary Distance(IPD) to ensure face isnt turned
# check true distance between eyes
eye_dist_px = math.sqrt(delta_x**2 + delta_y**2)
print(f"eye gap : {eye_dist_px:.2f}")

# How much of the face width do the eyes take up?
eye_face_width_ratio = eye_dist_px/ face_w
print(f"eye to face width ratio :{eye_face_width_ratio:.2f}")
if abs(eye_angle) > 2:
    print("Rejected, Head is tilted")
elif 0.40 < eye_face_width_ratio > 0.50:
    print("Rejected: Face is likely turned away from the camera or Measurement error")
    exit()



# normalize and calculate vertical distance between eyes
eye_distance = abs((int(left_eye.y * img_w)) - (int(right_eye.y * img_w)))
eye_face_width_ratio = eye_distance / face_w



if ratio <= 0.3 and horizontal_ratio <= 0.4:
    print("Rejected , Face is too small")
    exit()
if ratio >= 0.6 and horizontal_ratio >= 0.7:
    print("Rejected , Face is too large")
print("Face passed ")







