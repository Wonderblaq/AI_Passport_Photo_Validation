import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = r"C:\Users\PC\PycharmProjects\Face_Detetcor\assets\models\blaze_face_short_range.tflite"
image_path = r"C:\Users\PC\Downloads\group.jpeg"

# setup Detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

#  Load Image
image = cv2.imread(image_path)
if image is None:
    print("No Image Found")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# 3. Process Detection
result = detector.detect(mp_image)

if result.detections:
    detection = result.detections[0]
    bbox = detection.bounding_box

    # Use max() to prevent negative indexing crashes
    x = max(0, int(bbox.origin_x))
    y = max(0, int(bbox.origin_y))
    w = int(bbox.width)
    h = int(bbox.height)

    face = image[y:y + h, x:x + w]

    cv2.imwrite("cropped.jpg", face)
    cv2.imshow("Cropped Image", face)  # Show the 'face' variable

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected in the image.")
