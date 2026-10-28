from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from face_validation import close_detector, load_detector, validate_face

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_detector()      # load the model once at startup (fails fast if path is wrong)
    yield
    close_detector()     # free it on shutdown


app = FastAPI(title="Passport Photo Validator", lifespan=lifespan)

@app.get("/")
def home():
    return {
        "message": "GhIE Face Validation API is running."
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# Plain `def` (not `async def`) on purpose: FastAPI runs it in a worker thread,
# so the blocking MediaPipe/OpenCV work doesn't freeze the event loop.
@app.post("/validate-photo")
def validate_photo(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG or WebP images are accepted.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    # bytes -> numpy array -> decoded BGR image (None if it isn't a real image)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    result = validate_face(image)
    if result["code"] == "INVALID_IMAGE":
        raise HTTPException(status_code=400, detail=result["message"])
    return result