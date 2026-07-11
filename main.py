import io
from pathlib import Path
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


BASE_DIR = Path(__file__).resolve().parent
CONTROLNET_DIR = BASE_DIR / "ControlNet-v1-1-nightly"
if str(CONTROLNET_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLNET_DIR))

from annotator.dwpose.wholebody import Wholebody


# Register HEIF/HEIC support with Pillow before handling any uploaded images.
register_heif_opener()


app = FastAPI()
pose_engine = None


def get_pose_engine():
    global pose_engine
    if pose_engine is None:
        pose_engine = Wholebody()
    return pose_engine


def decode_image(image_bytes: bytes):
    """Decode an uploaded image into the BGR array expected by DWPose."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            # iPhone photos commonly store the display orientation in EXIF data.
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
            rgb_array = np.asarray(rgb_image)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode image file: {exc}",
        ) from exc

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


@app.get("/")
def root():
    return {"status": "ok", "service": "dwpose-ai-server"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/pose")
async def pose(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    min_height_ratio: float = 0.0,
    min_height_px: float = 0.0,
):
    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="Image file is required")

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        input_image = decode_image(image_bytes)
        keypoints, scores = get_pose_engine()(
            input_image,
            min_height_ratio=min_height_ratio,
            min_height_px=min_height_px,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pose inference failed: {exc}") from exc

    pose_people = []
    for person_index, person_keypoints in enumerate(keypoints):
        pose_people.append(
            {
                "personIndex": person_index,
                "keypoints": np.round(person_keypoints, 6).tolist(),
                "scores": np.round(scores[person_index], 6).tolist(),
            }
        )

    return {
        "poseAnalyzed": True,
        "posePersonCount": len(pose_people),
        "poseLandmarks": pose_people[0]["keypoints"] if pose_people else [],
        "posePeople": pose_people,
    }
