from pathlib import Path
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile


BASE_DIR = Path(__file__).resolve().parent
CONTROLNET_DIR = BASE_DIR / "ControlNet-v1-1-nightly"
if str(CONTROLNET_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLNET_DIR))

from annotator.dwpose.wholebody import Wholebody


app = FastAPI()
pose_engine = None


def get_pose_engine():
    global pose_engine
    if pose_engine is None:
        pose_engine = Wholebody()
    return pose_engine


def decode_image(image_bytes: bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image file")
    return image


@app.get("/")
def root():
    return {"status": "ok", "service": "dwpose-ai-server"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/pose")
async def pose(image: UploadFile | None = File(None), file: UploadFile | None = File(None)):
    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="Image file is required")

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        input_image = decode_image(image_bytes)
        keypoints, scores = get_pose_engine()(input_image)
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
