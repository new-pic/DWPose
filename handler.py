"""RunPod Serverless handler for DWPose inference only.

Supported input fields:
  - imageUrl: public HTTP(S) URL of an image
  - imageBase64: Base64 string or a data URL containing an image
  - minHeightRatio / min_height_ratio: optional value from 0.0 to 1.0
  - minHeightPx / min_height_px: optional non-negative pixel threshold
"""

print("[BOOT] handler process started", flush=True)

import traceback

print("[BOOT] traceback import complete", flush=True)

try:
    print("[BOOT] importing Python standard library modules", flush=True)
    import base64
    import binascii
    import logging
    import math
    import os
    import sys
    from pathlib import Path
    from typing import Any
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen
    print("[BOOT] Python standard library imports complete", flush=True)
except BaseException:
    print("[BOOT] Python standard library import failed", flush=True)
    traceback.print_exc()
    raise

try:
    print("[BOOT] importing numpy", flush=True)
    import numpy as np
    print(f"[BOOT] numpy import complete version={np.__version__}", flush=True)
except BaseException:
    print("[BOOT] numpy import failed", flush=True)
    traceback.print_exc()
    raise

try:
    print("[BOOT] importing runpod", flush=True)
    import runpod
    print(
        f"[BOOT] runpod import complete version={getattr(runpod, '__version__', 'unknown')}",
        flush=True,
    )
except BaseException:
    print("[BOOT] runpod import failed", flush=True)
    traceback.print_exc()
    raise

try:
    print("[BOOT] importing FastAPI HTTPException", flush=True)
    from fastapi import HTTPException
    print("[BOOT] FastAPI import complete", flush=True)
except BaseException:
    print("[BOOT] FastAPI import failed", flush=True)
    traceback.print_exc()
    raise

try:
    print("[BOOT] importing onnxruntime", flush=True)
    import onnxruntime as ort
    print(
        f"[BOOT] onnxruntime import complete version={ort.__version__}",
        flush=True,
    )
except BaseException:
    print("[BOOT] onnxruntime import failed", flush=True)
    traceback.print_exc()
    raise

try:
    print("[BOOT] importing DWPose application modules from /app/main.py", flush=True)
    from main import decode_image, get_pose_engine
    print("[BOOT] DWPose application imports complete", flush=True)
except BaseException:
    print("[BOOT] DWPose application import failed", flush=True)
    traceback.print_exc()
    raise


LOGGER = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
URL_TIMEOUT_SECONDS = 15

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = (
    APP_DIR / "ControlNet-v1-1-nightly" / "annotator" / "ckpts"
)
DETECTOR_MODEL_PATH = (CHECKPOINT_DIR / "yolox_l.onnx").resolve()
POSE_MODEL_PATH = (CHECKPOINT_DIR / "dw-ll_ucoco_384.onnx").resolve()


def _log_model_file(label: str, model_path: Path) -> None:
    exists = model_path.is_file()
    size = model_path.stat().st_size if exists else 0
    print(
        f"[BOOT] {label} path={model_path} exists={exists} size_bytes={size}",
        flush=True,
    )
    if not exists or size <= 0:
        raise FileNotFoundError(
            f"{label} ONNX model is missing or empty: {model_path}"
        )


# RunPod keeps this worker alive between jobs. Loading both ONNX models once at
# worker startup avoids paying the model initialization cost for every request.
try:
    print(f"[BOOT] cwd={os.getcwd()} app_dir={APP_DIR}", flush=True)
    print(
        "[BOOT] RunPod environment "
        f"queue_webhook_set={bool(os.environ.get('RUNPOD_WEBHOOK_GET_JOB'))} "
        f"ping_webhook_set={bool(os.environ.get('RUNPOD_WEBHOOK_PING'))} "
        f"pod_id_set={bool(os.environ.get('RUNPOD_POD_ID'))} "
        f"local_test={'--test_input' in sys.argv}",
        flush=True,
    )
    print(
        f"[BOOT] available ONNX providers={ort.get_available_providers()}",
        flush=True,
    )
    _log_model_file("detector_model", DETECTOR_MODEL_PATH)
    _log_model_file("pose_model", POSE_MODEL_PATH)
    print("[BOOT] loading DWPose ONNX sessions", flush=True)
    POSE_ENGINE = get_pose_engine()
    print(
        "[BOOT] detector session providers="
        f"{POSE_ENGINE.session_det.get_providers()}",
        flush=True,
    )
    print(
        "[BOOT] pose session providers="
        f"{POSE_ENGINE.session_pose.get_providers()}",
        flush=True,
    )
    print("[BOOT] DWPose ONNX sessions loaded", flush=True)
except BaseException:
    print("[BOOT] DWPose model initialization failed", flush=True)
    traceback.print_exc()
    raise


class InputValidationError(ValueError):
    """Raised when the Serverless job input is incomplete or invalid."""


def _decode_base64_image(value: str) -> bytes:
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or ";base64" not in header.lower():
            raise InputValidationError(
                "imageBase64 data URLs must include a ';base64' header."
            )

    payload = "".join(value.split())
    if not payload:
        raise InputValidationError("imageBase64 must not be empty.")

    try:
        payload += "=" * (-len(payload) % 4)
        image_bytes = base64.b64decode(
            payload, altchars=b"-_", validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise InputValidationError("imageBase64 is not valid Base64 data.") from exc

    if not image_bytes:
        raise InputValidationError("imageBase64 decoded to an empty image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InputValidationError(
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB input limit."
        )
    return image_bytes


def _download_image(url: str) -> bytes:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise InputValidationError("imageUrl must be a valid HTTP(S) URL.")

    request = Request(url, headers={"User-Agent": "dwpose-runpod-worker/1.0"})
    try:
        with urlopen(request, timeout=URL_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_IMAGE_BYTES:
                        raise InputValidationError(
                            "Image exceeds the "
                            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB input limit."
                        )
                except ValueError:
                    pass
            image_bytes = response.read(MAX_IMAGE_BYTES + 1)
    except InputValidationError:
        raise
    except HTTPError as exc:
        raise InputValidationError(
            f"imageUrl returned HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise InputValidationError(
            f"Unable to download imageUrl: {exc.reason}."
        ) from exc
    except OSError as exc:
        raise InputValidationError(
            f"Unable to download imageUrl: {exc}."
        ) from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InputValidationError(
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB input limit."
        )
    if not image_bytes:
        raise InputValidationError("imageUrl returned an empty response.")
    return image_bytes


def _get_image_bytes(job_input: dict[str, Any]) -> bytes:
    sources = [
        ("imageUrl", job_input.get("imageUrl")),
        ("imageBase64", job_input.get("imageBase64")),
    ]
    provided_sources = [(name, value) for name, value in sources if value is not None]

    if not provided_sources:
        raise InputValidationError(
            "Provide exactly one image input: imageUrl or imageBase64."
        )
    if len(provided_sources) > 1:
        raise InputValidationError("Provide only one of imageUrl or imageBase64.")

    source_name, source_value = provided_sources[0]
    if not isinstance(source_value, str) or not source_value.strip():
        raise InputValidationError(f"{source_name} must be a non-empty string.")

    if source_name == "imageUrl":
        return _download_image(source_value)
    return _decode_base64_image(source_value)


def _get_number(
    job_input: dict[str, Any],
    camel_case_key: str,
    snake_case_key: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    raw_value = job_input.get(camel_case_key, job_input.get(snake_case_key, 0.0))
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise InputValidationError(f"{camel_case_key} must be a number.")

    value = float(raw_value)
    if not math.isfinite(value) or value < minimum:
        raise InputValidationError(f"{camel_case_key} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise InputValidationError(f"{camel_case_key} must not exceed {maximum}.")
    return value


def _serialize_pose(keypoints: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
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


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Run DWPose on an image supplied as a URL or Base64 payload."""
    try:
        if not isinstance(job, dict):
            raise InputValidationError("Job must be a JSON object.")
        job_input = job.get("input", {})
        if not isinstance(job_input, dict):
            raise InputValidationError("input must be a JSON object.")

        image_bytes = _get_image_bytes(job_input)
        min_height_ratio = _get_number(
            job_input,
            "minHeightRatio",
            "min_height_ratio",
            minimum=0.0,
            maximum=1.0,
        )
        min_height_px = _get_number(
            job_input,
            "minHeightPx",
            "min_height_px",
            minimum=0.0,
        )

        input_image = decode_image(image_bytes)
        keypoints, scores = POSE_ENGINE(
            input_image,
            min_height_ratio=min_height_ratio,
            min_height_px=min_height_px,
        )
        return {"success": True, "result": _serialize_pose(keypoints, scores)}
    except InputValidationError as exc:
        return {
            "success": False,
            "error": {"code": "INVALID_INPUT", "message": str(exc)},
        }
    except HTTPException as exc:
        return {
            "success": False,
            "error": {"code": "IMAGE_DECODE_FAILED", "message": str(exc.detail)},
        }
    except Exception as exc:
        LOGGER.exception("DWPose inference failed")
        return {
            "success": False,
            "error": {"code": "INFERENCE_FAILED", "message": str(exc)},
        }


if __name__ == "__main__":
    print("[BOOT] registering RunPod Queue handler", flush=True)
    try:
        runpod.serverless.start({"handler": handler})
    except SystemExit as exc:
        # The RunPod SDK intentionally exits with code 0 after --test_input
        # local tests. Only non-zero exits are startup failures.
        if exc.code not in (None, 0):
            print(
                f"[BOOT] runpod.serverless.start exited code={exc.code}",
                flush=True,
            )
            traceback.print_exc()
        raise
    except BaseException:
        print("[BOOT] runpod.serverless.start failed", flush=True)
        traceback.print_exc()
        raise
