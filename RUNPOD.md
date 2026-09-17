# RunPod Serverless Queue Endpoint

This worker runs **DWPose only**. It does not package or call RMBG-2.0.

## Deploy

1. In RunPod Serverless, choose **New Endpoint** and then **Deploy from GitHub**
   (or build and push the Docker image yourself).
2. Select this repository. RunPod will use the root `Dockerfile` and start
   `handler.py`, which calls `runpod.serverless.start({"handler": handler})`.
3. Choose a GPU worker. The image uses CUDA 12.4 and `onnxruntime-gpu`; DWPose
   falls back to the CPU provider only when no CUDA provider is available.

To build an image locally on Apple Silicon, target RunPod's AMD64 workers:

```bash
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/dwpose-runpod:latest .
docker push YOUR_DOCKERHUB_USER/dwpose-runpod:latest
```

## Job input

Send exactly one image source in the request's `input` object. `imageBase64`
accepts either raw Base64 or a `data:image/...;base64,...` URL. HEIC is
supported.

```json
{
  "input": {
    "imageUrl": "https://example.com/photo.heic",
    "minHeightRatio": 0.0,
    "minHeightPx": 0
  }
}
```

```json
{
  "input": {
    "imageBase64": "data:image/jpeg;base64,/9j/..."
  }
}
```

Successful jobs return `success: true` with `poseLandmarks`, `posePeople`, and
`posePersonCount`. Invalid inputs and decoding or inference failures return
`success: false` with an error code and explanatory message.

Both the RunPod worker and FastAPI `/pose` select at most four people after
applying the minimum-height filters. People are ordered by descending detector
bounding-box area as an approximation of proximity to the camera (not measured
depth). With zero, one, two, or three eligible people, all of them are returned;
with four or more, only the four largest are processed. `poseLandmarks` contains
the first selected person's landmarks, and `posePeople` follows the same order.
