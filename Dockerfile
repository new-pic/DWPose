FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN set -eux; \
	mkdir -p ControlNet-v1-1-nightly/annotator/ckpts; \
	curl -L --fail -o ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true"; \
	curl -L --fail -o ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true"

# This Serverless worker only packages the DWPose runtime; RMBG assets and code
# remain outside the worker image.
COPY main.py handler.py ./
COPY ControlNet-v1-1-nightly/annotator/dwpose \
    ControlNet-v1-1-nightly/annotator/dwpose

CMD ["python", "-u", "handler.py"]
