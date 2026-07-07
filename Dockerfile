FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN set -eux; \
	mkdir -p ControlNet-v1-1-nightly/annotator/ckpts; \
	curl -L --fail -o ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true"; \
	curl -L --fail -o ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true"
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
