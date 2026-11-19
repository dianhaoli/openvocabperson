# syntax=docker/dockerfile:1
# One image serves the API and the React UI.
#   docker build -t openvocab-vision .                                     # CPU
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#                --build-arg REQUIREMENTS=requirements-local.txt -t openvocab-vision:gpu .

# ── stage 1: React UI ─────────────────────────────────────────────────────────
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── stage 2: Python backend ───────────────────────────────────────────────────
FROM python:3.12-slim

# torch wheel index: CPU by default (small); use a CUDA index (e.g. .../whl/cu124) for NVIDIA GPUs
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# requirements.txt (core) or requirements-local.txt (adds local Qwen-VL / HF detectors / sentence-transformers)
ARG REQUIREMENTS=requirements.txt

# git: the ultralytics CLIP package (YOLOE / YOLO-World text prompts) is installed from GitHub
# libgl1, libglib2.0-0: OpenCV (pulled in by ultralytics)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    OVP_STORAGE_DIR=/data/storage \
    OVP_WEIGHTS_DIR=/data/weights \
    OVP_FRONTEND_DIST=/app/frontend/dist \
    XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    YOLO_CONFIG_DIR=/data/cache/ultralytics

WORKDIR /app
COPY requirements*.txt ./
RUN pip install torch torchvision --index-url "${TORCH_INDEX_URL}" \
    && pip install -r "${REQUIREMENTS}"

COPY app.py ./
COPY ovp ./ovp
COPY --from=ui /ui/dist ./frontend/dist

# Ultralytics downloads weights (yoloe-*.pt, sam2.1_*.pt, mobileclip_blt.ts ...) into the working
# directory, so run from the weights volume: downloads persist across container restarts.
RUN mkdir -p /data/storage /data/weights /data/cache
WORKDIR /data/weights
VOLUME ["/data/storage", "/data/weights", "/data/cache"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1
CMD ["python", "-m", "ovp", "serve", "--host", "0.0.0.0", "--port", "8000"]
