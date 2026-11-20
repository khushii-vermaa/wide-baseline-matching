FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only PyTorch keeps the image small; swap the index URL for a CUDA build if you deploy on a GPU host
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e . --no-deps
# Pre-download model weights so the first request is fast (non-fatal if the build has no internet)
RUN python -c "import kornia.feature as KF; KF.LoFTR(pretrained='outdoor')" || true
RUN python -c "from lightglue import LightGlue, SuperPoint; SuperPoint(); LightGlue(features='superpoint')" || true
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
