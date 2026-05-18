# ── Production Dockerfile ─────────────────────────────────────────────────────
# Python 3.11-slim: smaller image, faster startup, better memory management
# than 3.10. Fully compatible with all project dependencies.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    cmake \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Force FFmpeg RTSP over TCP + discard corrupt HEVC frames ─────────────────
# This env var is read by OpenCV before VideoCapture is opened.
# rtsp_transport=tcp  → eliminates UDP packet loss + POC/duplicate errors
# fflags=discardcorrupt → drops corrupt packets instead of crashing
# stimeout=5000000   → 5-second FFmpeg-level socket timeout (µs)
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;discardcorrupt|stimeout;5000000"

# ── Python dependencies (cache layer) ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Static assets + application code ─────────────────────────────────────────
RUN mkdir -p /app/static /app/logs /app/saved_persons
COPY app/       ./app/
COPY templates/ ./templates/

# ── YOLO model files ──────────────────────────────────────────────────────────
COPY yolov4-tiny.cfg     ./
COPY yolov4-tiny.weights ./
COPY coco.names          ./

# ── Credentials (injected at deploy time) ─────────────────────────────────────
# In production prefer Docker secrets or environment variables;
# these COPY lines are kept for development convenience.
COPY .env             ./.env
COPY token.json       ./token.json
COPY credentials.json ./credentials.json

# ── Expose application port ───────────────────────────────────────────────────
EXPOSE 5000

# ── Docker HEALTHCHECK ────────────────────────────────────────────────────────
# Polls /health every 30s; marks container unhealthy after 3 consecutive fails.
# docker run --restart unless-stopped will restart an unhealthy container.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]