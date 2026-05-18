FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create optional static asset folder and copy project files
RUN mkdir -p /app/static
COPY app/ ./app/
COPY templates/ ./templates/

COPY yolov4-tiny.cfg     ./
COPY yolov4-tiny.weights ./
COPY coco.names          ./

COPY .env             ./.env
COPY token.json       ./token.json
COPY credentials.json ./credentials.json

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
