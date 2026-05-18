# Live CCTV Tracking System

A production-oriented live tracking system that captures RTSP camera video, performs real-time object detection, streams MJPEG to a browser, and backs up captured images to Google Drive.

---

## 🚀 Project Overview

This repository combines:

- `FastAPI` for the web server and streaming endpoints
- `OpenCV` + `YOLOv4-tiny` for object detection
- Google Drive API for remote image backup
- Background upload/retry logic for resilient uploads
- A local backup folder for failed upload recovery

---

## 📁 Project Structure

```
Live_Tracking_System/
├── app/
│   ├── config.py       ← application configuration
│   ├── detector.py     ← detection engine + motion / night mode logic
│   ├── drive.py        ← Google Drive upload worker
│   └── main.py         ← FastAPI server + MJPEG stream
├── templates/
│   └── index.html      ← browser live view UI
├── Dockerfile
├── requirements.txt
├── README.md
├── coco.names
├── yolov4-tiny.cfg
├── yolov4-tiny.weights
├── credentials.json
├── token.json
└── .env
```

> Required runtime files like `coco.names`, `yolov4-tiny.cfg`, `yolov4-tiny.weights`, `credentials.json`, and `token.json` are not included in version control.

---

## ✅ Prerequisites

Make sure the following files are available before starting:

- `token.json` — Google OAuth token
- `credentials.json` — Google Cloud credentials
- `yolov4-tiny.weights` — YOLO weights
- `yolov4-tiny.cfg` — YOLO config
- `coco.names` — COCO class labels
- `.env` — camera and Drive settings

---

## 🛠️ Local Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

Visit:

```bash
http://localhost:5000
```

---

## 🐳 Docker Build

Build the Docker image:

```bash
docker build -t kamiorakzai/cctvtracker:latest .
```

---

## ▶️ Docker Run

Run the container with port mapping:

```bash
docker run -d --name cctvtracker -p 5000:5000 kamiorakzai/cctvtracker:latest
```

If you prefer to mount environment variables at runtime instead of baking them into the image:

```bash
docker run -d --name cctvtracker -p 5000:5000 --env-file .env kamiorakzai/cctvtracker:latest
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Live browser view |
| `/video_feed` | GET | MJPEG stream |
| `/stats` | GET | Detection statistics |
| `/reset` | POST | Reset saved person state |
| `/health` | GET | Health check |

---

## ⚙️ Configuration

The app uses `.env` to configure camera and Drive settings. Supported variables:

- `CAM_USER` — camera username
- `CAM_PASS` — camera password
- `CAM_HOST` — camera host or IP
- `CAM_PORT` — RTSP port
- `DRIVE_FOLDER_ID` — Google Drive folder ID
- `DRIVE_SUBFOLDER` — Drive subfolder name
- `TOKEN_FILE` — OAuth token file path (default `token.json`)
- `YOLO_CFG` — YOLO config path
- `YOLO_WEIGHTS` — YOLO weights path
- `YOLO_NAMES` — class names path

---

## 📈 Roadmap

### Short-term enhancements

- ✅ Stable Docker deployment
- ✅ Local MJPEG live view
- ✅ Google Drive upload retry logic
- ✅ Camera reconnect and stream heartbeat support

### Next development goals

- Add web UI controls for detection filters
- Add authentication for browser access
- Add alert notifications for detected events
- Add runtime configuration UI for camera and Drive settings

### Future improvements

- Support multiple camera sources
- Add dashboard and detection history
- Add video recording and playback
- Add GPU-accelerated inference support

---

## 💡 Notes

- The app listens on port `5000` inside the container.
- `saved_persons/` stores local backup images before upload.
- `app.main` may mount `static/` if present, but `static/` is not required for normal operation.

---

## 🔧 Common Commands

```bash
# Build image
docker build -t aiconsultix/cctvtracker:latest .

# Run container
docker run -d --name aiconsultix -p 5000:5000 aiconsultix/cctvtracker:latest

# View container logs
docker logs -f cctvtracker

# Stop container
docker stop cctvtracker

docker rm cctvtracker
```
