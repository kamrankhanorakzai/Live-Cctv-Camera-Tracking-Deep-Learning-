"""
app/main.py — FastAPI server
  GET  /            → live view HTML page
  GET  /video_feed  → MJPEG stream
  GET  /stats       → JSON stats
  POST /reset       → reset encodings
  GET  /health      → health check
"""
import queue
import threading
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import LOCAL_BACKUP_DIR
from app.drive import DriveClient, upload_worker
from app.detector import DetectorEngine

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Live CCTV Tracker")

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

# ── Globals ───────────────────────────────────────────────────────────────────
upload_q = queue.Queue()
drive    = DriveClient()
engine   = DetectorEngine(drive, upload_q)

@app.on_event("startup")
async def startup():
    threading.Thread(
        target=upload_worker, args=(drive, upload_q), daemon=True
    ).start()
    engine.start()
    print("🚀 Detection engine started")

@app.on_event("shutdown")
async def shutdown():
    engine.stop()
    upload_q.put(None)
    upload_q.join()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    s = engine.get_stats()
    return JSONResponse({
        "fps":        round(float(s.get("fps", 0)), 1),
        "mode":       str(s.get("mode", "—")),
        "night":      1 if s.get("night") else 0,
        "detections": int(s.get("detections", 0)),
        "persons":    int(s.get("persons", 0)),
        "saved":      int(s.get("saved", 0)),
        "skipped":    int(s.get("skipped", 0)),
        "queue":      int(s.get("queue", 0)),
    })

@app.post("/reset")
def reset():
    engine.reset()
    return {"status": "reset done"}

def _mjpeg_generator():
    """
    Yield MJPEG frames.

    - If the engine has a new frame, send it.
    - If no new frame arrives within HEARTBEAT_S seconds, re-send the last
      known frame (or an offline placeholder).  This keeps the browser
      connection alive even when the camera is down or the thread is slow.
    """
    boundary       = b"--frame\r\n"
    header         = b"Content-Type: image/jpeg\r\n\r\n"
    POLL_SLEEP     = 0.01   # seconds between frame checks
    HEARTBEAT_S    = 2.0    # re-send frame at least every N seconds
    last_sent      = None
    last_sent_time = time.time()

    while True:
        frame = engine.get_frame()
        now   = time.time()

        if frame is not None and frame is not last_sent:
            # New frame available — send it
            last_sent      = frame
            last_sent_time = now
            yield boundary + header + frame + b"\r\n"
        elif now - last_sent_time >= HEARTBEAT_S:
            # Nothing new for HEARTBEAT_S seconds — resend whatever we have
            # to keep the multipart stream alive (prevents browser timeout)
            payload        = last_sent if last_sent is not None else _offline_jpeg()
            last_sent_time = now
            yield boundary + header + payload + b"\r\n"
        else:
            time.sleep(POLL_SLEEP)


def _offline_jpeg():
    """Return a static JPEG placeholder used when no frames have ever arrived."""
    import cv2, numpy as np
    canvas = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(canvas, "WAITING FOR STREAM…",
                (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return buf.tobytes()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()