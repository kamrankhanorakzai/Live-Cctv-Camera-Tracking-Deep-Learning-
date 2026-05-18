import queue
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import LOCAL_BACKUP_DIR
from app.drive import DriveClient, upload_worker
from app.detector import DetectorEngine
from app.logger import get_logger

log = get_logger("main")

_start_time = time.time()

# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    threading.Thread(
        target=upload_worker, args=(app.state.drive, app.state.upload_q),
        daemon=True, name="upload-worker"
    ).start()
    app.state.engine.start()
    log.info("🚀 CCTV system started")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Shutting down…")
    app.state.engine.stop()
    app.state.upload_q.put(None)     # poison pill for upload_worker
    try:
        app.state.upload_q.join()    # wait for in-flight uploads (max 30s)
    except Exception:
        pass
    log.info("Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Live CCTV Tracker", lifespan=lifespan)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

# ── Module-level singletons (shared via app.state) ───────────────────────────
_upload_q = queue.Queue()
_drive    = DriveClient()
_engine   = DetectorEngine(_drive, _upload_q)

app.state.upload_q = _upload_q
app.state.drive    = _drive
app.state.engine   = _engine


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Used by Docker HEALTHCHECK and external monitors.
    Returns HTTP 200 while the engine thread is alive.
    """
    engine    = app.state.engine
    alive     = engine._thread is not None and engine._thread.is_alive()
    uptime_s  = int(time.time() - _start_time)
    status    = "ok" if alive else "degraded"
    return JSONResponse(
        content={"status": status, "uptime_s": uptime_s,
                 "queue": app.state.upload_q.qsize()},
        status_code=200 if alive else 503,
    )


@app.get("/stats")
def stats():
    s = app.state.engine.get_stats()
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
    app.state.engine.reset()
    return {"status": "reset done"}


# ── MJPEG stream ──────────────────────────────────────────────────────────────

def _offline_jpeg() -> bytes:
    canvas = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(canvas, "WAITING FOR STREAM…",
                (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return buf.tobytes()


def _mjpeg_generator():
    """
    Yield MJPEG frames.
    Re-sends last known frame (or offline placeholder) at least every
    HEARTBEAT_S seconds to prevent browser connection timeout.
    """
    boundary       = b"--frame\r\n"
    header         = b"Content-Type: image/jpeg\r\n\r\n"
    POLL_SLEEP     = 0.01
    HEARTBEAT_S    = 2.0
    last_sent      = None
    last_sent_time = time.time()
    engine         = app.state.engine

    while True:
        frame = engine.get_frame()
        now   = time.time()

        if frame is not None and frame is not last_sent:
            last_sent      = frame
            last_sent_time = now
            yield boundary + header + frame + b"\r\n"
        elif now - last_sent_time >= HEARTBEAT_S:
            payload        = last_sent if last_sent is not None else _offline_jpeg()
            last_sent_time = now
            yield boundary + header + payload + b"\r\n"
        else:
            time.sleep(POLL_SLEEP)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()