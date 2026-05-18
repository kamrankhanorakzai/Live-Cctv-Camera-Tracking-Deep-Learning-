"""
app/detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detection engine: YOLO + CLAHE night mode + motion gate.

Production improvements:
  • TCP-forced RTSP transport (no UDP packet loss / HEVC POC errors)
  • Hard read-deadline thread with automatic reconnect
  • FPS throttle — skips YOLO on overloaded frames (configurable TARGET_PROC_FPS)
  • Explicit frame + numpy array de-referencing to prevent memory leaks
  • gc.collect() every 500 frames to reclaim fragmented memory
  • Corrupted / empty frames caught and discarded safely
  • Heartbeat offline frame keeps MJPEG stream alive during outages
  • [NEW] Dedicated frame-reader thread drains camera buffer continuously
    so the detection loop never blocks on cap.read() — eliminates stream lag
  • [FIX] DVR restart freeze: FrameReader.stop() now clears stale frame cache
    and joins thread before cap.release() to eliminate race conditions
  • [FIX] self._latest cleared on reconnect so MJPEG endpoint never serves
    frozen frames during DVR downtime
  • [FIX] FrameReader joins in-flight cap.read() thread before cap.release()
  • [FIX] read_fail_seen synced with reader.fail_count to avoid false reconnects
  • [FIX] frame age gate — stale cached frames no longer masquerade as live signal
  • [FIX] single-threaded cap.read() — never release() while another thread reads
"""

import cv2
import gc
import os
import re
import time
import threading
import queue
import numpy as np
from datetime import datetime
from collections import defaultdict

from app.config import (
    VIDEO_SOURCE, DETECT_SCALE,
    YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES,
    YOLO_CONF, YOLO_NMS,
    SAVE_CLASSES, MIN_BOX_AREA, MIN_FACE_SIZE,
    AUTO_NIGHT, NIGHT_BRIGHTNESS, CLAHE_CLIP, CLAHE_GRID,
    MOTION_THRESHOLD, MOTION_FRAMES,
    CLASS_COOLDOWN_S, PERSON_COOLDOWN_S, SAVE_MODE,
    MIN_STABLE_FRAMES, MIN_OBJECT_FRAME_RATIO, MAX_SAVES_PER_CLASS,
    TARGET_PROC_FPS,
    RTSP_OPEN_TIMEOUT_MS, RTSP_READ_TIMEOUT_MS,
    RTSP_RECONNECT_DELAY, RTSP_MAX_READ_FAILS, RTSP_STALE_FRAME_S,
    STREAM_MAX_WIDTH, STREAM_JPEG_QUALITY,
)
from app.logger import get_logger

log = get_logger("detector")

# Mask real credentials in log output
_SAFE_URL = re.sub(r"(rtsp://[^:]+:)[^@]+(@)", r"\1***\2", VIDEO_SOURCE)

# ── CLAHE ─────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def enhance_night(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    del lab, l, a, b
    return result


def is_night_frame(frame: np.ndarray) -> bool:
    hsv             = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_brightness = float(hsv[:, :, 2].mean())
    mean_saturation = float(hsv[:, :, 1].mean())
    return mean_brightness < NIGHT_BRIGHTNESS or mean_saturation < 20


# ── Colors ────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "person":     (0, 255, 0),
    "face":       (0, 200, 0),
    "car":        (255, 180, 0),
    "truck":      (255, 120, 0),
    "bus":        (255, 80,  0),
    "motorcycle": (200, 255, 0),
    "bicycle":    (160, 255, 0),
}
DEFAULT_COLOR = (0, 200, 255)


# ── YOLO ──────────────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self):
        self.net        = None
        self.classes    = []
        self.out_layers = []
        self._load()

    def _load(self):
        if not (os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS)):
            log.warning("YOLO weights not found — HOG+Haar fallback active")
            return
        if os.path.exists(YOLO_NAMES):
            with open(YOLO_NAMES) as f:
                self.classes = [l.strip() for l in f]
        net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
        cuda_ok = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                cuda_ok = True
                log.info("YOLO running on CUDA")
        except Exception:
            pass
        if not cuda_ok:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            log.info("YOLO running on CPU")
        layer_names      = net.getLayerNames()
        self.out_layers  = [layer_names[i - 1]
                            for i in net.getUnconnectedOutLayers().flatten()]
        self.net = net

    @property
    def ready(self) -> bool:
        return self.net is not None

    def detect(self, frame: np.ndarray) -> list:
        """Run YOLO inference. Returns list of (x,y,w,h,name,conf)."""
        try:
            H, W = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.out_layers)
            boxes, confs, class_ids = [], [], []
            for out in outs:
                for det in out:
                    scores = det[5:]
                    cid    = int(np.argmax(scores))
                    conf   = float(scores[cid])
                    if conf < YOLO_CONF:
                        continue
                    cx, cy, w, h = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                    boxes.append([int(cx - w/2), int(cy - h/2), int(w), int(h)])
                    confs.append(conf)
                    class_ids.append(cid)
            indices = cv2.dnn.NMSBoxes(boxes, confs, YOLO_CONF, YOLO_NMS)
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    name = (self.classes[class_ids[i]]
                            if self.classes else str(class_ids[i]))
                    results.append((x, y, w, h, name, confs[i]))
            # Explicitly release blob memory
            del blob, outs
            return results
        except Exception as exc:
            log.error(f"YOLO inference error: {exc}")
            return []


# ── HOG fallback ──────────────────────────────────────────────────────────────
_face_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_upper_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_upperbody.xml")
_body_hog = cv2.HOGDescriptor()
_body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def _hog_detect(frame: np.ndarray) -> list:
    small     = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    bodies, _ = _body_hog.detectMultiScale(small, winStride=(4, 4),
                                           padding=(8, 8), scale=1.02)
    s       = 1 / DETECT_SCALE
    results = [(int(x*s), int(y*s), int(w*s), int(h*s), "person", 0.75)
               for (x, y, w, h) in bodies] if len(bodies) > 0 else []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not results:
        uppers = _upper_cascade.detectMultiScale(
            gray, 1.1, 3, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        if len(uppers) > 0:
            results = [(x, y, w, h, "person", 0.6) for (x, y, w, h) in uppers]
    faces = _face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            results.append((x, y, w, h, "face", 0.7))
    return results


def _nms(detections: list, overlap: float = 0.4) -> list:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[2] * d[3], reverse=True)
    kept = []
    for det in detections:
        x1, y1, w1, h1 = det[:4]
        if w1 * h1 < MIN_BOX_AREA:
            continue
        dup = False
        for k in kept:
            kx, ky, kw, kh = k[:4]
            ix    = max(0, min(x1+w1, kx+kw) - max(x1, kx))
            iy    = max(0, min(y1+h1, ky+kh) - max(y1, ky))
            inter = ix * iy
            union = w1*h1 + kw*kh - inter
            if union > 0 and inter / union > overlap:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


# ── Motion gate ───────────────────────────────────────────────────────────────
class MotionGate:
    def __init__(self):
        self._prev  = None
        self._count = 0

    def has_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.GaussianBlur(
            cv2.cvtColor(
                cv2.resize(frame, (0, 0), fx=0.25, fy=0.25),
                cv2.COLOR_BGR2GRAY),
            (7, 7), 0)
        if self._prev is None:
            self._prev = gray
            return False
        diff  = cv2.absdiff(self._prev, gray)
        score = int(diff.sum())
        del diff
        self._prev = gray
        self._count = self._count + 1 if score > MOTION_THRESHOLD else 0
        return self._count >= MOTION_FRAMES

    def reset(self):
        self._prev  = None
        self._count = 0


# ── Cooldown tracker ──────────────────────────────────────────────────────────
class CooldownTracker:
    def __init__(self):
        self._last_saved  = defaultdict(float)
        self.person_count = 0

    def is_new(self, cls: str) -> bool:
        now      = time.time()
        cooldown = (PERSON_COOLDOWN_S if cls in ("person", "face")
                    else CLASS_COOLDOWN_S)
        if now - self._last_saved[cls] < cooldown:
            return False
        self._last_saved[cls] = now
        if cls in ("person", "face"):
            self.person_count += 1
        return True

    def reset(self):
        self._last_saved.clear()
        self.person_count = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_sharp(img: np.ndarray, thr: float = 80.0) -> bool:
    return (cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var() >= thr)


def is_bright_enough(img: np.ndarray, lo: int = 30, hi: int = 230) -> bool:
    m = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    return lo <= m <= hi


def img_to_bytes(img: np.ndarray, quality: int = 80) -> bytes:
    h, w = img.shape[:2]
    if w > 900:
        scale = 900 / w
        img   = cv2.resize(img, (900, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def datetime_filename(cls: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return f"{cls}_{ts}.jpg"


def build_crop(frame: np.ndarray, x: int, y: int,
               w: int, h: int, pad: int = 20) -> np.ndarray:
    H, W = frame.shape[:2]
    return frame[max(0, y-pad):min(H, y+h+pad),
                 max(0, x-pad):min(W, x+w+pad)]


# ── RTSP capture helper ───────────────────────────────────────────────────────
def _open_cap() -> "cv2.VideoCapture | None":
    """
    Open VideoCapture with:
      • rtsp_transport=tcp  — eliminates UDP packet loss + HEVC POC/duplicate errors
      • fflags=discardcorrupt — FFmpeg discards corrupt packets instead of crashing
      • stimeout             — FFmpeg-level socket timeout
      • hard outer timeout of 35s via join()
    """
    OPEN_TIMEOUT_S = 35
    result = [None]

    def _try():
        try:
            cap = cv2.VideoCapture(
                VIDEO_SOURCE,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS,
                ],
            )
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                result[0] = cap
            else:
                cap.release()
        except Exception as exc:
            log.error(f"VideoCapture open exception: {exc}")

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=OPEN_TIMEOUT_S)
    if t.is_alive():
        log.warning(
            f"VideoCapture open timed out after {OPEN_TIMEOUT_S}s — "
            "open thread still running (daemon, reaped on exit)")
        result[0] = None
    elif result[0] is None:
        log.warning(f"VideoCapture open failed after {OPEN_TIMEOUT_S}s")
    return result[0]


# ── Frame reader thread ───────────────────────────────────────────────────────
class FrameReader:
    """
    Single background thread owns cap.read() — no nested read threads.
    FFmpeg/VideoCapture is not thread-safe; only this thread touches self._cap.
    """

    def __init__(self, cap: "cv2.VideoCapture"):
        self._cap        = cap
        self._lock       = threading.Lock()
        self._frame      = None
        self._frame_ts   = 0.0
        self._fail_count = 0
        self._running    = True
        self._thread     = threading.Thread(
            target=self._loop, daemon=True, name="frame-reader")
        self._thread.start()

    def _loop(self):
        try:
            while self._running:
                try:
                    ret, frame = self._cap.read()
                    with self._lock:
                        if ret and frame is not None and frame.size > 0:
                            self._frame     = frame
                            self._frame_ts   = time.monotonic()
                            self._fail_count = 0
                        else:
                            self._fail_count += 1
                    if not ret:
                        time.sleep(0.05)
                except Exception as exc:
                    log.error(f"FrameReader error: {exc}")
                    with self._lock:
                        self._fail_count += 1
                    time.sleep(0.2)
        finally:
            try:
                self._cap.release()
            except Exception:
                pass
            log.debug("FrameReader released VideoCapture")

    def read(self) -> "tuple[bool, np.ndarray | None]":
        """Non-blocking: return (True, frame) or (False, None)."""
        with self._lock:
            if self._frame is None:
                return False, None
            age = time.monotonic() - self._frame_ts
            if age > RTSP_STALE_FRAME_S:
                return False, None
            return True, self._frame.copy()

    @property
    def fail_count(self) -> int:
        with self._lock:
            return self._fail_count

    @property
    def should_reconnect(self) -> bool:
        with self._lock:
            return self._fail_count >= RTSP_MAX_READ_FAILS

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> bool:
        """
        Stop the reader loop and wait for it to exit.
        Returns True if the thread exited (safe to call cap.release()).
        """
        join_s = (RTSP_READ_TIMEOUT_MS / 1000.0) + 5.0
        self._running = False
        with self._lock:
            self._frame = None
        self._thread.join(timeout=join_s)
        if self._thread.is_alive():
            log.warning(
                f"FrameReader still in cap.read() after {join_s:.0f}s — "
                "deferring cap.release()")
            return False
        return True


# ── Main detector engine ──────────────────────────────────────────────────────
class DetectorEngine:
    _HEARTBEAT_S     = 2.0
    _GC_EVERY_FRAMES = 500     # call gc.collect() to reclaim fragmented memory

    def __init__(self, drive_client, upload_queue: queue.Queue):
        self.drive       = drive_client
        self.upload_q    = upload_queue
        self._frame_lock = threading.Lock()
        self._latest     = None          # bytes (JPEG)
        self._stream_seq = 0
        self._stats      = {
            "saved": 0, "skipped": 0,
            "fps": 0.0, "night": False,
            "detections": 0, "persons": 0,
            "queue": 0, "mode": "—",
        }
        self._running  = False
        self._thread   = None
        self.registry  = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self.registry   = CooldownTracker()
        self._running   = True
        self._frame_idx = 0
        self._thread    = threading.Thread(
            target=self._run, daemon=True, name="detector")
        self._thread.start()
        log.info("Detection engine started")

    def stop(self):
        log.info("Stopping detection engine…")
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def get_frame(self) -> "bytes | None":
        """Return the latest JPEG bytes for the MJPEG endpoint."""
        with self._frame_lock:
            return self._latest

    @property
    def stream_seq(self) -> int:
        with self._frame_lock:
            return self._stream_seq

    def get_stats(self) -> dict:
        s = dict(self._stats)
        s["queue"]   = self.upload_q.qsize()
        s["persons"] = self.registry.person_count if self.registry else 0
        return s

    def reset(self):
        if self.registry:
            self.registry.reset()
        self._stats["saved"] = self._stats["skipped"] = 0
        log.info("Cooldowns and counters reset")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _push_offline_frame(self, msg: str = "CAMERA OFFLINE"):
        canvas = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(canvas, msg,
                    (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        cv2.putText(canvas, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        ok, buf = cv2.imencode(
            ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ok:
            with self._frame_lock:
                self._latest = buf.tobytes()
                self._stream_seq += 1
        del canvas

    def _push(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if w > STREAM_MAX_WIDTH:
            frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * STREAM_MAX_WIDTH / w)))
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
        if ok:
            with self._frame_lock:
                self._latest = buf.tobytes()
                self._stream_seq += 1

    def _draw_boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for (x, y, w, h, cls, conf) in detections:
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{cls} {conf:.0%}", (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def _push_stream_view(self, frame: np.ndarray, last_detections: list,
                          saved: int, banner: "str | None" = None):
        """Single preview encode per loop — overlays + HUD for the dashboard."""
        view = frame.copy()
        if last_detections:
            self._draw_boxes(view, last_detections)
        if banner:
            cv2.putText(view, banner, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
        self._hud(view, saved, last_detections)
        self._push(view)
        del view

    def _hud(self, frame: np.ndarray, saved: int, detections: list):
        night_tag = "NIGHT" if self._stats.get("night") else "DAY"
        cv2.putText(frame,
                    f"{self._stats['mode']} | {night_tag} | FPS:{self._stats['fps']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame,
                    f"Saved:{saved} Persons:{self.registry.person_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # ── Connect helper (used by both initial connect and reconnect) ────────────

    @staticmethod
    def _teardown_stream(cap, reader) -> None:
        """
        Stop the reader thread; it releases VideoCapture in its own finally block.
        Never call cap.release() from here — FFmpeg is not thread-safe.
        """
        if reader is not None:
            exited = reader.stop()
            if not exited:
                log.error(
                    "FrameReader stuck in cap.read() — old session orphaned; "
                    "opening a new connection anyway")
        time.sleep(0.5)

    def _connect(self) -> "tuple[cv2.VideoCapture, FrameReader] | tuple[None, None]":
        """
        Open the RTSP stream and return (cap, reader) or (None, None).

        FIX 4: clear self._latest at the start of every connect attempt so the
        MJPEG endpoint never serves a frozen frame between retries. The
        subsequent _push_offline_frame() immediately replaces None with a
        live status banner so get_frame() always has something to return.
        """
        attempt = 0
        while self._running:
            attempt += 1
            log.info(f"RTSP connect attempt #{attempt}: {_SAFE_URL}")

            # FIX 4: wipe stale MJPEG cache before pushing the offline banner.
            # Ensures get_frame() returns the *current* status, not a frozen
            # frame from a previous successful session.
            with self._frame_lock:
                self._latest = None

            self._push_offline_frame("CAMERA OFFLINE — connecting…")
            cap = _open_cap()
            if cap is not None:
                time.sleep(2)
                reader = FrameReader(cap)
                log.info(f"Stream opened on attempt #{attempt}")
                return cap, reader
            log.warning(
                f"Cannot open stream — retrying in {RTSP_RECONNECT_DELAY}s")
            for _ in range(RTSP_RECONNECT_DELAY):
                if not self._running:
                    return None, None
                time.sleep(1)
        return None, None

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run(self):
        # Set FFmpeg options for TCP transport + corrupt frame discard
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;discardcorrupt|stimeout;5000000"
        )

        yolo     = YOLODetector()
        use_yolo = yolo.ready
        self._stats["mode"] = "YOLO" if use_yolo else "HOG+Haar"
        log.info(f"Detector mode: {self._stats['mode']}")

        fps_t           = time.time()
        proc_fps_t      = time.time()
        saved           = skipped = 0
        last_detections = []
        stable_counts   = {}
        saves_per_class = {}

        # ── Initial connect ───────────────────────────────────────────────────
        cap, reader = self._connect()
        if cap is None:
            log.error("Detection engine stopped: stream never opened")
            return

        motion         = MotionGate()
        last_heartbeat = time.time()
        read_fail_seen = 0          # consecutive empty reads from FrameReader

        while self._running:
            frame     = None
            proc      = None
            annotated = None

            # ── FPS throttle ──────────────────────────────────────────────────
            now          = time.time()
            min_interval = 1.0 / TARGET_PROC_FPS
            elapsed      = now - proc_fps_t
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            # ── Grab latest frame from reader (non-blocking) ──────────────────
            ret, frame = reader.read()
            proc_fps_t  = time.time()

            stale_signal = not ret or frame is None
            need_reconnect = (
                not reader.is_alive()
                or reader.should_reconnect
                or stale_signal
            )

            # ── No signal / stale cache / read failures → reconnect ───────────
            if need_reconnect:
                if reader.should_reconnect or not reader.is_alive():
                    read_fail_seen = RTSP_MAX_READ_FAILS
                else:
                    read_fail_seen += 1

                if time.time() - last_heartbeat >= self._HEARTBEAT_S:
                    self._push_offline_frame("NO SIGNAL — waiting…")
                    last_heartbeat = time.time()

                if (
                    reader.is_alive()
                    and not reader.should_reconnect
                    and read_fail_seen < RTSP_MAX_READ_FAILS
                ):
                    time.sleep(0.05)
                    continue

                log.warning(
                    f"RTSP reconnect "
                    f"(fails={reader.fail_count}, seen={read_fail_seen}, "
                    f"alive={reader.is_alive()})")

                with self._frame_lock:
                    self._latest = None

                self._push_offline_frame("DVR RESTARTING — reconnecting…")
                self._teardown_stream(cap, reader)
                cap = reader = None
                read_fail_seen  = 0
                last_detections = []
                stable_counts   = {}

                time.sleep(1)
                cap, reader = self._connect()
                if cap is None:
                    break
                motion         = MotionGate()
                last_heartbeat = time.time()
                continue

            # ── Healthy frame ─────────────────────────────────────────────────
            read_fail_seen = 0
            last_heartbeat  = time.time()
            self._frame_idx += 1

            # Rolling FPS estimate
            now = time.time()
            self._stats["fps"] = round(
                0.9 * self._stats["fps"] + 0.1 / max(now - fps_t, 1e-6), 1)
            fps_t = now

            # ── Periodic GC ───────────────────────────────────────────────────
            if self._frame_idx % self._GC_EVERY_FRAMES == 0:
                gc.collect()
                log.info(
                    f"GC collect at frame #{self._frame_idx} "
                    f"| saved={saved} skipped={skipped}")

            # ── Motion gate ───────────────────────────────────────────────────
            if not motion.has_motion(frame):
                self._push_stream_view(frame, last_detections, saved, "NO MOTION")
                del frame
                frame = None
                continue

            # ── Run detection only every 3rd frame ────────────────────────────
            if self._frame_idx % 3 != 0:
                self._push_stream_view(frame, last_detections, saved)
                del frame
                frame = None
                continue

            # ── Night mode ────────────────────────────────────────────────────
            try:
                night = AUTO_NIGHT and is_night_frame(frame)
                self._stats["night"] = night
                proc  = enhance_night(frame) if night else frame
            except Exception as exc:
                log.error(f"Night mode error: {exc}")
                proc = frame

            # ── YOLO / HOG detection ──────────────────────────────────────────
            try:
                if use_yolo:
                    det_frame = cv2.resize(proc, (0, 0),
                                           fx=DETECT_SCALE, fy=DETECT_SCALE)
                    raw       = yolo.detect(det_frame)
                    s         = 1 / DETECT_SCALE
                    raw       = [(int(x*s), int(y*s), int(w*s), int(h*s), cls, conf)
                                 for (x, y, w, h, cls, conf) in raw]
                    del det_frame
                else:
                    raw = _hog_detect(proc)
            except Exception as exc:
                log.error(f"Detection inference error: {exc}")
                raw = []

            detections = _nms(raw)
            if SAVE_CLASSES:
                detections = [d for d in detections if d[4] in SAVE_CLASSES]

            last_detections = detections
            detected_cls    = {d[4] for d in detections}
            for cls in list(stable_counts.keys()):
                if cls not in detected_cls:
                    stable_counts[cls] = 0
            self._stats["detections"] = len(detections)

            annotated  = frame.copy()
            fH, fW     = frame.shape[:2]
            frame_area = fH * fW

            self._draw_boxes(annotated, detections)

            for (x, y, w, h, cls, conf) in detections:
                # Gate 1: object must fill minimum % of frame
                if (w * h) / frame_area < MIN_OBJECT_FRAME_RATIO:
                    skipped += 1
                    continue

                # Gate 2: stability across frames
                stable_counts[cls] = stable_counts.get(cls, 0) + 1
                if stable_counts[cls] < MIN_STABLE_FRAMES:
                    continue

                # Gate 3: per-class cap
                if MAX_SAVES_PER_CLASS > 0:
                    if saves_per_class.get(cls, 0) >= MAX_SAVES_PER_CLASS:
                        skipped += 1
                        continue

                crop = build_crop(frame, x, y, w, h)
                if crop.size == 0:
                    continue

                if not self.registry.is_new(cls):
                    skipped += 1
                    self._stats["skipped"] = skipped
                    continue

                save_img = frame if SAVE_MODE == "full" else crop
                if not is_sharp(save_img) or not is_bright_enough(save_img):
                    skipped += 1
                    continue

                try:
                    fname     = datetime_filename(cls)
                    img_bytes = img_to_bytes(save_img)
                    qdepth = self.upload_q.qsize()
                    if qdepth > 100:
                        log.warning(
                            f"Upload queue depth {qdepth} — Drive may be unreachable")
                    self.upload_q.put(("image", img_bytes, fname))
                    saved += 1
                    saves_per_class[cls] = saves_per_class.get(cls, 0) + 1
                    self._stats["saved"] = saved
                    log.info(f"[{cls}] queued for upload → {fname}")
                except Exception as exc:
                    log.error(f"Encode error for {cls}: {exc}")
                finally:
                    del crop

            self._hud(annotated, saved, detections)
            self._push(annotated)

            # ── Explicit cleanup ──────────────────────────────────────────────
            del annotated
            del proc
            del frame

        self._teardown_stream(cap, reader)
        log.info("Detection engine stopped cleanly")