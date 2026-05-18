"""
app/detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detection engine: YOLO + CLAHE night mode + motion gate.
Runs in a background thread, yields annotated frames.

AUTO-RECONNECT: if the DVR restarts or the stream drops,
  the engine retries automatically every RECONNECT_DELAY
  seconds.  The MJPEG stream keeps a heartbeat frame so
  the browser never hangs.
"""

import cv2
import os
import time
import threading
import queue
import numpy as np
from datetime import datetime
from collections import defaultdict

from app.config import (
    VIDEO_SOURCE, DETECT_SCALE, YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES,
    YOLO_CONF, YOLO_NMS, SAVE_CLASSES, MIN_BOX_AREA, MIN_FACE_SIZE,
    AUTO_NIGHT, NIGHT_BRIGHTNESS, CLAHE_CLIP, CLAHE_GRID,
    MOTION_THRESHOLD, MOTION_FRAMES,
    CLASS_COOLDOWN_S, PERSON_COOLDOWN_S, SAVE_MODE,
    MIN_STABLE_FRAMES, MIN_OBJECT_FRAME_RATIO, MAX_SAVES_PER_CLASS,
)


# ── CLAHE ─────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

def enhance_night(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def is_night_frame(frame):
    """
    Detect night/IR mode reliably.
    IR frames: moderate brightness but near-zero colour saturation (greyscale).
    Day frames: higher saturation (real colours visible).
    Trigger night mode if EITHER brightness is low OR saturation is near-zero.
    """
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
            print("⚠  YOLO weights not found — HOG+Haar fallback")
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
                print("✅ YOLO on CUDA")
        except Exception:
            pass
        if not cuda_ok:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("✅ YOLO on CPU")
        layer_names     = net.getLayerNames()
        self.out_layers = [layer_names[i-1]
                           for i in net.getUnconnectedOutLayers().flatten()]
        self.net = net

    @property
    def ready(self):
        return self.net is not None

    def detect(self, frame):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
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
                cx,cy,w,h = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
                confs.append(conf)
                class_ids.append(cid)
        indices = cv2.dnn.NMSBoxes(boxes, confs, YOLO_CONF, YOLO_NMS)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x,y,w,h = boxes[i]
                name = self.classes[class_ids[i]] if self.classes else str(class_ids[i])
                results.append((x,y,w,h,name,confs[i]))
        return results

# ── HOG fallback ──────────────────────────────────────────────────────────────
_face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
_body_hog      = cv2.HOGDescriptor()
_body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def _hog_detect(frame):
    small     = cv2.resize(frame, (0,0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    bodies, _ = _body_hog.detectMultiScale(small, winStride=(4,4), padding=(8,8), scale=1.02)
    s = 1/DETECT_SCALE
    results = [(int(x*s),int(y*s),int(w*s),int(h*s),"person",0.75)
               for (x,y,w,h) in bodies] if len(bodies) > 0 else []
    if not results:
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        uppers = _upper_cascade.detectMultiScale(gray, 1.1, 3, minSize=(MIN_FACE_SIZE,MIN_FACE_SIZE))
        if len(uppers) > 0:
            results = [(x,y,w,h,"person",0.6) for (x,y,w,h) in uppers]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE,MIN_FACE_SIZE))
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            results.append((x,y,w,h,"face",0.7))
    return results

def _nms(detections, overlap=0.4):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[2]*d[3], reverse=True)
    kept = []
    for det in detections:
        x1,y1,w1,h1 = det[:4]
        if w1*h1 < MIN_BOX_AREA:
            continue
        dup = False
        for k in kept:
            kx,ky,kw,kh = k[:4]
            ix = max(0, min(x1+w1,kx+kw) - max(x1,kx))
            iy = max(0, min(y1+h1,ky+kh) - max(y1,ky))
            inter = ix*iy
            union = w1*h1 + kw*kh - inter
            if union > 0 and inter/union > overlap:
                dup = True; break
        if not dup:
            kept.append(det)
    return kept

# ── Motion gate ───────────────────────────────────────────────────────────────
class MotionGate:
    def __init__(self):
        self._prev  = None
        self._count = 0

    def has_motion(self, frame):
        gray = cv2.GaussianBlur(
            cv2.cvtColor(cv2.resize(frame,(0,0),fx=0.25,fy=0.25), cv2.COLOR_BGR2GRAY),
            (7,7), 0)
        if self._prev is None:
            self._prev = gray; return False
        diff  = cv2.absdiff(self._prev, gray)
        score = int(diff.sum())
        self._prev = gray
        self._count = self._count+1 if score > MOTION_THRESHOLD else 0
        return self._count >= MOTION_FRAMES

# ── Cooldown tracker ──────────────────────────────────────────────────────────
class CooldownTracker:
    def __init__(self):
        self._last_saved = defaultdict(float)
        self.person_count = 0

    def is_new(self, cls):
        now      = time.time()
        cooldown = PERSON_COOLDOWN_S if cls in ("person", "face") else CLASS_COOLDOWN_S
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
def is_sharp(img, thr=80.0):
    return cv2.Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() >= thr

def is_bright_enough(img, lo=30, hi=230):
    m = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).mean()
    return lo <= m <= hi

def img_to_bytes(img, quality=80):
    h,w = img.shape[:2]
    if w > 900:
        scale = 900/w
        img   = cv2.resize(img,(900,int(h*scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def datetime_filename(cls):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return f"{cls}_{ts}.jpg"

def build_crop(frame, x, y, w, h, pad=20):
    H,W = frame.shape[:2]
    return frame[max(0,y-pad):min(H,y+h+pad), max(0,x-pad):min(W,x+w+pad)]

# ── Main detector thread ──────────────────────────────────────────────────────
class DetectorEngine:
    _MAX_READ_FAILS  = 15      # consecutive failures before reconnect
    _RECONNECT_DELAY = 10      # seconds between reconnect attempts (DDNS needs patience)
    _HEARTBEAT_S     = 2.0     # push offline frame at least this often

    def __init__(self, drive_client, upload_queue):
        self.drive        = drive_client
        self.upload_q     = upload_queue
        self._frame_lock  = threading.Lock()
        self._latest      = None
        self._stats       = {
            "saved": 0, "skipped": 0,
            "fps": 0.0, "night": False,
            "detections": 0, "persons": 0,
            "queue": 0, "mode": "—",
        }
        self._running     = False
        self._thread      = None
        self.registry     = None

    def start(self, known_encodings=None, person_count=0):
        self.registry   = CooldownTracker()
        self._running   = True
        self._frame_idx = 0
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._frame_lock:
            return self._latest

    def get_stats(self):
        s = dict(self._stats)
        s["queue"]   = self.upload_q.qsize()
        s["persons"] = self.registry.person_count if self.registry else 0
        return s

    def reset(self):
        if self.registry:
            self.registry.reset()
        self._stats["saved"] = self._stats["skipped"] = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _push_offline_frame(self, msg="CAMERA OFFLINE"):
        """Push a dark placeholder frame so the MJPEG stream keeps a heartbeat."""
        canvas = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(canvas, msg,
                    (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        cv2.putText(canvas, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ok:
            with self._frame_lock:
                self._latest = buf.tobytes()

    def _open_cap(self):
        """
        Try to open VideoCapture with a hard 35-second timeout.

        Increased from 10s → 35s to handle slow DDNS resolution and DVR
        wake-up latency.

        FFmpeg options:
          - CAP_PROP_OPEN_TIMEOUT_MSEC = 30_000  → 30 s open timeout
          - CAP_PROP_READ_TIMEOUT_MSEC = 15_000  → 15 s per-read timeout
        The outer thread join of 35 s ensures we never block longer than that
        even if FFmpeg ignores its own timeout.
        """
        OPEN_TIMEOUT = 35   # must be > CAP_PROP_OPEN_TIMEOUT_MSEC / 1000

        result = [None]

        def _try_open():
            try:
                cap = cv2.VideoCapture(
                    VIDEO_SOURCE,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30_000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15_000,
                    ]
                )
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    result[0] = cap
                else:
                    cap.release()
            except Exception as e:
                print(f"  _open_cap exception: {e}")

        t = threading.Thread(target=_try_open, daemon=True)
        t.start()
        t.join(timeout=OPEN_TIMEOUT)

        if result[0] is None:
            print(f"  ⚠  VideoCapture open timed out after {OPEN_TIMEOUT}s")
            return None

        return result[0]

    # ── Detection loop ─────────────────────────────────────────────────────────

    def _run(self):
        yolo     = YOLODetector()
        use_yolo = yolo.ready
        self._stats["mode"] = "YOLO" if use_yolo else "HOG+Haar"

        fps_t           = time.time()
        saved           = skipped = 0
        last_detections = []
        stable_counts   = {}
        saves_per_class = {}
        read_fail_count = 0

        import re as _re
        _safe_url = _re.sub(r"(rtsp://[^:]+:)[^@]+(@)", r"\1***\2", VIDEO_SOURCE)

        # ── Initial connection — keep trying until stream opens ────────────────
        cap     = None
        attempt = 0
        while self._running and cap is None:
            attempt += 1
            print(f"  🔌 Connection attempt #{attempt}: {_safe_url}")
            self._push_offline_frame("CAMERA OFFLINE — connecting…")
            cap = self._open_cap()
            if cap is None:
                print(f"❌ Cannot open: {_safe_url} — retrying in {self._RECONNECT_DELAY}s")
                for _ in range(self._RECONNECT_DELAY):
                    if not self._running:
                        break
                    time.sleep(1)

        if cap is None:   # _running went False while waiting
            return

        motion = MotionGate()
        print(f"✅ Stream opened on attempt #{attempt} | YOLO={'yes' if use_yolo else 'no (HOG)'}")

        last_heartbeat = time.time()

        while self._running:
            # ── Read frame with hard deadline ─────────────────────────────────
            # FIX: single if/else — no duplicate branch that overwrites ret=False
            READ_TIMEOUT_S = 15
            _result  = [False, None]
            _cap_ref = cap

            def _do_read(c=_cap_ref, r=_result):
                r[0], r[1] = c.read()

            _rt = threading.Thread(target=_do_read, daemon=True)
            _rt.start()
            _rt.join(timeout=READ_TIMEOUT_S)

            if _rt.is_alive():
                # Hard timeout — socket is truly dead
                print("  ⚠  cap.read() timed out — forcing reconnect")
                ret, frame = False, None
            else:
                ret, frame = _result

            # ── Handle read failure → auto-reconnect ──────────────────────────
            if not ret:
                read_fail_count += 1

                if time.time() - last_heartbeat >= self._HEARTBEAT_S:
                    self._push_offline_frame("NO SIGNAL — waiting…")
                    last_heartbeat = time.time()

                if read_fail_count < self._MAX_READ_FAILS:
                    time.sleep(0.05)
                    continue

                print(f"⚠  {read_fail_count} consecutive read failures — reconnecting…")
                cap.release()
                cap             = None
                read_fail_count = 0
                last_detections = []
                stable_counts   = {}

                while self._running:
                    self._push_offline_frame("DVR RESTARTING — reconnecting…")
                    for _ in range(self._RECONNECT_DELAY):
                        if not self._running:
                            break
                        time.sleep(1)
                    cap = self._open_cap()
                    if cap:
                        motion = MotionGate()
                        print("✅ Reconnected successfully")
                        last_heartbeat = time.time()
                        break

                if cap is None:
                    break
                continue

            # ── Successful read ───────────────────────────────────────────────
            read_fail_count = 0
            last_heartbeat  = time.time()

            self._frame_idx += 1
            now = time.time()
            self._stats["fps"] = round(
                0.9 * self._stats["fps"] + 0.1 / (max(now - fps_t, 1e-6)), 1)
            fps_t = now

            # Always push raw frame immediately for smooth stream
            self._push(frame)

            # ── Motion check ──────────────────────────────────────────────────
            if not motion.has_motion(frame):
                annotated = frame.copy()
                cv2.putText(annotated, "NO MOTION", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
                self._push(annotated)
                continue

            # ── Run detection only every 3rd frame ────────────────────────────
            if self._frame_idx % 3 != 0:
                if last_detections:
                    annotated = frame.copy()
                    for (x, y, w, h, cls, conf) in last_detections:
                        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(annotated, f"{cls} {conf:.0%}", (x, y-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    self._hud(annotated, saved, last_detections)
                    self._push(annotated)
                continue

            night = AUTO_NIGHT and is_night_frame(frame)
            self._stats["night"] = night
            proc  = enhance_night(frame) if night else frame

            if use_yolo:
                det_frame = cv2.resize(proc, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
                raw       = yolo.detect(det_frame)
                s         = 1 / DETECT_SCALE
                raw       = [(int(x*s), int(y*s), int(w*s), int(h*s), cls, conf)
                             for (x, y, w, h, cls, conf) in raw]
            else:
                raw = _hog_detect(proc)

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

            for (x, y, w, h, cls, conf) in detections:
                color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated, f"{cls} {conf:.0%}", (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Gate 1: object must fill minimum % of frame
                obj_ratio = (w * h) / frame_area
                if obj_ratio < MIN_OBJECT_FRAME_RATIO:
                    skipped += 1
                    continue

                # Gate 2: object must be stable for MIN_STABLE_FRAMES
                stable_counts[cls] = stable_counts.get(cls, 0) + 1
                if stable_counts[cls] < MIN_STABLE_FRAMES:
                    continue

                # Gate 3: max saves per class cap
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
                    self.upload_q.put(("image", img_bytes, fname))
                    saved += 1
                    saves_per_class[cls] = saves_per_class.get(cls, 0) + 1
                    self._stats["saved"] = saved
                    print(f"  ✅ [{cls}] → {fname}")
                except Exception as e:
                    print(f"  ⚠  Encode error: {e}")

            self._hud(annotated, saved, detections)
            self._push(annotated)

        if cap:
            cap.release()

    def _hud(self, frame, saved, detections):
        night     = self._stats.get("night", False)
        night_tag = "NIGHT" if night else "DAY"
        cv2.putText(frame, f"{self._stats['mode']} | {night_tag} | FPS:{self._stats['fps']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Saved:{saved} Persons:{self.registry.person_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    def _push(self, frame):
        h, w = frame.shape[:2]
        if w > 854:
            frame = cv2.resize(frame, (854, int(h * 854 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ok:
            with self._frame_lock:
                self._latest = buf.tobytes()