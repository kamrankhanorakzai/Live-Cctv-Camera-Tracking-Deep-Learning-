"""
app/config.py — all settings in one place.
Sensitive values (credentials, IDs) live in .env — never hardcoded here.
"""
import os
import urllib.parse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Camera ────────────────────────────────────────────────────────────────────
USERNAME = os.getenv("CAM_USER")
PASSWORD = os.getenv("CAM_PASS")
HOST     = os.getenv("CAM_HOST", "192.168.1.10")
PORT     = int(os.getenv("CAM_PORT"))

_u = urllib.parse.quote(USERNAME, safe="")
_p = urllib.parse.quote(PASSWORD, safe="")

# Force TCP transport via ffmpeg options — avoids UDP packet loss + HEVC POC errors
VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    f"rtsp://{_u}:{_p}@{HOST}:{PORT}/Streaming/Channels/101"
)

# ── Google Drive ──────────────────────────────────────────────────────────────
DRIVE_FOLDER_ID      = os.getenv("DRIVE_FOLDER_ID", "")
DRIVE_SUBFOLDER_NAME = os.getenv("DRIVE_SUBFOLDER", "DetectedObjects")
ENCODINGS_FILENAME   = "encodings.pkl"
TOKEN_FILE           = os.getenv("TOKEN_FILE", "token.json")

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_CFG     = os.getenv("YOLO_CFG",     "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov4-tiny.weights")
YOLO_NAMES   = os.getenv("YOLO_NAMES",   "coco.names")
YOLO_CONF    = float(os.getenv("YOLO_CONF", "0.30"))
YOLO_NMS     = float(os.getenv("YOLO_NMS",  "0.40"))

# ── Detection ─────────────────────────────────────────────────────────────────
DETECT_SCALE  = float(os.getenv("DETECT_SCALE", "0.5"))
# Comma-separated list of COCO class names to save; empty = save all
_save_cls_raw = os.getenv("SAVE_CLASSES", "person,car,truck,bus,motorcycle,bicycle")
SAVE_CLASSES  = [c.strip() for c in _save_cls_raw.split(",") if c.strip()]
MIN_BOX_AREA  = int(os.getenv("MIN_BOX_AREA",  "2000"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE",  "50"))

# ── Cooldowns ─────────────────────────────────────────────────────────────────
# How long (seconds) to wait before saving another image of the same person/class.
# Lower = more images captured per encounter; raise to prevent burst storage.
# Recommended range — person: 8–20 s, other classes: 5–12 s
PERSON_COOLDOWN_S = float(os.getenv("PERSON_COOLDOWN_S", "3"))
CLASS_COOLDOWN_S  = float(os.getenv("CLASS_COOLDOWN_S",   "3"))

# ── Motion gate ───────────────────────────────────────────────────────────────
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "800"))
MOTION_FRAMES    = int(os.getenv("MOTION_FRAMES",    "2"))

# ── Smart save gate ───────────────────────────────────────────────────────────
MIN_OBJECT_FRAME_RATIO = float(os.getenv("MIN_OBJECT_FRAME_RATIO", "0.004"))
MIN_STABLE_FRAMES      = int(os.getenv("MIN_STABLE_FRAMES",        "1"))
# 0 = unlimited saves per class per session
MAX_SAVES_PER_CLASS    = int(os.getenv("MAX_SAVES_PER_CLASS",      "0"))

# ── Night mode ────────────────────────────────────────────────────────────────
AUTO_NIGHT       = os.getenv("AUTO_NIGHT", "true").lower() == "true"
NIGHT_BRIGHTNESS = int(os.getenv("NIGHT_BRIGHTNESS", "100"))
CLAHE_CLIP       = float(os.getenv("CLAHE_CLIP", "3.0"))
_cg              = os.getenv("CLAHE_GRID", "8,8").split(",")
CLAHE_GRID       = (int(_cg[0]), int(_cg[1]))

# ── Save ──────────────────────────────────────────────────────────────────────
# "full" saves the entire frame; "crop" saves only the detected bounding box
SAVE_MODE        = os.getenv("SAVE_MODE", "full")
LOCAL_BACKUP_DIR = os.getenv("LOCAL_BACKUP_DIR", "saved_persons")

# ── FPS throttle ──────────────────────────────────────────────────────────────
# Max frames per second to process through YOLO (caps CPU on t3.small)
TARGET_PROC_FPS = float(os.getenv("TARGET_PROC_FPS", "5.0"))

# ── Upload retry ──────────────────────────────────────────────────────────────
UPLOAD_MAX_RETRIES     = int(os.getenv("UPLOAD_MAX_RETRIES",    "6"))
UPLOAD_BASE_BACKOFF_S  = float(os.getenv("UPLOAD_BASE_BACKOFF_S", "1.0"))  # doubles each attempt
UPLOAD_REQUEST_TIMEOUT = int(os.getenv("UPLOAD_REQUEST_TIMEOUT", "30"))    # seconds
RETRY_SCAN_INTERVAL_S  = int(os.getenv("RETRY_SCAN_INTERVAL_S",  "60"))

# ── RTSP ─────────────────────────────────────────────────────────────────────
RTSP_OPEN_TIMEOUT_MS  = int(os.getenv("RTSP_OPEN_TIMEOUT_MS",  "30000"))
RTSP_READ_TIMEOUT_MS  = int(os.getenv("RTSP_READ_TIMEOUT_MS",  "15000"))
RTSP_RECONNECT_DELAY  = int(os.getenv("RTSP_RECONNECT_DELAY",  "10"))
RTSP_MAX_READ_FAILS   = int(os.getenv("RTSP_MAX_READ_FAILS",   "8"))
# Treat cached frames older than this as dead signal (DVR restart / frozen stream)
RTSP_STALE_FRAME_S    = float(os.getenv("RTSP_STALE_FRAME_S",  "5.0"))

# ── Browser MJPEG preview (does not affect detection / uploads) ─────────────
STREAM_MAX_WIDTH     = int(os.getenv("STREAM_MAX_WIDTH",     "960"))
STREAM_JPEG_QUALITY  = int(os.getenv("STREAM_JPEG_QUALITY",  "78"))
STREAM_TARGET_FPS    = float(os.getenv("STREAM_TARGET_FPS",  "20.0"))