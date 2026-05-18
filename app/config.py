"""
app/config.py — all settings in one place.
Sensitive values (credentials, IDs) live in .env — never hardcoded here.
"""
import os
import urllib.parse

# Load .env automatically if python-dotenv is available.
# Install once:  pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # .env already exported by the shell / Docker — no problem

# ── Camera ────────────────────────────────────────────────────────────────────
# Use os.getenv with defaults so missing .env never raises KeyError
USERNAME = os.getenv("CAM_USER", "kami")
PASSWORD = os.getenv("CAM_PASS", "kami123")
HOST     = os.getenv("CAM_HOST", "192.168.1.10")
PORT     = int(os.getenv("CAM_PORT", "554"))

_u = urllib.parse.quote(USERNAME, safe="")
_p = urllib.parse.quote(PASSWORD, safe="")

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
YOLO_CONF    = float(os.getenv("YOLO_CONF", "0.30"))   # was 0.45 — detect lower-confidence objects
YOLO_NMS     = float(os.getenv("YOLO_NMS",  "0.40"))

# ── Detection ─────────────────────────────────────────────────────────────────
DETECT_SCALE  = 0.5
SAVE_CLASSES  = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
MIN_BOX_AREA  = 2000    # was 4000 — allow smaller bounding boxes
MIN_FACE_SIZE = 50

# ── Cooldowns ─────────────────────────────────────────────────────────────────
PERSON_COOLDOWN_S = 30   # was 90 — re-save same person every 30s
CLASS_COOLDOWN_S  = 15   # was 45 — re-save cars/bikes every 15s

# ── Motion gate ───────────────────────────────────────────────────────────────
MOTION_THRESHOLD = 800   # was 1200 — trigger on subtler motion
MOTION_FRAMES    = 2

# ── Smart save gate ───────────────────────────────────────────────────────────
MIN_OBJECT_FRAME_RATIO = float(os.getenv("MIN_OBJECT_FRAME_RATIO", "0.002"))   # was 0.007
MIN_STABLE_FRAMES      = int(os.getenv("MIN_STABLE_FRAMES", "1"))
MAX_SAVES_PER_CLASS    = int(os.getenv("MAX_SAVES_PER_CLASS", "0"))

# ── Night mode ────────────────────────────────────────────────────────────────
AUTO_NIGHT       = True
NIGHT_BRIGHTNESS = 100
CLAHE_CLIP       = 3.0
CLAHE_GRID       = (8, 8)

# ── Save ──────────────────────────────────────────────────────────────────────
SAVE_MODE        = "full"
LOCAL_BACKUP_DIR = "saved_persons"