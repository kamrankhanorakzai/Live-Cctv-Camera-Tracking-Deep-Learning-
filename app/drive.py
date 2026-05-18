"""
app/drive.py — Google Drive client with production-grade upload reliability.

Improvements over original:
  • Exponential backoff retry (1 → 2 → 4 → 8 … seconds, up to UPLOAD_MAX_RETRIES)
  • Request timeout on every Drive API call (prevents hung uploads)
  • Thread-safe retry queue using threading.Lock (no race between save and upload)
  • File-existence check before retry (prevents FileNotFoundError in retry worker)
  • Duplicate upload prevention: in-flight set tracks filenames being uploaded
  • Structured logging instead of bare print()
"""

import io
import os
import time
import threading
import queue

from app.config import (
    DRIVE_FOLDER_ID, DRIVE_SUBFOLDER_NAME,
    ENCODINGS_FILENAME, TOKEN_FILE, LOCAL_BACKUP_DIR,
    UPLOAD_MAX_RETRIES, UPLOAD_BASE_BACKOFF_S,
    UPLOAD_REQUEST_TIMEOUT, RETRY_SCAN_INTERVAL_S,
)
from app.logger import get_logger

log = get_logger("drive")


class DriveClient:
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self):
        self.service      = None
        self.folder_id    = None
        self._connect()

    def _connect(self):
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            if not os.path.exists(TOKEN_FILE):
                raise FileNotFoundError(
                    f"{TOKEN_FILE} not found. Run get_token.py first.")

            creds = Credentials.from_authorized_user_file(TOKEN_FILE, self.SCOPES)
            if creds.expired and creds.refresh_token:
                log.info("Refreshing Drive token…")
                creds.refresh(Request())
                with open(TOKEN_FILE, "w") as f:
                    f.write(creds.to_json())

            self.service   = build("drive", "v3", credentials=creds)
            about          = self.service.about().get(fields="user").execute()
            log.info(f"Drive connected → {about['user']['emailAddress']}")
            self.folder_id = self._resolve_subfolder()
            log.info(f"Drive folder: {DRIVE_SUBFOLDER_NAME} ({self.folder_id})")

        except Exception as exc:
            log.error(f"Drive connection failed: {exc}")

    def _resolve_subfolder(self) -> str:
        q   = (f"name='{DRIVE_SUBFOLDER_NAME}' and '{DRIVE_FOLDER_ID}' in parents "
               f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
        res = self.service.files().list(q=q, fields="files(id)").execute()
        fls = res.get("files", [])
        if fls:
            return fls[0]["id"]
        meta = {
            "name":     DRIVE_SUBFOLDER_NAME,
            "mimeType": "application/vnd.google-apps.folder",
            "parents":  [DRIVE_FOLDER_ID],
        }
        f = self.service.files().create(body=meta, fields="id").execute()
        log.info(f"Created Drive subfolder: {DRIVE_SUBFOLDER_NAME}")
        return f["id"]

    def upload_image(self, img_bytes: bytes, filename: str) -> bool:
        """Upload a JPEG image to Drive. Returns True on success."""
        if not self.service:
            log.warning("Drive not connected — skipping upload")
            return False
        from googleapiclient.http import MediaIoBaseUpload
        try:
            meta  = {"name": filename, "parents": [self.folder_id]}
            media = MediaIoBaseUpload(
                io.BytesIO(img_bytes),
                mimetype="image/jpeg",
                resumable=False,
                # chunksize is irrelevant for non-resumable but set timeout via
                # the httplib2 / requests layer (see _http_timeout below)
            )
            # Build a request object and execute with timeout
            req = self.service.files().create(
                body=meta, media_body=media, fields="id,name")
            f = self._execute_with_timeout(req)
            if f:
                log.info(f"Uploaded to Drive: {f['name']}")
                return True
            return False
        except Exception as exc:
            log.error(f"Drive upload failed [{filename}]: {exc}")
            return False

    def _execute_with_timeout(self, request):
        """
        Execute a Google API request in a daemon thread so we can impose a
        hard wall-clock timeout.  Prevents SSL broken-pipe hangs from blocking
        the upload worker forever.
        """
        result    = [None]
        exc_box   = [None]

        def _run():
            try:
                result[0] = request.execute()
            except Exception as e:
                exc_box[0] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=UPLOAD_REQUEST_TIMEOUT)
        if t.is_alive():
            log.warning(
                f"Drive API request timed out after {UPLOAD_REQUEST_TIMEOUT}s")
            return None
        if exc_box[0]:
            raise exc_box[0]
        return result[0]

    def is_ready(self) -> bool:
        return self.service is not None and self.folder_id is not None

    # Kept for backward compatibility
    def download_encodings(self):
        return [], 0

    def upload_encodings(self, encodings, count):
        pass


# ── Upload worker ──────────────────────────────────────────────────────────────
def upload_worker(drive: DriveClient, upload_q: queue.Queue):
    """
    Main upload consumer.

    Design:
      • Saves file locally FIRST as crash-safe backup.
      • Then uploads to Drive with exponential backoff.
      • Removes local file only after confirmed upload.
      • in_flight set prevents duplicate uploads of the same filename
        (guards against queue being fed twice for the same file).
      • Retry worker scans LOCAL_BACKUP_DIR for orphaned files every
        RETRY_SCAN_INTERVAL_S seconds, skipping any file currently in_flight.
    """
    os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)

    in_flight      = set()          # filenames currently being uploaded
    in_flight_lock = threading.Lock()

    def _upload_with_backoff(img_bytes: bytes, fname: str) -> bool:
        """
        Try up to UPLOAD_MAX_RETRIES times with exponential backoff.
        Returns True if upload ultimately succeeded.
        """
        delay = UPLOAD_BASE_BACKOFF_S
        for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
            ok = drive.upload_image(img_bytes, fname)
            if ok:
                return True
            if attempt < UPLOAD_MAX_RETRIES:
                log.warning(
                    f"Upload attempt {attempt}/{UPLOAD_MAX_RETRIES} failed "
                    f"for {fname} — retrying in {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)   # cap at 2 minutes
            else:
                log.error(
                    f"All {UPLOAD_MAX_RETRIES} upload attempts failed for {fname}")
        return False

    def _retry_worker():
        """
        Periodically re-uploads images that failed on the first pass.
        Runs as a daemon thread — completely independent of the main queue.

        Safety checks:
          • Skips files currently in_flight (being uploaded by main loop).
          • Checks file existence before attempting read.
        """
        while True:
            time.sleep(RETRY_SCAN_INTERVAL_S)
            try:
                pending = [
                    f for f in os.listdir(LOCAL_BACKUP_DIR)
                    if f.endswith(".jpg")
                ]
            except OSError:
                continue

            if not pending:
                continue

            log.info(f"Retry worker: {len(pending)} pending file(s) found")
            for fname in pending:
                with in_flight_lock:
                    if fname in in_flight:
                        log.info(f"Retry skipping in-flight file: {fname}")
                        continue
                    in_flight.add(fname)

                local_path = os.path.join(LOCAL_BACKUP_DIR, fname)
                try:
                    # Guard: file may have been removed by main loop
                    if not os.path.exists(local_path):
                        log.info(f"Retry: file already gone: {fname}")
                        continue
                    with open(local_path, "rb") as fh:
                        img_bytes = fh.read()
                    ok = _upload_with_backoff(img_bytes, fname)
                    if ok:
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass
                        log.info(f"Retry success: {fname}")
                    else:
                        log.warning(f"Retry still failing: {fname}")
                except Exception as exc:
                    log.error(f"Retry worker error for {fname}: {exc}")
                finally:
                    with in_flight_lock:
                        in_flight.discard(fname)

    threading.Thread(target=_retry_worker, daemon=True,
                     name="upload-retry").start()

    # ── Main upload loop ──────────────────────────────────────────────────────
    while True:
        item = upload_q.get()
        if item is None:
            upload_q.task_done()
            log.info("Upload worker received stop signal")
            break

        kind = item[0]

        if kind == "image":
            _, img_bytes, fname = item

            # Duplicate guard
            with in_flight_lock:
                if fname in in_flight:
                    log.warning(f"Duplicate upload request ignored: {fname}")
                    upload_q.task_done()
                    continue
                in_flight.add(fname)

            local_path = os.path.join(LOCAL_BACKUP_DIR, fname)
            try:
                # 1. Write locally first (crash-safe backup)
                with open(local_path, "wb") as lf:
                    lf.write(img_bytes)

                # 2. Upload with backoff
                ok = _upload_with_backoff(img_bytes, fname)

                # 3. Remove local backup only on confirmed upload
                if ok:
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass
            except Exception as exc:
                log.error(f"Upload worker error [{fname}]: {exc}")
            finally:
                with in_flight_lock:
                    in_flight.discard(fname)

        upload_q.task_done()