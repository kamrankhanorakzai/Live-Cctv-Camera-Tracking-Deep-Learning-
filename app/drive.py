"""
app/drive.py — Google Drive client using OAuth token.json
"""
import io
import os
import time
import threading
import queue

from app.config import (
    DRIVE_FOLDER_ID, DRIVE_SUBFOLDER_NAME,
    ENCODINGS_FILENAME, TOKEN_FILE, LOCAL_BACKUP_DIR
)


class DriveClient:
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self):
        self.service      = None
        self.folder_id    = None
        self._enc_file_id = None
        self._connect()

    def _connect(self):
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            if not os.path.exists(TOKEN_FILE):
                raise FileNotFoundError(f"{TOKEN_FILE} not found. Run get_token.py first.")

            creds = Credentials.from_authorized_user_file(TOKEN_FILE, self.SCOPES)
            if creds.expired and creds.refresh_token:
                print("🔄 Refreshing Drive token...")
                creds.refresh(Request())
                with open(TOKEN_FILE, "w") as f:
                    f.write(creds.to_json())

            self.service = build("drive", "v3", credentials=creds)
            about        = self.service.about().get(fields="user").execute()
            print(f"✅ Drive → {about['user']['emailAddress']}")
            self.folder_id = self._resolve_subfolder()
            print(f"   Folder: {DRIVE_SUBFOLDER_NAME} ({self.folder_id})")

        except Exception as e:
            print(f"❌ Drive connection failed: {e}")

    def _resolve_subfolder(self):
        q   = (f"name='{DRIVE_SUBFOLDER_NAME}' and '{DRIVE_FOLDER_ID}' in parents "
               f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
        res = self.service.files().list(q=q, fields="files(id)").execute()
        fls = res.get("files", [])
        if fls:
            return fls[0]["id"]
        meta = {
            "name":     DRIVE_SUBFOLDER_NAME,
            "mimeType": "application/vnd.google-apps.folder",
            "parents":  [DRIVE_FOLDER_ID]
        }
        f = self.service.files().create(body=meta, fields="id").execute()
        print(f"   📁 Created subfolder: {DRIVE_SUBFOLDER_NAME}")
        return f["id"]

    def download_encodings(self):
        # Kept for backward compatibility — no longer used
        return [], 0

    def upload_encodings(self, encodings, count):
        # Kept for backward compatibility — no longer used
        pass

    def upload_image(self, img_bytes, filename):
        if not self.service:
            return False
        from googleapiclient.http import MediaIoBaseUpload
        try:
            meta  = {"name": filename, "parents": [self.folder_id]}
            media = MediaIoBaseUpload(io.BytesIO(img_bytes),
                                      mimetype="image/jpeg", resumable=False)
            f = self.service.files().create(
                body=meta, media_body=media, fields="id,name").execute()
            print(f"  ☁  Saved to Drive: {f['name']}")
            return True
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            return False

    def is_ready(self):
        return self.service is not None and self.folder_id is not None


# ── Upload worker thread ───────────────────────────────────────────────────────
def upload_worker(drive: DriveClient, upload_q: queue.Queue):
    global _enc_counter
    os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)

    # ── Retry thread: watches LOCAL_BACKUP_DIR for files that failed to upload ──
    # Runs independently — does not touch the main queue or any other logic.
    def _retry_worker():
        """
        Every 60 seconds, scan LOCAL_BACKUP_DIR for leftover .jpg files.
        These are images that were saved locally but whose Drive upload failed.
        Try to re-upload them; delete local file only on success.
        Completely separate from the main upload_worker loop — no shared state.
        """
        RETRY_INTERVAL = 60   # seconds between scans
        while True:
            time.sleep(RETRY_INTERVAL)
            try:
                files = [
                    f for f in os.listdir(LOCAL_BACKUP_DIR)
                    if f.endswith(".jpg")
                ]
            except OSError:
                continue

            if not files:
                continue

            print(f"  🔁 Retrying {len(files)} pending local image(s)…")
            for fname in files:
                local_path = os.path.join(LOCAL_BACKUP_DIR, fname)
                try:
                    with open(local_path, "rb") as fh:
                        img_bytes = fh.read()
                    ok = drive.upload_image(img_bytes, fname)
                    if ok:
                        os.remove(local_path)
                        print(f"  ✅ Retry uploaded & removed: {fname}")
                    else:
                        print(f"  ⏳ Retry failed, will try again: {fname}")
                except Exception as e:
                    print(f"  ⚠  Retry error for {fname}: {e}")

    # Start retry thread as daemon — dies automatically when main process exits
    threading.Thread(target=_retry_worker, daemon=True).start()

    # ── Main upload loop — unchanged ──────────────────────────────────────────
    while True:
        item = upload_q.get()
        if item is None:
            upload_q.task_done()
            break

        kind = item[0]

        if kind == "image":
            _, img_bytes, fname = item
            # Save locally first as backup
            local_path = os.path.join(LOCAL_BACKUP_DIR, fname)
            with open(local_path, "wb") as lf:
                lf.write(img_bytes)
            # Upload to Drive
            ok = drive.upload_image(img_bytes, fname)
            if ok:
                try:
                    os.remove(local_path)
                except OSError:
                    pass

        upload_q.task_done()