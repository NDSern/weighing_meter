"""Image save worker — saves images locally and uploads to MinIO."""

import json
import io
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from minio import Minio
from minio.error import S3Error

from config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    RESULT_JPEG_QUALITY,
    SERVICE_DIR,
)

_log_fn = None


def set_log_fn(log_fn):
    """Set the logging function to use."""
    global _log_fn
    _log_fn = log_fn


def log(level: str, msg: str):
    """Log using the configured log function."""
    if _log_fn:
        _log_fn(level, msg)


_minio = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

_pending_file = os.path.join(SERVICE_DIR, "storage", "upload_pending.jsonl")
_upload_queue = queue.Queue()
_pending_lock = threading.Lock()
_pending_tasks = {}
_worker_lock = threading.Lock()
_worker_thread = None
_worker_started = False
_stop_event = threading.Event()
_sync_executor_lock = threading.Lock()
_sync_executor = None


class ImageSaveWorker:
    """Save images locally first, then upload with persisted retry queue."""

    @staticmethod
    def start_upload_worker():
        ImageSaveWorker._ensure_worker()

    @staticmethod
    def start_save_thread(items_list):
        ImageSaveWorker._ensure_worker()
        for item in items_list:
            fpath, frame, object_key = item
            item[1] = None
            if frame is None:
                continue
            if ImageSaveWorker._save_local(fpath, frame):
                ImageSaveWorker._enqueue_upload(fpath, object_key)
        return None

    @staticmethod
    def save_and_upload_now(items_list):
        """Save and upload all images before returning. Used when MQTT must wait for valid image URLs."""
        ImageSaveWorker._ensure_worker()
        work_items = []
        for item in items_list:
            fpath, frame, object_key = item
            item[1] = None
            if frame is not None:
                work_items.append((fpath, frame, object_key))
        if not work_items:
            return False

        ok = True
        executor = ImageSaveWorker._get_sync_executor()
        futures = [executor.submit(ImageSaveWorker._encode_save_upload_now_one, *work_item) for work_item in work_items]
        for future in as_completed(futures):
            ok = bool(future.result()) and ok
        return ok

    @staticmethod
    def save_and_enqueue_upload(items_list):
        """Persist images locally, persist upload tasks, then return without network dependency."""
        ImageSaveWorker._ensure_worker()
        work_items = []
        for item in items_list:
            fpath, frame, object_key = item
            item[1] = None
            if frame is not None:
                work_items.append((fpath, frame, object_key))
        if not work_items:
            return False

        ok = True
        for fpath, frame, object_key in work_items:
            if ImageSaveWorker._encode_save_local(fpath, frame):
                ImageSaveWorker._enqueue_upload(fpath, object_key)
            else:
                ok = False
        return ok

    @staticmethod
    def save_local_only(fpath, frame):
        """Persist one image locally without adding a MinIO upload task."""
        ImageSaveWorker._ensure_worker()
        if frame is None:
            return False
        return ImageSaveWorker._encode_save_local(fpath, frame)

    @staticmethod
    def _get_sync_executor():
        global _sync_executor
        with _sync_executor_lock:
            if _sync_executor is None:
                _sync_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="image-publish")
            return _sync_executor

    @staticmethod
    def wait_for_pending(timeout=15.0):
        """Wait briefly for queued uploads during shutdown."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _upload_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.2)
        log("WARNING", f"Image upload wait timed out with {_upload_queue.unfinished_tasks} task(s) pending")
        return False

    @staticmethod
    def has_pending(object_keys):
        ImageSaveWorker._ensure_worker()
        with _pending_lock:
            return any(object_key in _pending_tasks for object_key in object_keys)

    @staticmethod
    def _ensure_worker():
        global _worker_thread, _worker_started
        with _worker_lock:
            if _worker_started:
                return
            os.makedirs(os.path.dirname(_pending_file), exist_ok=True)
            ImageSaveWorker._load_pending_uploads()
            _stop_event.clear()
            _worker_thread = threading.Thread(target=ImageSaveWorker._upload_loop, daemon=True)
            _worker_thread.start()
            _worker_started = True

    @staticmethod
    def _save_local(fpath, frame):
        fname = os.path.basename(fpath)
        try:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            cv2.imwrite(fpath, frame)
            log("SAVE", f"Saved {fname}")
            return True
        except Exception as exc:
            log("ERROR", f"Failed to save {fname}: {exc}")
            return False
        finally:
            del frame

    @staticmethod
    def _save_and_upload_now_one(fpath, frame, object_key):
        if not ImageSaveWorker._save_local(fpath, frame):
            return False
        return ImageSaveWorker._upload_now(fpath, object_key)

    @staticmethod
    def _encode_save_local(fpath, frame):
        fname = os.path.basename(fpath)
        started_at = time.time()
        try:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, RESULT_JPEG_QUALITY])
            if not ok:
                log("ERROR", f"Failed to encode {fname}")
                return False
            data = encoded.tobytes()
            with open(fpath, "wb") as fp:
                fp.write(data)
                fp.flush()
                os.fsync(fp.fileno())
            log("SAVE", f"Saved {fname} bytes={len(data)} encode_write={(time.time() - started_at) * 1000:.0f}ms")
            return True
        except Exception as exc:
            log("ERROR", f"Failed to save {fname}: {exc}")
            return False
        finally:
            del frame

    @staticmethod
    def _upload_now(fpath, object_key):
        if not os.path.exists(fpath):
            log("ERROR", f"MinIO upload skipped; local file missing: {fpath}")
            return False
        log("MINIO", f"Uploading {object_key} ...")
        try:
            res = _minio.fput_object(MINIO_BUCKET, object_key, fpath, content_type="image/jpeg")
            log("MINIO", f"Upload OK — etag={res.etag}")
            return True
        except S3Error as exc:
            log("ERROR", f"MinIO upload failed for {object_key}: {exc}")
        except Exception as exc:
            log("ERROR", f"MinIO upload error for {object_key}: {exc}")
        ImageSaveWorker._enqueue_upload(fpath, object_key)
        log("MINIO", f"Queued failed upload for retry: {object_key}")
        return False

    @staticmethod
    def _encode_save_upload_now_one(fpath, frame, object_key):
        fname = os.path.basename(fpath)
        started_at = time.time()
        try:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, RESULT_JPEG_QUALITY])
            if not ok:
                log("ERROR", f"Failed to encode {fname}")
                return False
            data = encoded.tobytes()
            with open(fpath, "wb") as fp:
                fp.write(data)
            saved_at = time.time()
            log("SAVE", f"Saved {fname} bytes={len(data)} encode_write={(saved_at - started_at) * 1000:.0f}ms")
        except Exception as exc:
            log("ERROR", f"Failed to save {fname}: {exc}")
            return False
        finally:
            del frame

        log("MINIO", f"Uploading {object_key} ...")
        try:
            upload_started_at = time.time()
            res = _minio.put_object(
                MINIO_BUCKET,
                object_key,
                io.BytesIO(data),
                length=len(data),
                content_type="image/jpeg",
            )
            log("MINIO", f"Upload OK — etag={res.etag} bytes={len(data)} upload={(time.time() - upload_started_at) * 1000:.0f}ms")
            return True
        except S3Error as exc:
            log("ERROR", f"MinIO upload failed for {object_key}: {exc}")
        except Exception as exc:
            log("ERROR", f"MinIO upload error for {object_key}: {exc}")
        ImageSaveWorker._enqueue_upload(fpath, object_key)
        log("MINIO", f"Queued failed upload for retry: {object_key}")
        return False

    @staticmethod
    def _load_pending_uploads():
        if not os.path.exists(_pending_file):
            return
        with _pending_lock:
            with open(_pending_file, "r") as fp:
                for line in fp:
                    try:
                        task = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fpath = task.get("fpath")
                    object_key = task.get("object_key")
                    if not fpath or not object_key or not os.path.exists(fpath):
                        continue
                    _pending_tasks[object_key] = {"fpath": fpath, "object_key": object_key}
            for task in _pending_tasks.values():
                _upload_queue.put(task)
        if _pending_tasks:
            log("MINIO", f"Loaded {len(_pending_tasks)} pending upload(s)")

    @staticmethod
    def _persist_pending_locked():
        tmp_path = _pending_file + ".tmp"
        os.makedirs(os.path.dirname(_pending_file), exist_ok=True)
        with open(tmp_path, "w") as fp:
            for task in _pending_tasks.values():
                fp.write(json.dumps(task, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, _pending_file)

    @staticmethod
    def _enqueue_upload(fpath, object_key):
        task = {"fpath": fpath, "object_key": object_key}
        with _pending_lock:
            _pending_tasks[object_key] = task
            ImageSaveWorker._persist_pending_locked()
        _upload_queue.put(task)

    @staticmethod
    def _mark_uploaded(object_key):
        with _pending_lock:
            _pending_tasks.pop(object_key, None)
            ImageSaveWorker._persist_pending_locked()

    @staticmethod
    def _upload_loop():
        while not _stop_event.is_set():
            try:
                task = _upload_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ImageSaveWorker._upload_task(task)
            finally:
                _upload_queue.task_done()

    @staticmethod
    def _upload_task(task):
        fpath = task["fpath"]
        object_key = task["object_key"]
        with _pending_lock:
            pending = _pending_tasks.get(object_key)
            if pending != task:
                return
        if not os.path.exists(fpath):
            log("ERROR", f"MinIO upload skipped; local file missing: {fpath}")
            return

        log("MINIO", f"Uploading {object_key} ...")
        try:
            res = _minio.fput_object(MINIO_BUCKET, object_key, fpath, content_type="image/jpeg")
            log("MINIO", f"Upload OK — etag={res.etag}")
            ImageSaveWorker._mark_uploaded(object_key)
        except S3Error as exc:
            log("ERROR", f"MinIO upload failed for {object_key}: {exc}")
            time.sleep(2.0)
            _upload_queue.put(task)
        except Exception as exc:
            log("ERROR", f"MinIO upload error for {object_key}: {exc}")
            time.sleep(2.0)
            _upload_queue.put(task)
