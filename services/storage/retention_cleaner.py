"""Image retention cleanup for local storage."""

import json
import os
import re
import threading
import time
from datetime import date, datetime

from config import (
    IMAGE_DEAD_LETTER_RETENTION_DAYS,
    LOG_DIR,
    LOG_FILE_PREFIX,
    LOG_RETENTION_DAYS,
    SCALE_DATA_DIR,
    SCALE_DATA_RETENTION_DAYS,
    MQTT_DEAD_LETTER_RETENTION_DAYS,
    SERVICE_DIR,
)

CLEANED_SUFFIX = "--Cleaned"
COMPACT_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]")
SEPARATED_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)")


class ImageRetentionCleaner:
    """Deletes old image files from configured roots on a low-frequency schedule."""

    def __init__(self, roots, retention_days, check_interval_seconds, extensions, log_fn=None):
        self.roots = list(roots)
        self.retention_days = retention_days
        self.check_interval_seconds = check_interval_seconds
        self.extensions = {ext.lower() for ext in extensions}
        self.log_fn = log_fn
        self._stop_event = threading.Event()
        self._thread = None

    def _log(self, level, msg):
        if self.log_fn:
            self.log_fn(level, msg)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log("INFO", "ImageRetentionCleaner started")

    def stop(self, timeout=3.0):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._log("INFO", "ImageRetentionCleaner stopped")

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self._log("ERROR", f"Image retention failed: {exc}")
            self._stop_event.wait(self.check_interval_seconds)

    def _is_image_name(self, filename):
        return os.path.splitext(filename)[1].lower() in self.extensions

    def _subtree_has_images(self, root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))]
            if any(self._is_image_name(filename) for filename in filenames):
                return True
        return False

    def _iter_dirs_deepest_first(self, root):
        dirs = []
        for dirpath, dirnames, _ in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))]
            for dirname in dirnames:
                dirs.append(os.path.join(dirpath, dirname))
        dirs.sort(key=lambda path: path.count(os.sep), reverse=True)
        return dirs

    @staticmethod
    def _make_date(year, month, day):
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            return None

    def _date_from_path(self, fpath, root):
        rel = os.path.relpath(fpath, root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            return None
        year, month, day = parts[:3]
        if not (year.isdigit() and month.isdigit() and day.isdigit()):
            return None
        return self._make_date(year, month, day)

    def _date_from_filename(self, filename):
        for regex in (COMPACT_DATE_RE, SEPARATED_DATE_RE):
            match = regex.search(filename)
            if not match:
                continue
            parsed = self._make_date(*match.groups())
            if parsed:
                return parsed
        return None

    def _image_age_source(self, fpath, root, filename, stat):
        parsed = self._date_from_path(fpath, root)
        if parsed:
            return parsed, "path_date"
        parsed = self._date_from_filename(filename)
        if parsed:
            return parsed, "filename_date"
        return datetime.fromtimestamp(stat.st_mtime), "mtime"

    def _pending_upload_paths(self):
        pending_file = os.path.join(SERVICE_DIR, "storage", "upload_pending.jsonl")
        paths = set()
        if not os.path.exists(pending_file):
            return paths
        try:
            with open(pending_file, "r") as fp:
                for line in fp:
                    try:
                        task = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fpath = task.get("fpath")
                    if fpath:
                        paths.add(os.path.abspath(fpath))
        except OSError as exc:
            self._log("WARNING", f"Image retention pending upload read failed: {exc}")
        return paths

    def _tag_cleaned_directories(self):
        tagged = 0
        tag_failed = 0
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for dirpath in self._iter_dirs_deepest_first(root):
                dirname = os.path.basename(dirpath)
                if dirname.endswith(CLEANED_SUFFIX) or os.path.islink(dirpath):
                    continue
                if self._subtree_has_images(dirpath):
                    continue
                target = dirpath + CLEANED_SUFFIX
                if os.path.exists(target):
                    tag_failed += 1
                    self._log("WARNING", f"Cleaned tag target already exists, skipping: {target}")
                    continue
                try:
                    os.rename(dirpath, target)
                    tagged += 1
                except OSError as exc:
                    tag_failed += 1
                    self._log("WARNING", f"Cleaned tag failed for {dirpath}: {exc}")
        return tagged, tag_failed

    def run_once(self, now=None):
        now_ts = time.time() if now is None else now
        cutoff = now_ts - (self.retention_days * 24 * 60 * 60)
        cutoff_date = datetime.fromtimestamp(cutoff).date()
        scanned = 0
        deleted = 0
        failed = 0
        reclaimed = 0
        deleted_by_path_date = 0
        deleted_by_filename_date = 0
        deleted_by_mtime = 0
        skipped_pending_uploads = 0
        pending_upload_paths = self._pending_upload_paths()

        for root in self.roots:
            if not os.path.isdir(root):
                self._log("INFO", f"Image retention root missing, skipping: {root}")
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))]
                for filename in filenames:
                    if not self._is_image_name(filename):
                        continue
                    fpath = os.path.join(dirpath, filename)
                    if os.path.islink(fpath):
                        continue
                    if os.path.abspath(fpath) in pending_upload_paths:
                        skipped_pending_uploads += 1
                        continue
                    scanned += 1
                    try:
                        stat = os.stat(fpath, follow_symlinks=False)
                    except OSError as exc:
                        failed += 1
                        self._log("WARNING", f"Image retention stat failed for {fpath}: {exc}")
                        continue
                    age_value, age_source = self._image_age_source(fpath, root, filename, stat)
                    if age_source == "mtime":
                        should_delete = age_value.timestamp() < cutoff
                    else:
                        should_delete = age_value < cutoff_date
                    if not should_delete:
                        continue
                    try:
                        os.remove(fpath)
                        deleted += 1
                        reclaimed += stat.st_size
                        if age_source == "path_date":
                            deleted_by_path_date += 1
                        elif age_source == "filename_date":
                            deleted_by_filename_date += 1
                        else:
                            deleted_by_mtime += 1
                    except OSError as exc:
                        failed += 1
                        self._log("WARNING", f"Image retention delete failed for {fpath}: {exc}")

        tagged, tag_failed = self._tag_cleaned_directories()

        self._log(
            "INFO",
            f"Image retention complete: scanned={scanned} deleted={deleted} failed={failed} "
            f"tagged={tagged} tag_failed={tag_failed} "
            f"deleted_by_path_date={deleted_by_path_date} "
            f"deleted_by_filename_date={deleted_by_filename_date} deleted_by_mtime={deleted_by_mtime} "
            f"skipped_pending_uploads={skipped_pending_uploads} "
            f"reclaimed={reclaimed / (1024 * 1024):.1f}MB cutoff_days={self.retention_days}",
        )
        return {
            "scanned": scanned,
            "deleted": deleted,
            "failed": failed,
            "reclaimed": reclaimed,
            "tagged": tagged,
            "tag_failed": tag_failed,
            "deleted_by_path_date": deleted_by_path_date,
            "deleted_by_filename_date": deleted_by_filename_date,
            "deleted_by_mtime": deleted_by_mtime,
            "skipped_pending_uploads": skipped_pending_uploads,
        }


class StorageMaintenance:
    """Apply age retention to logs and dead-letter records."""

    def __init__(self, check_interval_seconds, log_fn=None):
        self.check_interval_seconds = check_interval_seconds
        self.log_fn = log_fn
        self._stop_event = threading.Event()
        self._thread = None

    def _log(self, level, msg):
        if self.log_fn:
            self.log_fn(level, msg)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout=3.0):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self._log("ERROR", f"Storage maintenance failed: {exc}")
            self._stop_event.wait(self.check_interval_seconds)

    @staticmethod
    def _remove_older_than(root, retention_days, predicate, now):
        cutoff = now - retention_days * 86400
        deleted = 0
        if not os.path.isdir(root):
            return deleted
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if not predicate(filename) or os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    deleted += 1
            except OSError:
                continue
        return deleted

    @staticmethod
    def _dated_name(name):
        try:
            return datetime.strptime(name, "%Y-%m-%d").date()
        except ValueError:
            return None

    def _remove_expired_logs(self, now):
        cutoff = datetime.fromtimestamp(now - LOG_RETENTION_DAYS * 86400).date()
        deleted = directories_deleted = 0
        if not os.path.isdir(LOG_DIR):
            return deleted, directories_deleted
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            if os.path.islink(path) or not os.path.isdir(path):
                continue
            date_value = self._dated_name(name)
            if date_value is None or date_value > cutoff:
                continue
            log_path = os.path.join(path, f"{LOG_FILE_PREFIX}.log")
            try:
                if os.path.isfile(log_path) and not os.path.islink(log_path):
                    os.remove(log_path)
                    deleted += 1
                if not os.listdir(path):
                    os.rmdir(path)
                    directories_deleted += 1
            except OSError:
                continue
        return deleted, directories_deleted

    def _remove_expired_scale_databases(self, now):
        cutoff = datetime.fromtimestamp(now - SCALE_DATA_RETENTION_DAYS * 86400).date()
        deleted = 0
        if not os.path.isdir(SCALE_DATA_DIR):
            return deleted
        for name in os.listdir(SCALE_DATA_DIR):
            match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.db(?:-(?:wal|shm))?", name)
            if not match:
                continue
            date_value = self._dated_name(match.group(1))
            if date_value is None or date_value > cutoff:
                continue
            path = os.path.join(SCALE_DATA_DIR, name)
            try:
                if os.path.isfile(path) and not os.path.islink(path):
                    os.remove(path)
                    deleted += 1
            except OSError:
                continue
        return deleted

    def run_once(self, now=None):
        now = time.time() if now is None else now
        log_deleted, log_directories_deleted = self._remove_expired_logs(now)
        scale_databases_deleted = self._remove_expired_scale_databases(now)
        dead_letter_dir = os.path.join(SERVICE_DIR, "storage", "dead-letter")
        mqtt_deleted = self._remove_older_than(
            dead_letter_dir,
            MQTT_DEAD_LETTER_RETENTION_DAYS,
            lambda name: name.startswith("mqtt-") and name.endswith(".jsonl"),
            now,
        )
        image_deleted = self._remove_older_than(
            dead_letter_dir,
            IMAGE_DEAD_LETTER_RETENTION_DAYS,
            lambda name: (name.startswith("minio-") or name.startswith("malformed-")) and name.endswith(".jsonl"),
            now,
        )
        self._log(
            "INFO",
            f"Storage maintenance complete: logs_deleted={log_deleted} "
            f"log_directories_deleted={log_directories_deleted} "
            f"scale_databases_deleted={scale_databases_deleted} "
            f"mqtt_dead_letters_deleted={mqtt_deleted} image_dead_letters_deleted={image_deleted}",
        )
        return {
            "logs_deleted": log_deleted,
            "log_directories_deleted": log_directories_deleted,
            "scale_databases_deleted": scale_databases_deleted,
            "mqtt_deleted": mqtt_deleted,
            "image_deleted": image_deleted,
        }
