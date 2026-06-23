"""Image retention cleanup for local storage."""

import json
import os
import re
import threading
import time
from datetime import date, datetime

from config import SERVICE_DIR

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
            self.run_once()
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
