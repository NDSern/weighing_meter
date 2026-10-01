"""Image retention cleanup for local storage."""

import gzip
import json
import hashlib
import os
import re
import shutil
import subprocess
import time
from datetime import date, datetime, timedelta

from config import (
    IMAGE_DEAD_LETTER_RETENTION_DAYS,
    LOG_COMPRESSION_ENABLED,
    LOG_COMPRESS_AFTER_DAYS,
    LOG_DIR,
    LOG_FILE_PREFIX,
    LOG_RETENTION_DAYS,
    MINIO_BUCKET,
    SCALE_DATA_DIR,
    SCALE_DATA_RETENTION_DAYS,
    MQTT_DEAD_LETTER_RETENTION_DAYS,
    SERVICE_DIR,
)
from services.runtime.background_worker import BackgroundWorker
from services.storage.dated_tree import (
    date_from_filename,
    date_from_relative_path,
    iter_dirs_deepest_first,
    make_date,
)

CLEANED_SUFFIX = "--Cleaned"
PRESSURE_CLEANUP_GRACE_SECONDS = 60 * 60
# Do not tag a directory that may still be receiving its first write.
TAG_GRACE_SECONDS = 60 * 60


class ImageRetentionCleaner(BackgroundWorker):
    """Deletes old image files from configured roots on a low-frequency schedule."""

    worker_name = "ImageRetentionCleaner"
    error_label = "Image retention failed"

    def __init__(
        self, roots, retention_days, check_interval_seconds, extensions, log_fn=None,
        pressure_free_bytes=None,
    ):
        super().__init__(check_interval_seconds, log_fn)
        self.roots = list(roots)
        self.retention_days = retention_days
        self.extensions = {ext.lower() for ext in extensions}
        self.pressure_free_bytes = pressure_free_bytes

    def _is_image_name(self, filename):
        return os.path.splitext(filename)[1].lower() in self.extensions

    def _subtree_has_images(self, root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))]
            if any(self._is_image_name(filename) for filename in filenames):
                return True
        return False

    def _iter_dirs_deepest_first(self, root):
        return iter_dirs_deepest_first(root)

    @staticmethod
    def _make_date(year, month, day):
        return make_date(year, month, day)

    def _date_from_path(self, fpath, root):
        return date_from_relative_path(os.path.relpath(fpath, root))

    def _date_from_filename(self, filename):
        return date_from_filename(filename)

    def _image_age_source(self, fpath, root, filename, stat):
        parsed = self._date_from_path(fpath, root)
        if parsed:
            return parsed, "path_date"
        parsed = self._date_from_filename(filename)
        if parsed:
            return parsed, "filename_date"
        return datetime.fromtimestamp(stat.st_mtime), "mtime"

    def _pending_image_paths(self):
        paths = set()
        pending_files = (
            ("upload", os.path.join(SERVICE_DIR, "storage", "upload_pending.jsonl")),
            ("publish", os.path.join(SERVICE_DIR, "storage", "publish_pending.jsonl")),
        )
        for queue_name, pending_file in pending_files:
            if not os.path.exists(pending_file):
                continue
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
                        for path in task.get("image_paths") or []:
                            paths.add(os.path.abspath(path))
            except OSError as exc:
                self._log(
                    "WARNING",
                    f"Image retention pending {queue_name} read failed: {exc}",
                )
        return paths

    def _tag_cleaned_directories(self, now_ts=None):
        now_ts = time.time() if now_ts is None else now_ts
        today = datetime.fromtimestamp(now_ts).date()
        grace_cutoff = now_ts - TAG_GRACE_SECONDS
        tagged = 0
        tag_failed = 0
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for dirpath in self._iter_dirs_deepest_first(root):
                dirname = os.path.basename(dirpath)
                if dirname.endswith(CLEANED_SUFFIX) or os.path.islink(dirpath):
                    continue
                # A writer may have created the day directory but not yet
                # written its first frame; renaming it here would lose the
                # evidence. Skip today/yesterday and anything touched recently.
                rel_parts = os.path.relpath(dirpath, root).split(os.sep)
                dir_date = make_date(*rel_parts) if len(rel_parts) == 3 else None
                if dir_date is not None and (today - dir_date).days <= 1:
                    continue
                try:
                    dir_stat = os.stat(dirpath, follow_symlinks=False)
                except OSError as exc:
                    self._log("WARNING", f"Cleaned tag stat failed for {dirpath}: {exc}")
                    continue
                if dir_stat.st_mtime > grace_cutoff:
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

    def _pressure_cleanup(self, pending_upload_paths, now_ts):
        if self.pressure_free_bytes is None:
            return 0, 0, 0
        storage_root = next((root for root in self.roots if os.path.isdir(root)), None)
        if storage_root is None:
            return 0, 0, 0
        try:
            free_bytes = shutil.disk_usage(storage_root).free
        except OSError as exc:
            self._log("WARNING", f"Image pressure cleanup free-space check failed: {exc}")
            return 0, 1, 0
        required = self.pressure_free_bytes - free_bytes
        if required <= 0:
            return 0, 0, 0

        candidates = []
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [
                    name for name in dirnames
                    if not os.path.islink(os.path.join(dirpath, name))
                ]
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    fpath = os.path.join(dirpath, filename)
                    if os.path.islink(fpath) or os.path.abspath(fpath) in pending_upload_paths:
                        continue
                    try:
                        stat = os.stat(fpath, follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.st_mtime >= now_ts - PRESSURE_CLEANUP_GRACE_SECONDS:
                        continue
                    age_value, _source = self._image_age_source(fpath, root, filename, stat)
                    if isinstance(age_value, date) and not isinstance(age_value, datetime):
                        age_value = datetime.combine(age_value, datetime.min.time())
                    candidates.append((age_value.timestamp(), fpath, stat.st_size))

        deleted = failed = reclaimed = 0
        for _age, fpath, size in sorted(candidates):
            if reclaimed >= required:
                break
            try:
                os.remove(fpath)
                deleted += 1
                reclaimed += size
            except OSError as exc:
                failed += 1
                self._log("WARNING", f"Image pressure cleanup delete failed for {fpath}: {exc}")
        return deleted, failed, reclaimed

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
        pending_image_paths = self._pending_image_paths()

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
                    if os.path.abspath(fpath) in pending_image_paths:
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

        pressure_deleted, pressure_failed, pressure_reclaimed = self._pressure_cleanup(
            pending_image_paths, now_ts
        )
        tagged, tag_failed = self._tag_cleaned_directories(now_ts)

        self._log(
            "INFO",
            f"Image retention complete: scanned={scanned} deleted={deleted} failed={failed} "
            f"tagged={tagged} tag_failed={tag_failed} "
            f"deleted_by_path_date={deleted_by_path_date} "
            f"deleted_by_filename_date={deleted_by_filename_date} deleted_by_mtime={deleted_by_mtime} "
            f"skipped_pending_uploads={skipped_pending_uploads} "
            f"reclaimed={reclaimed / (1024 * 1024):.1f}MB cutoff_days={self.retention_days} "
            f"pressure_deleted={pressure_deleted} pressure_failed={pressure_failed} "
            f"pressure_reclaimed={pressure_reclaimed / (1024 * 1024):.1f}MB",
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
            "pressure_deleted": pressure_deleted,
            "pressure_failed": pressure_failed,
            "pressure_reclaimed": pressure_reclaimed,
        }


class VerifiedMinioCacheCleaner(BackgroundWorker):
    """Evict published local images only after exact MinIO verification."""

    worker_name = "VerifiedMinioCacheCleaner"
    error_label = "MinIO cache cleanup failed"

    def __init__(self, root, retention_days, check_interval_seconds, client_factory, log_fn=None):
        super().__init__(check_interval_seconds, log_fn)
        self.root = os.path.abspath(root)
        self.retention_days = retention_days
        self.client_factory = client_factory

    @staticmethod
    def _md5(path):
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _date_from_relative_path(relative_path):
        return date_from_relative_path(relative_path)

    @staticmethod
    def _pending_image_paths():
        return ImageRetentionCleaner([], 0, 0, set())._pending_image_paths()

    def run_once(self, now=None, client=None):
        now = time.time() if now is None else now
        cutoff_date = datetime.fromtimestamp(now - self.retention_days * 86400).date()
        pending_paths = self._pending_image_paths()
        remote = client or self.client_factory()
        scanned = verified = deleted = failed = skipped_pending = remote_mismatch = 0
        reclaimed = 0
        if not os.path.isdir(self.root):
            return {"scanned": 0, "verified": 0, "deleted": 0, "failed": 0,
                    "skipped_pending": 0, "remote_mismatch": 0,
                    "reclaimed": 0}
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))]
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                path = os.path.abspath(os.path.join(dirpath, filename))
                if path in pending_paths or os.path.islink(path):
                    skipped_pending += 1
                    continue
                relative_path = os.path.relpath(path, self.root)
                image_date = self._date_from_relative_path(relative_path)
                if image_date is None or image_date >= cutoff_date:
                    continue
                scanned += 1
                object_key = "storage/weighbridge/" + relative_path.replace(os.sep, "/")
                try:
                    stat = remote.stat_object(MINIO_BUCKET, object_key)
                    local_size = os.path.getsize(path)
                    remote_etag = str(getattr(stat, "etag", "")).strip('"').lower()
                    if stat.size != local_size or not remote_etag or "-" in remote_etag:
                        remote_mismatch += 1
                        continue
                    if self._md5(path) != remote_etag:
                        remote_mismatch += 1
                        continue
                    verified += 1
                    os.remove(path)
                    deleted += 1
                    reclaimed += local_size
                except OSError as exc:
                    failed += 1
                    self._log("WARNING", f"MinIO cache cleanup failed path={path}: {exc}")
                except Exception as exc:
                    failed += 1
                    self._log("WARNING", f"MinIO cache verification failed key={object_key}: {exc}")
        self._log(
            "INFO",
            f"MinIO cache cleanup complete: scanned={scanned} verified={verified} deleted={deleted} "
            f"failed={failed} skipped_pending={skipped_pending} "
            f"remote_mismatch={remote_mismatch} reclaimed={reclaimed / (1024 * 1024):.1f}MB "
            f"cutoff_days={self.retention_days}",
        )
        return {"scanned": scanned, "verified": verified, "deleted": deleted, "failed": failed,
                "skipped_pending": skipped_pending,
                "remote_mismatch": remote_mismatch, "reclaimed": reclaimed}


class DiagnosticArchiveCleaner(BackgroundWorker):
    """Archive completed diagnostic days, then expire their archives."""

    worker_name = "DiagnosticArchiveCleaner"
    error_label = "Diagnostic archive cleanup failed"

    def __init__(self, roots, archive_after_days, retention_days, check_interval_seconds, log_fn=None):
        super().__init__(check_interval_seconds, log_fn)
        self.roots = list(roots)
        self.archive_after_days = archive_after_days
        self.retention_days = retention_days

    @staticmethod
    def _listdir(path):
        try:
            return os.listdir(path)
        except OSError:
            return []

    @staticmethod
    def _day_paths(root):
        for year in DiagnosticArchiveCleaner._listdir(root):
            year_path = os.path.join(root, year)
            if not year.isdigit() or not os.path.isdir(year_path) or os.path.islink(year_path):
                continue
            for month in DiagnosticArchiveCleaner._listdir(year_path):
                month_path = os.path.join(year_path, month)
                if not month.isdigit() or not os.path.isdir(month_path) or os.path.islink(month_path):
                    continue
                for name in DiagnosticArchiveCleaner._listdir(month_path):
                    path = os.path.join(month_path, name)
                    if name.endswith(".tar.zst"):
                        day = make_date(year, month, name[:-8])
                        if day is not None and os.path.isfile(path) and not os.path.islink(path):
                            yield day, path, True
                    elif os.path.isdir(path) and not os.path.islink(path):
                        day = make_date(year, month, name)
                        if day is not None:
                            yield day, path, False

    def _run_archive_command(self, command):
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        while process.poll() is None:
            if self._stop_event.wait(0.2):
                process.terminate()
                process.communicate()
                raise subprocess.SubprocessError("Diagnostic archive interrupted by shutdown")
        _, stderr = process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr)

    def _archive_day(self, day_path):
        parent = os.path.dirname(day_path)
        name = os.path.basename(day_path)
        archive_path = day_path + ".tar.zst"
        temp_path = archive_path + ".tmp"
        if os.path.exists(archive_path):
            # A previous run finished the archive but crashed before removing
            # the source day directory. The archive is authoritative; clear the
            # leftover source so this day does not wedge every later pass.
            shutil.rmtree(day_path)
            return archive_path
        try:
            self._run_archive_command(["tar", "--zstd", "-cf", temp_path, "-C", parent, name])
            self._run_archive_command(["tar", "--zstd", "-tf", temp_path])
            os.replace(temp_path, archive_path)
            shutil.rmtree(day_path)
            return archive_path
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    def run_once(self, now=None):
        today = datetime.fromtimestamp(time.time() if now is None else now).date()
        archive_cutoff = today - timedelta(days=self.archive_after_days)
        delete_cutoff = today - timedelta(days=self.retention_days)
        archived = archive_deleted = failed = 0
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for day, path, is_archive in self._day_paths(root):
                try:
                    if is_archive:
                        if day <= delete_cutoff:
                            os.remove(path)
                            archive_deleted += 1
                    elif day <= archive_cutoff:
                        self._archive_day(path)
                        archived += 1
                except (OSError, subprocess.SubprocessError) as exc:
                    failed += 1
                    self._log("WARNING", f"Diagnostic archive cleanup failed path={path}: {exc}")
        self._log(
            "INFO",
            f"Diagnostic archive cleanup complete: archived={archived} archive_deleted={archive_deleted} "
            f"failed={failed} archive_after_days={self.archive_after_days} retention_days={self.retention_days}",
        )
        return {"archived": archived, "archive_deleted": archive_deleted, "failed": failed}


class StorageMaintenance(BackgroundWorker):
    """Apply age retention to logs and dead-letter records."""

    worker_name = "StorageMaintenance"
    error_label = "Storage maintenance failed"
    announce_lifecycle = False

    def __init__(self, check_interval_seconds, log_fn=None):
        super().__init__(check_interval_seconds, log_fn)

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

    @staticmethod
    def _gzip_matches_source(source_path, gzip_path):
        source_digest = hashlib.sha256()
        gzip_digest = hashlib.sha256()
        try:
            with open(source_path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    source_digest.update(chunk)
            with gzip.open(gzip_path, "rb") as compressed:
                for chunk in iter(lambda: compressed.read(1024 * 1024), b""):
                    gzip_digest.update(chunk)
        except (OSError, EOFError, gzip.BadGzipFile):
            return False
        return source_digest.digest() == gzip_digest.digest()

    def _compress_log(self, path):
        compressed_path = path + ".gz"
        temp_path = compressed_path + ".tmp"
        if os.path.exists(compressed_path):
            if os.path.islink(compressed_path) or not os.path.isfile(compressed_path):
                return False
            if not self._gzip_matches_source(path, compressed_path):
                return False
            os.remove(path)
            return True
        try:
            with open(path, "rb") as source, open(temp_path, "xb") as raw_output:
                with gzip.GzipFile(
                    filename=os.path.basename(path), mode="wb", fileobj=raw_output,
                    compresslevel=1, mtime=0,
                ) as compressed:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
                raw_output.flush()
                os.fsync(raw_output.fileno())
            if not self._gzip_matches_source(path, temp_path):
                return False
            os.replace(temp_path, compressed_path)
            os.remove(path)
            return True
        except OSError:
            return False
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def _open_file_paths():
        paths = set()
        proc_root = "/proc"
        try:
            pids = os.listdir(proc_root)
        except OSError:
            return paths
        for pid in pids:
            if not pid.isdigit():
                continue
            fd_root = os.path.join(proc_root, pid, "fd")
            try:
                fds = os.listdir(fd_root)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(os.path.join(fd_root, fd))
                except OSError:
                    continue
                if target.endswith(" (deleted)"):
                    target = target[:-10]
                if os.path.isabs(target):
                    paths.add(os.path.abspath(target))
        return paths

    def _compress_completed_logs(self, now):
        cutoff = datetime.fromtimestamp(now).date() - timedelta(days=LOG_COMPRESS_AFTER_DAYS)
        compressed = failed = 0
        if not LOG_COMPRESSION_ENABLED:
            return compressed, failed
        open_paths = self._open_file_paths()
        if not os.path.isdir(LOG_DIR):
            return compressed, failed
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            candidates = []
            match = re.fullmatch(
                rf"{re.escape(LOG_FILE_PREFIX)}_(\d{{4}}-\d{{2}}-\d{{2}})\.log", name,
            )
            if match:
                candidates.append((self._dated_name(match.group(1)), path))
            else:
                match = re.fullmatch(r"resource-watchdog\.(\d{8})_\d{6}\.jsonl", name)
                if match:
                    try:
                        candidates.append((datetime.strptime(match.group(1), "%Y%m%d").date(), path))
                    except ValueError:
                        pass
                elif not os.path.islink(path) and os.path.isdir(path):
                    date_value = self._dated_name(name)
                    candidates.append(
                        (date_value, os.path.join(path, f"{LOG_FILE_PREFIX}.log"))
                    )
            for file_date, candidate in candidates:
                if file_date is None or file_date >= cutoff:
                    continue
                if (
                    os.path.islink(candidate)
                    or not os.path.isfile(candidate)
                    or os.path.abspath(candidate) in open_paths
                ):
                    continue
                if self._compress_log(candidate):
                    compressed += 1
                else:
                    failed += 1
                    self._log("WARNING", f"Log compression failed path={candidate}")
        return compressed, failed

    def _remove_expired_logs(self, now):
        cutoff = datetime.fromtimestamp(now - LOG_RETENTION_DAYS * 86400).date()
        deleted = directories_deleted = 0
        if not os.path.isdir(LOG_DIR):
            return deleted, directories_deleted
        open_paths = self._open_file_paths()
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            file_date = None
            match = re.fullmatch(
                rf"{re.escape(LOG_FILE_PREFIX)}_(\d{{4}}-\d{{2}}-\d{{2}})\.log(?:\.gz)?", name,
            )
            if match:
                file_date = self._dated_name(match.group(1))
            else:
                match = re.fullmatch(r"resource-watchdog\.(\d{8})_\d{6}\.jsonl(?:\.gz)?", name)
                if match:
                    try:
                        file_date = datetime.strptime(match.group(1), "%Y%m%d").date()
                    except ValueError:
                        file_date = None
            if file_date is not None:
                if (
                    file_date <= cutoff
                    and os.path.isfile(path)
                    and not os.path.islink(path)
                    and os.path.abspath(path) not in open_paths
                ):
                    try:
                        os.remove(path)
                        deleted += 1
                    except OSError:
                        pass
                continue
            if os.path.islink(path) or not os.path.isdir(path):
                continue
            date_value = self._dated_name(name)
            if date_value is None or date_value > cutoff:
                continue
            log_paths = (
                os.path.join(path, f"{LOG_FILE_PREFIX}.log"),
                os.path.join(path, f"{LOG_FILE_PREFIX}.log.gz"),
            )
            try:
                for log_path in log_paths:
                    if (
                        os.path.isfile(log_path)
                        and not os.path.islink(log_path)
                        and os.path.abspath(log_path) not in open_paths
                    ):
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
        logs_compressed, log_compression_failed = self._compress_completed_logs(now)
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
            f"Storage maintenance complete: logs_compressed={logs_compressed} "
            f"log_compression_failed={log_compression_failed} logs_deleted={log_deleted} "
            f"log_directories_deleted={log_directories_deleted} "
            f"scale_databases_deleted={scale_databases_deleted} "
            f"mqtt_dead_letters_deleted={mqtt_deleted} image_dead_letters_deleted={image_deleted}",
        )
        return {
            "logs_compressed": logs_compressed,
            "log_compression_failed": log_compression_failed,
            "logs_deleted": log_deleted,
            "log_directories_deleted": log_directories_deleted,
            "scale_databases_deleted": scale_databases_deleted,
            "mqtt_deleted": mqtt_deleted,
            "image_deleted": image_deleted,
        }
