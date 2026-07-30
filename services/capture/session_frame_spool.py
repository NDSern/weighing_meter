"""Bounded, durable JPEG capture for active weighing sessions."""

import json
import os
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2


class SessionFrameSpool:
    """Periodically persist latest camera frames and queue finalized sessions."""

    def __init__(
        self,
        root_dir,
        cam1_grabber,
        cam3_grabber,
        cam2_grabber=None,
        interval=0.2,
        jpeg_quality=90,
        notification_queue_size=32,
        disk_cap_bytes=None,
        min_free_bytes=64 * 1024 * 1024,
        cv2_module=cv2,
        metadata_provider=None,
        quarantine_retention_days=30,
    ):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.root_dir = os.path.abspath(root_dir)
        self.sessions_dir = os.path.join(self.root_dir, "sessions")
        self.active_dir = os.path.join(self.root_dir, "active")
        self.jobs_dir = os.path.join(self.root_dir, "pending")
        self.processing_dir = os.path.join(self.root_dir, "processing")
        self.cleanup_dir = os.path.join(self.root_dir, "cleanup")
        self.failed_dir = os.path.join(self.root_dir, "failed")
        self.orphan_dir = os.path.join(self.root_dir, "orphan")
        self._grabbers = {"cam1": cam1_grabber, "cam3": cam3_grabber}
        if cam2_grabber is not None:
            self._grabbers["cam2"] = cam2_grabber
        self._interval = interval
        self._jpeg_quality = jpeg_quality
        self._disk_cap_bytes = disk_cap_bytes
        self._min_free_bytes = min_free_bytes
        self._cv2 = cv2_module
        self._metadata_provider = metadata_provider
        self._quarantine_retention_seconds = quarantine_retention_days * 86400
        self._notifications = queue.Queue(maxsize=notification_queue_size)
        self._notification_lock = threading.Lock()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._active = None
        self._sequence = 0
        self._known_notifications = set()
        self._bytes_written = 0

        for path in (
            self.sessions_dir, self.active_dir, self.jobs_dir, self.processing_dir,
            self.cleanup_dir, self.failed_dir, self.orphan_dir,
        ):
            os.makedirs(path, exist_ok=True)
        self._recover_disk_state()
        self._bytes_written = self._spool_size()
        self._recover_notifications()

    def start(self):
        """Start sampler thread. Safe to call more than once."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop, name="session-frame-spool", daemon=True
            )
            self._thread.start()

    def stop(self, timeout=3.0):
        """Request sampler shutdown and report whether it stopped in time."""
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout)
        return not thread or not thread.is_alive()

    def begin_session(self, session_id, start_frames=None, metadata=None):
        """Begin one session, optionally persisting already-captured camera frames."""
        session_id = str(session_id)
        if not session_id or session_id in (".", "..") or os.path.basename(session_id) != session_id:
            raise ValueError("session_id must be one path-safe segment")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("a session is already active")
            session_dir = os.path.join(self.sessions_dir, session_id)
            if os.path.exists(session_dir):
                raise FileExistsError(session_dir)
            os.makedirs(session_dir)
            self._active = {
                "session_id": session_id,
                "session_dir": session_dir,
                "started_at": self._now(),
                "files": [],
                "frame_metadata": {},
                "incomplete": False,
                "errors": [],
                "counts": {camera: 0 for camera in self._grabbers},
                "metadata": dict(metadata or {}),
            }
            self._write_active_locked()
            for camera, frame in (start_frames or {}).items():
                if camera in self._grabbers and frame is not None:
                    self._save_frame_locked(camera, frame, "start")
        return session_dir

    def update_active_metadata(self, session_id, metadata):
        """Durably replace metadata attached to the active session."""
        with self._lock:
            if self._active is None or self._active["session_id"] != str(session_id):
                raise ValueError("session is not active")
            self._active["metadata"] = metadata
            self._write_active_locked()

    def end_session(self, session_id, metadata):
        """Finalize active session atomically and return durable job manifest path."""
        with self._lock:
            if self._active is None or self._active["session_id"] != str(session_id):
                raise ValueError("session is not active")
            active = self._active
            self._sequence += 1
            job_name = "%020d-%08d-%s.json" % (
                time.time_ns(), self._sequence, uuid.uuid4().hex
            )
            manifest_path = os.path.join(self.jobs_dir, job_name)
            manifest = {
                "session_id": active["session_id"],
                "session_dir": active["session_dir"],
                "started_at": active["started_at"],
                "ended_at": self._now(),
                "files": list(active["files"]),
                "frame_metadata": dict(active["frame_metadata"]),
                "frame_counts": dict(active["counts"]),
                "capture_interval_seconds": self._interval,
                "metadata": metadata,
                "incomplete": active["incomplete"],
                "errors": list(active["errors"]),
            }
            self._atomic_json(manifest_path, manifest)
            self._unlink_durable(self._active_path(active["session_id"]))
            self._active = None
        self._notify(manifest_path)
        return manifest_path

    def get_pending_job(self, timeout=None):
        """Atomically claim and return the oldest pending manifest path."""
        if self._notifications.empty():
            self._recover_notifications()
        try:
            path = self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._notification_lock:
            self._known_notifications.discard(path)
        if os.path.exists(path):
            claimed = os.path.join(self.processing_dir, os.path.basename(path))
            try:
                os.replace(path, claimed)
                self._fsync_dir(self.jobs_dir)
                self._fsync_dir(self.processing_dir)
                return claimed
            except FileNotFoundError:
                pass
        return self.get_pending_job(timeout=0)

    def acknowledge_job(self, manifest_path):
        """Remove a processed job marker and its captured session files."""
        path = self._job_path(manifest_path, (self.processing_dir, self.jobs_dir))
        cleanup_path = os.path.join(self.cleanup_dir, os.path.basename(path))
        os.replace(path, cleanup_path)
        self._fsync_dir(os.path.dirname(path))
        self._fsync_dir(self.cleanup_dir)
        session_dir = None
        try:
            with open(cleanup_path, encoding="utf-8") as handle:
                session_dir = json.load(handle).get("session_dir")
        except (OSError, ValueError, TypeError):
            self._move_failed(cleanup_path)
            return
        if session_dir:
            session_dir = os.path.abspath(session_dir)
            if self._is_session_dir(session_dir):
                try:
                    size = self._directory_size(session_dir)
                    shutil.rmtree(session_dir)
                    self._bytes_written = max(0, self._bytes_written - size)
                except OSError:
                    return
        self._unlink_durable(cleanup_path)
        with self._notification_lock:
            self._known_notifications.discard(path)

    def abort_session(self, session_id):
        """Discard one partially initialized active session."""
        with self._lock:
            if self._active is None or self._active["session_id"] != str(session_id):
                return False
            session_dir = self._active["session_dir"]
            active_path = self._active_path(str(session_id))
            self._active = None
            size = self._directory_size(session_dir)
            shutil.rmtree(session_dir, ignore_errors=True)
            self._unlink_durable(active_path)
            self._bytes_written = max(0, self._bytes_written - size)
            return True

    def fail_job(self, manifest_path):
        """Move a permanently failed manifest out of FIFO while preserving frames."""
        path = self._job_path(manifest_path, (self.processing_dir, self.jobs_dir))
        target = os.path.join(self.failed_dir, os.path.basename(path))
        os.replace(path, target)
        self._fsync_dir(os.path.dirname(path))
        self._fsync_dir(self.failed_dir)
        with self._notification_lock:
            self._known_notifications.discard(path)
        return target

    def save_session_frame(self, session_id, name, frame):
        """Persist an auxiliary frame inside the active session directory."""
        with self._lock:
            if self._active is None or self._active["session_id"] != str(session_id):
                raise ValueError("session is not active")
            if os.path.basename(name) != name:
                raise ValueError("name must be one path-safe segment")
            path = os.path.join(self._active["session_dir"], name)
            ok, encoded = self._cv2.imencode(
                ".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                raise OSError("JPEG encoding failed")
            data = encoded.tobytes()
            reason = self._space_failure(len(data))
            if reason:
                self._active["incomplete"] = True
                if reason not in self._active["errors"]:
                    self._active["errors"].append(reason)
                return None
            previous_size = os.path.getsize(path) if os.path.exists(path) else 0
            self._atomic_bytes(path, data)
            self._bytes_written += len(data) - previous_size
            relative_path = os.path.relpath(path, self._active["session_dir"])
            if relative_path not in self._active["files"]:
                self._active["files"].append(relative_path)
            self._write_active_locked()
            return path

    def _capture_loop(self):
        next_capture = time.monotonic()
        next_cleanup = next_capture
        while not self._stop_event.is_set():
            delay = max(0.0, next_capture - time.monotonic())
            if self._stop_event.wait(delay):
                break
            with self._lock:
                if self._active is not None:
                    for camera, grabber in self._grabbers.items():
                        frame, frame_id = self._read_frame(grabber)
                        if frame is not None:
                            self._save_frame_locked(camera, frame, "sample", frame_id)
            if time.monotonic() >= next_cleanup:
                self._resume_cleanup()
                next_cleanup = time.monotonic() + 10.0
            next_capture = max(next_capture + self._interval, time.monotonic())

    def _resume_cleanup(self):
        for path in self._manifest_paths(self.cleanup_dir):
            manifest = self._read_manifest(path)
            if manifest is None:
                self._move_failed(path)
                continue
            session_dir = os.path.abspath(str(manifest.get("session_dir", "")))
            if not self._is_session_dir(session_dir):
                self._move_failed(path)
                continue
            try:
                size = self._directory_size(session_dir)
                shutil.rmtree(session_dir)
            except FileNotFoundError:
                size = 0
            except OSError:
                continue
            self._bytes_written = max(0, self._bytes_written - size)
            self._unlink_durable(path)
        self._expire_quarantine()

    def _expire_quarantine(self):
        cutoff = time.time() - self._quarantine_retention_seconds
        live_references = self._manifest_session_references(
            (self.jobs_dir, self.processing_dir, self.cleanup_dir)
        )
        failed_references = self._manifest_session_reference_counts((self.failed_dir,))
        for directory in (self.failed_dir, self.orphan_dir):
            for path in self._expired_paths(directory, cutoff):
                try:
                    session_dir = self._failed_session_dir(path) if directory == self.failed_dir else None
                    removed_bytes = self._path_size(path)
                    if (session_dir and session_dir not in live_references
                            and failed_references.get(session_dir, 0) <= 1
                            and os.path.isdir(session_dir)):
                        removed_bytes += self._directory_size(session_dir)
                        shutil.rmtree(session_dir)
                    self._remove_path(path)
                    self._bytes_written = max(0, self._bytes_written - removed_bytes)
                except OSError:
                    continue

    @staticmethod
    def _expired_paths(directory, cutoff):
        try:
            names = os.listdir(directory)
        except OSError:
            return []
        expired = []
        for name in names:
            path = os.path.join(directory, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    expired.append(path)
            except OSError:
                continue
        return expired

    def _failed_session_dir(self, manifest_path):
        if not os.path.isfile(manifest_path):
            return None
        manifest = self._read_manifest(manifest_path)
        if not manifest:
            return None
        session_dir = os.path.abspath(str(manifest.get("session_dir", "")))
        return session_dir if self._is_session_dir(session_dir) else None

    def _path_size(self, path):
        return self._directory_size(path) if os.path.isdir(path) else os.path.getsize(path)

    @staticmethod
    def _remove_path(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)

    def _manifest_session_references(self, directories):
        return set(self._manifest_session_reference_counts(directories))

    def _manifest_session_reference_counts(self, directories):
        counts = {}
        for directory in directories:
            for path in self._manifest_paths(directory):
                manifest = self._read_manifest(path)
                if not manifest:
                    continue
                session_dir = os.path.abspath(str(manifest.get("session_dir", "")))
                if self._is_session_dir(session_dir):
                    counts[session_dir] = counts.get(session_dir, 0) + 1
        return counts

    @staticmethod
    def _read_frame(grabber):
        if grabber is None:
            return None, None
        peek_with_id = getattr(grabber, "peek_latest_frame_with_id", None)
        if peek_with_id:
            return peek_with_id(copy_frame=True)
        peek = getattr(grabber, "peek_latest_frame", None)
        if peek:
            return peek(copy_frame=True), None
        if callable(grabber):
            return grabber(), None
        raise TypeError("grabber must be callable or expose peek_latest_frame")

    def _save_frame_locked(self, camera, frame, kind, frame_id=None):
        active = self._active
        index = active["counts"][camera]
        name = "%s-%06d-%s.jpg" % (camera, index, kind)
        path = os.path.join(active["session_dir"], name)
        try:
            ok, encoded = self._cv2.imencode(
                ".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                raise OSError("JPEG encoding failed")
            data = encoded.tobytes()
            reason = self._space_failure(len(data))
            if reason:
                active["incomplete"] = True
                if reason not in active["errors"]:
                    active["errors"].append(reason)
                return False
            self._atomic_bytes(path, data)
            self._bytes_written += len(data)
            active["counts"][camera] += 1
            active["files"].append(os.path.relpath(path, active["session_dir"]))
            relative_path = os.path.relpath(path, active["session_dir"])
            active["frame_metadata"][relative_path] = self._frame_metadata(camera, frame_id)
            return True
        except Exception as exc:
            active["incomplete"] = True
            active["errors"].append("%s: %s" % (camera, exc))
            return False

    def _frame_metadata(self, camera, frame_id=None):
        captured_at = self._now()
        tracks = []
        if self._metadata_provider:
            tracks = self._metadata_provider(camera, frame_id)
        return {"captured_at": captured_at, "frame_id": frame_id, "tracks": tracks}

    def _space_failure(self, size):
        if self._disk_cap_bytes is not None and self._bytes_written + size > self._disk_cap_bytes:
            return "disk cap reached"
        try:
            if shutil.disk_usage(self.root_dir).free - size < self._min_free_bytes:
                return "minimum free space reached"
        except OSError as exc:
            return "free-space check failed: %s" % exc
        return None

    def _recover_notifications(self):
        try:
            paths = sorted(
                os.path.join(self.jobs_dir, name)
                for name in os.listdir(self.jobs_dir)
                if name.endswith(".json")
            )
        except OSError:
            return
        for path in paths:
            self._notify(path)

    def _recover_disk_state(self):
        # Claims are leases: restart makes every interrupted inference retryable.
        for path in self._manifest_paths(self.processing_dir):
            target = os.path.join(self.jobs_dir, os.path.basename(path))
            if os.path.exists(target):
                self._move_failed(path)
            else:
                os.replace(path, target)
                self._fsync_dir(self.processing_dir)
                self._fsync_dir(self.jobs_dir)

        # Cleanup means inference already succeeded. Never put these jobs back.
        self._resume_cleanup()

        referenced = set()
        for directory in (self.jobs_dir, self.failed_dir):
            for path in self._manifest_paths(directory):
                manifest = self._read_manifest(path)
                if manifest is None:
                    if directory != self.failed_dir:
                        self._move_failed(path)
                    continue
                session_dir = os.path.abspath(str(manifest.get("session_dir", "")))
                if not self._is_session_dir(session_dir):
                    if directory != self.failed_dir:
                        self._move_failed(path)
                    continue
                referenced.add(session_dir)

        for path in self._manifest_paths(self.active_dir):
            active = self._read_manifest(path)
            if active is None:
                self._move_failed(path)
                continue
            session_dir = os.path.abspath(str(active.get("session_dir", "")))
            session_id = str(active.get("session_id", ""))
            if not self._is_session_dir(session_dir) or session_dir != os.path.join(self.sessions_dir, session_id):
                self._move_failed(path)
                continue
            if session_dir in referenced:
                self._unlink_durable(path)
                continue
            files = self._session_files(session_dir)
            if files:
                metadata = active.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                ended_at = self._now()
                started_at = metadata.get("started_at") or active.get("started_at")
                try:
                    duration_s = max(
                        0.0,
                        datetime.fromisoformat(ended_at).timestamp()
                        - datetime.fromisoformat(started_at).timestamp(),
                    )
                except (TypeError, ValueError):
                    duration_s = 0.0
                metadata.setdefault("end_reason", "machine_offline")
                metadata.setdefault("ended_at", ended_at)
                metadata.setdefault("duration_s", duration_s)
                metadata["recovered_after_restart"] = True
                metadata["incomplete"] = True
                manifest = {
                    "session_id": session_id,
                    "session_dir": session_dir,
                    "started_at": active.get("started_at"),
                    "ended_at": ended_at,
                    "files": files,
                    "frame_metadata": dict(active.get("frame_metadata", {})),
                    "frame_counts": active.get("counts", {"cam1": 0, "cam3": 0}),
                    "capture_interval_seconds": active.get("capture_interval_seconds", self._interval),
                    "metadata": metadata,
                    "incomplete": True,
                    "errors": list(active.get("errors", [])) + ["recovered after restart"],
                }
                target = os.path.join(self.jobs_dir, self._job_name())
                self._atomic_json(target, manifest)
                self._unlink_durable(path)
                referenced.add(session_dir)
            else:
                target = os.path.join(self.orphan_dir, session_id)
                if os.path.exists(target):
                    target += "-" + uuid.uuid4().hex
                if os.path.isdir(session_dir):
                    os.replace(session_dir, target)
                self._move_failed(path)

        for name in os.listdir(self.sessions_dir):
            path = os.path.abspath(os.path.join(self.sessions_dir, name))
            if self._is_session_dir(path) and os.path.isdir(path) and path not in referenced:
                target = os.path.join(self.orphan_dir, name)
                if os.path.exists(target):
                    target += "-" + uuid.uuid4().hex
                os.replace(path, target)

        for directory in (self.jobs_dir, self.processing_dir, self.cleanup_dir,
                          self.active_dir, self.failed_dir, self.orphan_dir, self.sessions_dir):
            self._fsync_dir(directory)

    def _write_active_locked(self):
        active = self._active
        manifest = {
            "session_id": active["session_id"],
            "session_dir": active["session_dir"],
            "started_at": active["started_at"],
            "files": list(active["files"]),
            "frame_metadata": dict(active["frame_metadata"]),
            "counts": dict(active["counts"]),
            "capture_interval_seconds": self._interval,
            "metadata": active["metadata"],
            "incomplete": active["incomplete"],
            "errors": list(active["errors"]),
        }
        self._atomic_json(self._active_path(active["session_id"]), manifest)

    def _active_path(self, session_id):
        return os.path.join(self.active_dir, session_id + ".json")

    def _job_name(self):
        self._sequence += 1
        return "%020d-%08d-%s.json" % (time.time_ns(), self._sequence, uuid.uuid4().hex)

    @staticmethod
    def _manifest_paths(directory):
        try:
            return sorted(os.path.join(directory, name) for name in os.listdir(directory)
                          if name.endswith(".json"))
        except OSError:
            return []

    @staticmethod
    def _read_manifest(path):
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def _move_failed(self, path):
        target = os.path.join(self.failed_dir, os.path.basename(path))
        if os.path.exists(target):
            target += "-" + uuid.uuid4().hex + ".json"
        os.replace(path, target)
        return target

    def _job_path(self, path, allowed_dirs):
        path = os.path.abspath(os.fspath(path))
        if os.path.dirname(path) not in allowed_dirs or not os.path.basename(path).endswith(".json"):
            raise ValueError("manifest is outside allowed job directory")
        return path

    def _is_session_dir(self, path):
        return os.path.dirname(path) == self.sessions_dir and os.path.basename(path) not in ("", ".", "..")

    @staticmethod
    def _session_files(session_dir):
        try:
            return sorted(name for name in os.listdir(session_dir)
                          if os.path.isfile(os.path.join(session_dir, name)))
        except OSError:
            return []

    def _notify(self, path):
        with self._notification_lock:
            if path in self._known_notifications:
                return
            try:
                self._notifications.put_nowait(path)
                self._known_notifications.add(path)
            except queue.Full:
                pass

    def _spool_size(self):
        return sum(
            self._directory_size(path)
            for path in (self.sessions_dir, self.failed_dir, self.orphan_dir)
        )

    @staticmethod
    def _atomic_bytes(path, data):
        temp = path + ".tmp-" + uuid.uuid4().hex
        try:
            with open(temp, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            SessionFrameSpool._fsync_dir(os.path.dirname(path))
        finally:
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass

    @staticmethod
    def _unlink_durable(path):
        try:
            os.unlink(path)
            SessionFrameSpool._fsync_dir(os.path.dirname(path))
        except FileNotFoundError:
            pass

    @staticmethod
    def _fsync_dir(path):
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    @classmethod
    def _atomic_json(cls, path, value):
        cls._atomic_bytes(
            path,
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )

    @staticmethod
    def _directory_size(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
