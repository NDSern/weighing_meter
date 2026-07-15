"""Durable publish outbox for offline MQTT delivery."""

import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime

from config import PENDING_RETENTION_DAYS, SERVICE_DIR
from services.storage.dead_letter import append_dead_letter, is_expired

_log_fn = None


def set_log_fn(log_fn):
    global _log_fn
    _log_fn = log_fn


def log(level: str, msg: str):
    if _log_fn:
        _log_fn(level, msg)


_outbox_file = os.path.join(SERVICE_DIR, "storage", "publish_pending.jsonl")
_pending_lock = threading.Lock()
_pending_events = {}
_publish_queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
_worker_thread = None
_stop_event = threading.Event()
_mqtt_svc = None


class PublishOutbox:
    """Persist finalized sessions until local images exist and MQTT acks."""

    @staticmethod
    def start(mqtt_svc):
        global _worker_started, _worker_thread, _mqtt_svc
        _mqtt_svc = mqtt_svc
        with _worker_lock:
            if _worker_started:
                return
            os.makedirs(os.path.dirname(_outbox_file), exist_ok=True)
            PublishOutbox._load_pending()
            _stop_event.clear()
            _worker_thread = threading.Thread(target=PublishOutbox._publish_loop, daemon=True)
            _worker_thread.start()
            _worker_started = True

    @staticmethod
    def stop(timeout=5.0):
        global _worker_started, _worker_thread, _mqtt_svc
        _stop_event.set()
        if _worker_thread:
            _worker_thread.join(timeout=timeout)
            if _worker_thread.is_alive():
                return False
        with _worker_lock:
            _worker_started = False
            _worker_thread = None
            _mqtt_svc = None
        return True

    @staticmethod
    def enqueue(session_result, image_object_keys=None, image_paths=None):
        event_id = session_result.get("offline_event_id") or uuid.uuid4().hex
        session_result["offline_event_id"] = event_id
        event = {
            "id": event_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_object_keys": list(image_object_keys or []),
            "image_paths": list(image_paths or []),
            "session_result": session_result,
        }
        with _pending_lock:
            if event_id in _pending_events:
                log("OFFLINE", f"Publish event already queued id={event_id}")
                return event_id
            _pending_events[event_id] = event
            PublishOutbox._persist_locked()
        _publish_queue.put(event_id)
        log("OFFLINE", f"Queued publish event id={event_id} plate={session_result.get('official_plate')}")
        return event_id

    @staticmethod
    def pending_count():
        with _pending_lock:
            return len(_pending_events)

    @staticmethod
    def _load_pending():
        if not os.path.exists(_outbox_file):
            return
        with _pending_lock:
            with open(_outbox_file, "r") as fp:
                for line in fp:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        append_dead_letter("malformed", {"raw": line.rstrip()}, "invalid MQTT JSON", "malformed")
                        continue
                    event_id = event.get("id")
                    if not event_id or not event.get("session_result"):
                        append_dead_letter("malformed", event, "incomplete MQTT record", "malformed")
                        continue
                    _pending_events[event_id] = event
            PublishOutbox._persist_locked()
            for event_id in _pending_events:
                _publish_queue.put(event_id)
        if _pending_events:
            log("OFFLINE", f"Loaded {len(_pending_events)} pending publish event(s)")

    @staticmethod
    def _persist_locked():
        tmp_path = _outbox_file + ".tmp"
        os.makedirs(os.path.dirname(_outbox_file), exist_ok=True)
        with open(tmp_path, "w") as fp:
            for event in _pending_events.values():
                fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, _outbox_file)

    @staticmethod
    def _mark_published(event_id):
        with _pending_lock:
            _pending_events.pop(event_id, None)
            PublishOutbox._persist_locked()

    @staticmethod
    def _publish_loop():
        while not _stop_event.is_set():
            try:
                event_id = _publish_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                PublishOutbox._publish_event(event_id)
            except Exception as exc:
                log("ERROR", f"Publish worker error event id={event_id}: {exc}")
                if not _stop_event.wait(10.0):
                    _publish_queue.put(event_id)
            finally:
                _publish_queue.task_done()

    @staticmethod
    def _publish_event(event_id):
        with _pending_lock:
            event = _pending_events.get(event_id)
        if not event:
            return
        if is_expired(event.get("created_at"), PENDING_RETENTION_DAYS):
            append_dead_letter("mqtt", event, "pending event exceeded retry retention", "expired")
            PublishOutbox._mark_published(event_id)
            log("ERROR", f"Moved expired publish event to dead letter id={event_id}")
            return
        missing_paths = [p for p in event.get("image_paths") or [] if not os.path.exists(p)]
        if missing_paths:
            log("ERROR", f"Publishing without missing local images event id={event_id} missing={len(missing_paths)}")
            event["image_paths"] = [p for p in event.get("image_paths") or [] if p not in missing_paths]
        if _mqtt_svc is None:
            if not _stop_event.wait(5.0):
                _publish_queue.put(event_id)
            return
        ok = _mqtt_svc.publish_weighbridge_event(
            event["session_result"],
            wait_for_ack=True,
            timeout=10.0,
        )
        if ok:
            PublishOutbox._mark_published(event_id)
            log("OFFLINE", f"Published queued event id={event_id}")
            log("METRIC", json.dumps(
                {"event": "session_publish_acknowledged", "id": event_id},
                separators=(",", ":"), sort_keys=True,
            ))
            return
        if not _stop_event.wait(10.0):
            _publish_queue.put(event_id)
