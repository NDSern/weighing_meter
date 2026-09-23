"""Mock compensation integrations for duplicate session review.

These helpers NEVER touch the real MQTT broker or MinIO. They only write
inspectable intent records to a local directory so a future real
compensation integration (with a broker/frontend contract and
acknowledgement semantics) can be built and validated.
"""

import json
import os
from datetime import datetime, timezone

from config import MINIO_BUCKET, MQTT_TOPIC, WEIGHBRIDGE_ID

MQTT_MOCK_FILE = "mqtt_compensation.jsonl"
MINIO_MOCK_FILE = "minio_removal.jsonl"
REASON_CONTINUOUS_LOAD = "duplicate_session_continuous_load"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _append(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        fp.flush()
        os.fsync(fp.fileno())


def build_mqtt_compensation_intent(losing, surviving, reason=REASON_CONTINUOUS_LOAD):
    """Structured intent to retract a previously published duplicate event."""
    return {
        "intent": "mqtt_compensation",
        "provisional": True,
        "reason": reason,
        "created_at": _now(),
        "topic": MQTT_TOPIC,
        "topic_status": "provisional_unconfirmed",
        "payload": {
            "op": "retract_weighbridge_event",
            "weighbridge_id": WEIGHBRIDGE_ID,
            "session_id": losing.get("session_id"),
            "published_event_id": losing.get("outbox_event_id") or losing.get("session_id"),
            "plate": losing.get("plate"),
            "plate_status": losing.get("plate_status"),
            "stable_weight_kg": losing.get("stable_weight_kg"),
            "surviving_session_id": surviving.get("session_id"),
            "surviving_published_event_id": surviving.get("outbox_event_id"),
        },
        "note": "Mock only. A broker cannot unsend; real action requires a "
                "frontend/backend compensation protocol with acknowledgement.",
    }


def build_minio_removal_intent(losing, reason=REASON_CONTINUOUS_LOAD):
    """Intent to remove exactly the stored objects of the losing session."""
    keys = losing.get("image_object_keys") or []
    if not isinstance(keys, list):
        keys = list(keys)
    return {
        "intent": "minio_object_removal",
        "provisional": True,
        "reason": reason,
        "created_at": _now(),
        "bucket": MINIO_BUCKET,
        "session_id": losing.get("session_id"),
        "objects": [str(key) for key in keys],
        "note": "Mock only. Exact recorded object keys; no prefix deletes.",
    }


def emit_mock_actions(mock_dir, mqtt_intent, minio_intent):
    """Append the two intents; return their local file paths."""
    mqtt_path = os.path.join(mock_dir, MQTT_MOCK_FILE)
    minio_path = os.path.join(mock_dir, MINIO_MOCK_FILE)
    _append(mqtt_path, mqtt_intent)
    _append(minio_path, minio_intent)
    return mqtt_path, minio_path