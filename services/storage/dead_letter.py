"""Persist terminal queue failures for later inspection or replay."""

import json
import os
from datetime import datetime, timedelta

from config import SERVICE_DIR


DEAD_LETTER_DIR = os.path.join(SERVICE_DIR, "storage", "dead-letter")


def append_dead_letter(kind, record, error, error_class):
    os.makedirs(DEAD_LETTER_DIR, exist_ok=True)
    now = datetime.now()
    path = os.path.join(DEAD_LETTER_DIR, f"{kind}-{now:%Y-%m}.jsonl")
    entry = {
        "failed_at": now.isoformat(timespec="seconds"),
        "last_error": str(error),
        "error_class": error_class,
        "original_record": record,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(path, "a") as fp:
        fp.write(line)
        fp.flush()
        os.fsync(fp.fileno())


def is_expired(created_at, retention_days, now=None):
    if not created_at:
        return False
    try:
        created = datetime.fromisoformat(created_at)
        reference = now or datetime.now()
        # Tolerate mixed aware/naive timestamps instead of letting the
        # subtraction raise (which would keep the entry alive forever).
        if (created.tzinfo is None) != (reference.tzinfo is None):
            created = created.replace(tzinfo=None)
            reference = reference.replace(tzinfo=None)
        return reference - created >= timedelta(days=retention_days)
    except (TypeError, ValueError):
        return False
