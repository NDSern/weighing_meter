#!/usr/bin/env python3
"""Archive aged diagnostics without sharing the weighing service cgroup."""

import os
import sys

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_DIR)

from config import IMAGE_RETENTION_CHECK_INTERVAL_SECONDS, LPR_SPOOL_DIR, NO_PLATE_DIR, NO_STABLE_DIR
from services.storage.retention_cleaner import DiagnosticArchiveCleaner


def spool_is_busy():
    return any(
        os.path.isdir(os.path.join(LPR_SPOOL_DIR, state))
        and os.listdir(os.path.join(LPR_SPOOL_DIR, state))
        for state in ("active", "processing")
    )


def log(level, message):
    print(f"[{level}] {message}", flush=True)


def main():
    if spool_is_busy():
        log("INFO", "Diagnostic archive skipped: LPR spool is active")
        return 0
    result = DiagnosticArchiveCleaner(
        [NO_STABLE_DIR, NO_PLATE_DIR],
        3,
        30,
        IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
        log_fn=log,
    ).run_once()
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
