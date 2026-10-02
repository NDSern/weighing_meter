"""
Weighing Service — DETECT-FIRST architecture.

- Camera detection runs CONTINUOUSLY (not gated by scale)
- PlateTracker accumulates weighted votes across frames
- When scale stabilizes → pairs confirmed plate with stable weight
- When scale returns to zero → clears plate tracker for next vehicle
- Uses test3 YOLOv8-OBB RKNN detector + PP-OCR RKNN recognizer

Resource construction and teardown live in ``services.runtime.bootstrap``.
"""

import os
import signal
import sys
import threading

os.environ.setdefault("MALLOC_ARENA_MAX", "4")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "2")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ────────────────────────────────────────────────────
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
LPR_DIR = os.path.join(SERVICE_DIR, "yolov5lpr")
sys.path.insert(0, SERVICE_DIR)
sys.path.insert(0, LPR_DIR)

from config import LOG_DIR, LOG_FILE_PREFIX
from services.runtime.async_logging import AsyncLogger
from services.runtime.bootstrap import (
    ServiceResources,
    configure_module_logging,
    construct_service,
    shutdown_service,
)

_logger = AsyncLogger(LOG_DIR, LOG_FILE_PREFIX)
log = _logger.log
close_log = _logger.close

configure_module_logging(log)


# ── Main Service ──────────────────────────────────────────────────
def main():
    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):
        stop_event.set()

    previous_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    resources = ServiceResources()

    try:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        construct_service(log, resources)
        while not stop_event.wait(0.1):
            if resources.reader.state == "failed":
                raise RuntimeError(f"Scale reader failed: {resources.reader.last_error}")
            if resources.session_manager.fatal_error:
                raise RuntimeError(resources.session_manager.fatal_error)
    except KeyboardInterrupt:
        request_stop()
    finally:
        shutdown_service(resources, log)
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception as exc:
                log("ERROR", f"Cleanup failed resource=signal_{sig}: {exc}")
        try:
            log("INFO", "Weighing Service stopped.")
        except Exception as exc:
            log("ERROR", f"Cleanup failed resource=final_log: {exc}")
        try:
            close_log()
        except Exception:
            pass


if __name__ == "__main__":
    main()