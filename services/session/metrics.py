"""Structured metric logging shared by session components."""

import json


def log_metric(log_fn, event, **fields):
    """Emit one JSON metric line through the configured log function."""
    log_fn("METRIC", json.dumps({"event": event, **fields}, separators=(",", ":"), sort_keys=True))
