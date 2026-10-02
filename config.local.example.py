"""Host-local configuration overrides.

Copy this file to ``config.local.py`` on the bridge host and fill in the
values that differ from the tracked defaults. ``config.local.py`` is never
committed and holds host identity, camera overrides, and — when not supplied
by the environment — credentials.

Resolution order (later wins):

1. Tracked code defaults in ``config.py`` (no credentials)
2. This ``config.local.py``
3. Environment values, loaded by systemd from
   ``/etc/weighing-meter/weighing.env``

Secrets should normally be supplied by the environment file so they stay out
of every repository and backup. Camera addresses and the weighbridge identity
are not secret and can live here.
"""

# --- Identity -----------------------------------------------------------
# Also selects the per-host policy block in config.py.
# WEIGHBRIDGE_ID = "100ecc11-dbcb-4c23-8e89-d41ccefcda37"

# --- Cameras ------------------------------------------------------------
# Share one credential pair, or set full URLs explicitly.
# RTSP_USERNAME = "camera-user"
# RTSP_PASSWORD = "camera-password"
# RTSP_HOST_CAM1 = "192.168.1.181"
# RTSP_HOST_CAM2 = "192.168.1.179"
# RTSP_HOST_CAM3 = "192.168.1.177"
# RTSP_URL = "rtsp://camera-user:camera-password@192.168.1.182:554/live/0/MAIN"
# RTSP_URL_2 = "rtsp://camera-user:camera-password@192.168.1.179:554/ch01/0"
# RTSP_URL_3 = "rtsp://camera-user:camera-password@192.168.1.182:554/live/0/MAIN"

# --- Credentials (prefer the environment file instead) ------------------
# MQTT_USERNAME = ""
# MQTT_PASSWORD = ""
# MINIO_ACCESS_KEY = ""
# MINIO_SECRET_KEY = ""

# --- Per-host behavior --------------------------------------------------
# OPTIONAL. Duplicate-session review is mock-only and off by default.
# DUPLICATE_REVIEW_ENABLED = True