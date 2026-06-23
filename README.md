# Weighing Meter

Production weighbridge service for reading scale data, recognizing truck license plates from RTSP cameras, saving evidence images, and publishing confirmed weighing events over MQTT.

## Runtime Flow

```text
RTSP cam1/cam3 -> CameraGrabber -> DetectCoordinator
              -> YOLOv8-OBB plate detector -> PP-OCR recognizer
              -> PlateTracker

D2008 scale -> D2008Reader -> SessionManager
            -> session start/end -> confirmed plate + stable weight

Publish -> local image save -> MinIO upload retry queue
        -> confirmed plate DB -> MQTT publish outbox
```

## Main Entrypoint

```bash
python3 weighing_service.py
```

Systemd service on production devices:

```bash
sudo systemctl restart weighing_service.service
systemctl is-active weighing_service.service
journalctl -u weighing_service.service -n 100 --no-pager
```

## Key Files

```text
weighing_service.py                         service entrypoint and wiring
config.py                                   shared defaults and config.local.py loader
config.local.py                             per-device overrides, untracked
mqtt_service.py                             MQTT publisher
d2008_scale_reader.py                       D2008 serial scale reader
registered_license_plates.json              active plate registry
services/capture/frame_source.py            RTSP latest-frame grabbers
services/capture/detect_coordinator.py      LPR and vehicle detection coordinators
services/pipeline/license_plate_recognition.py  production LPR pipeline
services/session/session_manager.py         session lifecycle, publishing, DB counts
services/storage/image_save_worker.py       local image save and MinIO retry queue
services/storage/publish_outbox.py          durable MQTT outbox
services/tracking/plate_tracker.py          plate aggregation and image selection
services/tracking/vehicle_tracker.py        vehicle stability/left detection
```

## Local Configuration

Production devices keep host-specific settings in untracked `config.local.py`.

Common overrides:

```python
RTSP_URL = "rtsp://user:pass@camera/front"
RTSP_URL_2 = "rtsp://user:pass@camera/rear"
RTSP_URL_3 = "rtsp://user:pass@camera/side"
CAM2_RESULT_CROP = "left"  # left, right, or full
WEIGHBRIDGE_ID = "..."
MQTT_WEIGHBRIDGE_TOPIC_ID = WEIGHBRIDGE_ID[:8]
```

Do not commit `config.local.py` or `weighing_service.service`.

## Runtime Data

Runtime data is intentionally untracked:

```text
/storage/
/logs/
/captures/
*.db
*.db-shm
*.db-wal
```

Important runtime files:

```text
scale_data.db                         scale readings
confirmed_license_plates.db           confirmed plate counts
storage/upload_pending.jsonl          MinIO upload retry queue
storage/publish_pending.jsonl         MQTT publish retry queue
storage/weighbridge/YYYY/MM/DD/       evidence images
storage/undetectable/                 unknown plate evidence
```

## Plate Confirmation

Plate observations are accumulated during active weighing sessions. A plate is confirmed when tracker count reaches `PLATE_CONFIRM_THRESHOLD`.
Confirmation also requires the plate to be selected as the main OCR result at least `MIN_SELECTED_PLATE_HITS` times and observed across at least `MIN_PLATE_OBSERVATION_SPAN_SECONDS`. Alternate OCR candidates can support a selected plate, but cannot confirm a plate by themselves.

Post-processing order:

```text
OCR candidate formatting
same-session detailed variant preference
registered plate correction
duplicate session skip
local image save
confirmed plate DB increment
MQTT outbox enqueue
```

Registered correction uses `registered_license_plates.json`:

```text
exact match first
unique family match second
unique edit-distance <= 1 match third
ambiguous matches keep OCR result
```

9-character plates are supported, for example:

```text
29R2-123.45
15G1-659.23
```

## Publishing Guarantees

Evidence images are saved locally before MQTT outbox enqueue. MinIO upload failures stay in `storage/upload_pending.jsonl` and retry in background. MQTT events stay in `storage/publish_pending.jsonl` until required local images exist and MQTT publish receives acknowledgement. MQTT publish does not wait for MinIO upload completion, so event rows can arrive before photo URLs become available in MinIO.

Local images with pending MinIO uploads are protected from retention cleanup until their upload succeeds.

Unchosen LPR camera images are saved locally only. They are not uploaded to MinIO and are not included in MQTT payloads.

## Deployment

Local Git authority:

```text
/home/son/Projects/weighing_meter
```

Production paths:

```text
cang-hp1:/home/aibox-vnpay2/apps/weighing_meter
cang-hp2:/home/vta-giavu-weightbridge2/apps/weighing_meter
```

Devices use HTTPS Git remote:

```bash
git remote -v
```

Expected remote:

```text
https://github.com/NDSern/weighing_meter.git
```

Deploy latest on a device:

```bash
cd /home/<device-user>/apps/weighing_meter
git pull --ff-only
python3 -m py_compile weighing_service.py services/capture/frame_source.py services/session/session_manager.py services/storage/image_save_worker.py services/tracking/plate_tracker.py
sudo systemctl restart weighing_service.service
systemctl is-active weighing_service.service
```

## Verification

Local syntax check:

```bash
python3 -m py_compile weighing_service.py services/capture/frame_source.py services/session/session_manager.py services/storage/image_save_worker.py services/tracking/plate_tracker.py
```

Production health checks:

```bash
systemctl is-active weighing_service.service
journalctl -u weighing_service.service -n 100 --no-pager
git rev-parse --short HEAD
```

Queue checks:

```bash
wc -l storage/upload_pending.jsonl storage/publish_pending.jsonl
```

Expected normal state after network recovery:

```text
upload_pending.jsonl: 0
publish_pending.jsonl: 0
```

## Known Operational Notes

RTSP/HEVC decoder warnings can appear in logs and are usually camera stream noise:

```text
log2_parallel_merge_level_minus2 out of range: -1
PPS id out of range: 0
```

If MinIO is full, uploads fail and retry until backend storage is available:

```text
XMinioStorageFull: Storage backend has reached its minimum free drive threshold
```

If `ModuleNotFoundError: No module named 'services.storage'` appears, check that `/storage/` is ignored but `services/storage/` exists in Git.
