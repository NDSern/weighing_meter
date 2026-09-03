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
tests/                                      unit tests and manual MQTT probe
services/capture/frame_source.py            RTSP latest-frame grabbers
services/capture/detect_coordinator.py      LPR and vehicle detection coordinators
services/pipeline/license_plate_recognition.py  production LPR pipeline
services/runtime/async_logging.py            nonblocking console/file logger
services/session/session_manager.py         session lifecycle and publishing
services/session/finalization_store.py      finalized-session ledger
services/session/plate_store.py             confirmed-plate persistence
services/session/diagnostic_archive.py      no-stable/no-plate evidence archive
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
```

Known HP-01 and HP-02 identities select their canonical transaction direction and plate-loss policy automatically. Any explicit local policy must match that canonical mapping.

Do not commit `config.local.py` or `weighing_service.service`.

## Tests

Run unit tests from repository root:

```bash
make verify
```

This runs model checksums and the complete unit test suite. Use `make test` for unit tests only.

The manual MQTT probe is intentionally outside test discovery:

```bash
python3 tests/mqtt_probe.py
python3 tests/mqtt_probe.py --publish
```

## LPR Benchmark

Compare explicit RKNN variants on device without changing production model paths:

```bash
python3 benchmark_lpr.py storage/lpr-samples \
  --variant fine-tuned models/lpr/license_plate_detector.rknn models/lpr/license_plate_recognizer.rknn models/lpr/charset.txt 960 \
  --manifest storage/lpr-samples/manifest.json
```

Each repeated `--variant` takes its own detector, recognizer, charset, and image size. Output includes artifact hashes, source dimensions, detector/OCR timing, optional exact-match labels, and per-image plate results. Variants load sequentially on the selected NPU core; use `--core 1` or `--core 2` when needed.

## Fine-Tuned LPR Deployment

The tracked LPR bundle targets RK3588 with RKNNLite 2.3.2. Startup verifies model and decoder hashes, the three-output detector contract, the 38-class OCR contract, finite tensors, and all dynamic OCR widths before MQTT or camera workers start.

Camera 1 can enforce the decoded main-stream resolution through host-local configuration:

```python
RTSP_URL = "rtsp://user:pass@192.168.1.181:554/<verified-main-route>"
CAM1_EXPECTED_RESOLUTION = (2880, 1624)
```

Leave `CAM1_EXPECTED_RESOLUTION = None` until the camera route, exact decoded dimensions, and isolated 2K acceptance checks are complete. See `LPR_2K_DEPLOYMENT.md` for deployment, health checks, and rollback.

## Runtime Data

Runtime data is intentionally untracked:

```text
/storage/
/logs/
/scale_data/
/captures/
*.db
*.db-shm
*.db-wal
```

Important runtime files:

```text
logs/YYYY-MM-DD/weighing_service.log  service logs (60-day retention)
scale_data/YYYY-MM-DD.db              daily scale readings (365-day retention)
scale_data/scale_data.archive.db      legacy root database after migration
confirmed_license_plates.db           confirmed plate counts
storage/upload_pending.jsonl          MinIO upload retry queue
storage/publish_pending.jsonl         MQTT publish retry queue
storage/weighbridge/YYYY/MM/DD/       evidence images
storage/undetectable/                 unknown plate evidence
```

After deploying daily scale storage, stop the service and migrate its legacy root database once:

```bash
sudo systemctl stop weighing_service.service
python3 scripts/migrate_scale_data.py
sudo systemctl start weighing_service.service
```

The script partitions readings by local date, validates each daily database, then moves the verified source to `scale_data/scale_data.archive.db`.

## Plate Confirmation

Plate observations are accumulated during active weighing sessions. A plate is confirmed when tracker count reaches `PLATE_CONFIRM_THRESHOLD`.
Confirmation also requires the plate to be selected as the main OCR result at least `MIN_SELECTED_PLATE_HITS` times and observed across at least `MIN_PLATE_OBSERVATION_SPAN_SECONDS`. Alternate OCR candidates can support a selected plate, but cannot confirm a plate by themselves.

## LPR Model Behavior

The production pipeline currently uses the YOLOv8-OBB RKNN detector and PP-OCR RKNN recognizer:

```text
models/lpr/license_plate_detector.rknn      YOLOv8 oriented-box plate detector
models/lpr/license_plate_recognizer.rknn    PP-OCR CTC plate recognizer
```

Observed behavior from the RK3588 comparison harness:

```text
normal/close plate crops       OBB detector is best
high-resolution full scenes    old axis YOLO detector is more reliable
```

Normal or close plate images work well with the OBB detector because the plate remains large enough after resize to `LPR_IMAGE_SIZE = 960`. The OBB crop also preserves rotation and improves OCR on compact or two-row plates.

High-resolution weighbridge frames, such as 2880x1620 full-scene camera images, shrink the plate heavily before OBB inference. In the remote test harness, OBB scene detection missed all evaluated sessions even with scene tiling, while the older axis-aligned YOLO detector found usable low-confidence boxes when combined with geometry filters and session voting.

Remote high-resolution comparison summary:

```text
OBB scene profile       0/5 valid session majorities
axis YOLO scene profile 5/5 valid session majorities
```

Axis YOLO scene results from the remote harness:

```text
session_01 -> 14C-017.80
session_04 -> 14C-017.80
session_05 -> 24H-5016
session_08 -> 14C-017.80
session_10 -> 34H-605.21
```

Practical guidance:

```text
Keep OBB as primary for normal/close plates.
Use old axis YOLO only as a high-resolution full-frame fallback when OBB misses.
Do not lower detector confidence globally without geometry filters and session voting.
```

Relevant old axis detector path already exists in config:

```python
LP_DETECTOR_RKNN = os.path.join(LPR_DIR, "model", "LP_detector.rknn")
```

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
