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
services/runtime/bootstrap.py                service construction and shutdown wiring
services/session/session_manager.py         session lifecycle and publishing orchestration
services/session/weight_state.py            loaded-weight candidate and trend state
services/session/plate_registry.py          registered-plate correction cache
services/session/evidence_selection.py      UNKNOWN_* result-image/frame selection
services/session/result_builder.py          publish payload and result-image construction
services/session/finalization_store.py      finalized-session ledger
services/session/plate_store.py             confirmed-plate persistence
services/session/diagnostic_archive.py      no-stable/no-plate evidence archive
services/scale/evidence.py                  raw scale reads, cycles, loaded-stable mode
services/review/duplicate_review.py         mock-only duplicate session reviewer
services/review/mock_integrations.py        mock MQTT/MinIO removal intents
services/storage/image_save_worker.py       local image save and MinIO retry queue
services/storage/publish_outbox.py          durable MQTT outbox
services/storage/retention_cleaner.py       retention, log compression, archive cleaner
services/capture/camera_light_controller.py session-driven camera light control
services/tracking/plate_tracker.py          plate aggregation and image selection
services/tracking/vehicle_tracker.py        vehicle stability/left detection
scripts/recover_missed_session.py           restricted audited-miss recovery tool
scripts/cleanup_absorbed_peak_candidates.py safe exact peak-evidence cleanup
```

## Local Configuration

Production devices keep host-specific settings in untracked `config.local.py`.

Common overrides:

```python
RTSP_HOST_CAM1 = "192.168.1.181"   # non-secret camera addresses
RTSP_USERNAME = "..."              # secrets belong in the environment, see below
RTSP_PASSWORD = "..."
CAM2_RESULT_CROP = "left"  # left, right, or full
WEIGHBRIDGE_ID = "..."
```

Effective setting precedence, lowest to highest:

1. non-secret code defaults in `config.py`
2. host-local `config.local.py` (device policy)
3. `WEIGHING_*` environment variables, normally supplied by the root-owned
   `EnvironmentFile=/etc/weighing-meter/weighing.env` systemd drop-in

`validate_runtime_config()` runs before any worker starts and fails startup when
required identities, endpoints, RTSP URLs, or credentials are missing or malformed.
RTSP URLs are derived from `RTSP_USERNAME`/`RTSP_PASSWORD`/`RTSP_HOST_CAM*`/`RTSP_PATH`
unless explicitly overridden. Never commit real credentials; track them only in the
documented `WEIGHING_*` variables. See `SECURITY.md` and `docs/operations.md`.

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
  --fastalpr-variant fastalpr /tmp/alpr-rknn/yolo-v9-t-384-license-plates-pre-nms-fp16-rk3588.rknn /tmp/alpr-rknn/cct_xs_v2_global-fp16-rk3588.rknn \
  --labels-from-filenames \
  --manifest storage/lpr-samples/manifest.json \
  --output storage/lpr-benchmark.json
```

Each repeated `--variant` takes its own detector, recognizer, charset, and image size. `--fastalpr-variant` takes a YOLOv9 detector and CCT-XS v2 recognizer using their fixed 384x384 and 128x64 contracts. Output includes artifact hashes, source dimensions, detector/OCR timing, process RSS, detection and valid-format rates, optional exact-match labels, and per-image plate results. `--labels-from-filenames` reads labels from `UUID_PLATE_photo-*.jpg` archives and ignores `UNKNOWN`. Explicit manifest entries override inferred labels. Variants load sequentially on the selected NPU core; use `--core 1` or `--core 2` when needed.

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
logs/YYYY-MM-DD/weighing_service.log  service logs
scale_data/YYYY-MM-DD.db              daily scale readings (365-day retention)
scale_data/scale_data.archive.db      legacy root database after migration
confirmed_license_plates.db           confirmed plate counts
storage/upload_pending.jsonl          MinIO upload retry queue
storage/publish_pending.jsonl         MQTT publish retry queue
storage/weighbridge/YYYY/MM/DD/       published/publishable evidence images
storage/undetectable/                 unknown plate evidence
storage/no-stable/YYYY/MM/DD/         local-only no-usable-weight diagnostics
storage/no-plate/YYYY/MM/DD/          local-only no-confirmed-plate diagnostics
storage/peak-candidates/              shadow peak evidence
storage/review-mock/                  mock-only duplicate-review intents
storage/recovery-audit.jsonl          append-only recovery audit
```

Retention is host-scoped. Defaults: logs and diagnostics 30 days (logs compressed
after 1 day once enabled), `storage/undetectable`/`storage/peak-candidates` 20 days,
raw scale 365 days. The HP-02 host policy narrows logs and no-stable/no-plate
diagnostics to 14 days, enables verified log gzip, and suppresses redundant
absorbed-active-session peak evidence. `storage/no-stable`, `storage/no-plate`, and
`photo-unchosen` never had MinIO copies and are local-only. See `docs/storage.md`.

## Unknown Sessions

Weight-backed sessions without a confirmed plate publish one of two explicit values:

- `UNKNOWN_OCR`: detector found a plate, but OCR did not produce a confirmed plate.
- `UNKNOWN_DETECTION`: no plate region was detected.
- `UNKNOWN_OCR` and `UNKNOWN_DETECTION` publish thumbnails nearest the first sustained local peak. They fall back to filtered peak, raw peak, then recorded weight when local peak timing is unavailable. A camera is omitted when its nearest frame is more than 1 second from that timestamp or more than 1 second from the synchronized camera group, preventing stale cameras from creating mixed-time photo sets.
- RTSP sources use a native GStreamer pipeline with a 500ms bounded jitter buffer and `appsink max-buffers=1 drop=true sync=false`. This drops superseded decoded frames instead of allowing decoder queues to drift behind real time. Rockchip hosts decode H.265 through `mppvideodec` and convert its stride-padded NV12 output after the one-frame sink; other hosts use `avdec_h265`.

No scale-reading, stability, or peak-selection rule changes. Each photo's `captured_at` remains its actual frame acquisition time, not the target time.

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
local image save
confirmed plate DB increment
MQTT outbox enqueue
```

Sessions are scale-authoritative: each confirmed empty-separated raw scale cycle is
its own session, and durable finalization/outbox event IDs provide retry idempotency.
OCR plates never collapse distinct scale cycles (the legacy same-plate/time skip was
removed).

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

Result images are the merged/front/rear set for recognised sessions and the per-camera
set for `UNKNOWN_OCR`/`UNKNOWN_DETECTION`. Local-only diagnostic evidence
(`storage/no-stable`, `storage/no-plate`) and shadow peak evidence are never uploaded to
MinIO. The former local-only `photo-unchosen` files were removed: they created MinIO
cache-cleaner mismatches without being part of any payload.

Optional mock-only duplicate review (`services/review/*`, disabled unless
`DUPLICATE_REVIEW_ENABLED` is set per host) can flag a losing session of a
continuous-load pair. It only appends structured mock MQTT/MinIO intents to
`storage/review-mock/`; it never contacts the broker/MinIO or deletes anything.

Audited-miss recovery is operationally restricted: `scripts/recover_missed_session.py`
is dry-run by default, allowlists exact session IDs, requires `--service-stopped`,
explicit frontend-absence confirmation, and (for image-less cases) explicit
`--allow-image-less` approval. See `docs/operations.md`.

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
git@github.com:NDSern/weighing_meter.git
```

Deploy reviewed code on a device:

```bash
cd /home/<device-user>/apps/weighing_meter
git fetch origin
# apply only reviewed files/hunks; hosts carry unrelated local edits
python3 -m compileall weighing_service.py services scripts
sudo systemctl restart weighing_service.service
systemctl is-active weighing_service.service
```

Never use `git reset`, `git clean`, force pull, or broad checkout on a production host.

## Verification

Local checks (models, secret scan, lint, unit tests with warnings as errors):

```bash
make verify
```

CI runs the same sequence on Python 3.10 (production parity) and 3.12.

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

Image retention runs hourly. Evidence older than the host retention window is removed
first (default 20 days for `storage/undetectable`/`storage/peak-candidates`; see
`docs/storage.md` for per-host diagnostic windows). When free disk falls below 7 GiB,
oldest uploaded local JPEG/PNG evidence is removed until the target is restored; files
still queued in `storage/upload_pending.jsonl` are never removed. Session capture keeps a
separate 5 GiB hard reserve. Completed diagnostic days are archived after 2 days and
expired by evidence date, so archive/expiry never depends on file mtime.

Recover capture after a stopped service or low-disk incident:

```bash
scripts/recover_image_capture.sh
```

The script installs a persistent 1 GiB system-journal cap, vacuums archived journals, runs
storage cleanup, verifies the 5 GiB session reserve, and starts an inactive service. It
never restarts an active service; an inactive service is allowed to recover its durable
session markers on startup.

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

## Documentation

```text
docs/architecture.md   scale-authoritative lifecycle, finalization, outbox, evidence flow
docs/operations.md     secrets, safe restart, queue/evidence checks, rollback, serial-stall
docs/storage.md        retention classes and local-only evidence
SECURITY.md            credential handling and rotation responsibilities
LPR_2K_DEPLOYMENT.md   camera 1 2K rollout, health checks, rollback
```

## Known Follow-ups

- Add a serial-reader watchdog: the reader thread can stall silently while the port stays
  open, freezing scale ingestion until service restart. Detect stale frames and recover.
- Consumer-side replay visibility: re-published audited events must be verifiable on the
  frontend; compensation for a delivered MQTT event needs a consumer protocol.
- Review and separately integrate the preserved LPR/camera experiment branch
  (`wip/lpr-camera-experiments-2026-09-29`); it is intentionally not part of canonical
  production.
- Coordinate credential rotation and install the root-owned environment file on both
  hosts (see `SECURITY.md`).
