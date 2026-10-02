# Weighing Meter

Production weighbridge service: reads a D2008 scale, recognizes truck plates from
RTSP cameras, saves evidence images, and publishes weighing events over MQTT.

## Runtime Flow

```text
RTSP cam1/cam3 -> CameraGrabber -> DetectCoordinator
              -> YOLOv8-OBB plate detector -> PP-OCR recognizer -> PlateTracker

D2008 scale -> D2008Reader -> SessionManager
            -> session start/end -> confirmed plate + stable weight

Publish -> local image save -> MinIO upload retry queue
        -> confirmed plate DB -> MQTT publish outbox
```

Sessions are scale-authoritative: each confirmed empty-separated scale cycle is its
own session, and durable finalization/outbox event IDs give retry idempotency. OCR
plates never merge distinct scale cycles.

## Run

```bash
python3 weighing_service.py
```

Production (systemd):

```bash
sudo systemctl restart weighing_service.service
systemctl is-active weighing_service.service
journalctl -u weighing_service.service -n 100 --no-pager
```

## Configuration

Precedence, lowest to highest:

1. non-secret defaults in `config.py`
2. host-local, untracked `config.local.py`
3. `WEIGHING_*` environment variables (root-owned
   `EnvironmentFile=/etc/weighing-meter/weighing.env`)

`validate_runtime_config()` runs before any worker starts and fails on missing or
malformed identities, endpoints, RTSP URLs, or credentials. Never commit
`config.local.py`, `weighing_service.service`, or real credentials. See
`SECURITY.md` and `docs/operations.md`.

## Export Sessions

Export finalized sessions (`published`, `duplicate`, `no_plate`, `no_weight`) from
`storage/session-finalization.db` to Excel. Read-only; requires `openpyxl`.

```bash
# one sheet per machine
python3 scripts/export_sessions.py \
  --db hp1=hp1-finalization.db --db hp2=hp2-finalization.db --out sessions.xlsx

# single combined sheet with a Machine column, one UTC start date
python3 scripts/export_sessions.py --db hp1=hp1.db --db hp2=hp2.db \
  --combined --date 2026-10-01 --out sessions.xlsx
```

`--from` / `--to` take ISO timestamps for a start-time range. Columns: Machine,
Start/End (UTC), Plate, Weight (kg), Plate status, Outcome, Unknown type, Weight
source, Session ID, End reason, Duration (s).

## Sessions and Publishing

- A plate is confirmed at `PLATE_CONFIRM_THRESHOLD` tracker votes, with at least
  `MIN_SELECTED_PLATE_HITS` main-OCR selections over
  `MIN_PLATE_OBSERVATION_SPAN_SECONDS`. Alternate OCR candidates cannot confirm alone.
- Registered correction (`registered_license_plates.json`): exact match, then unique
  family, then unique edit distance <= 1; ambiguous matches keep the OCR result.
- Weight-backed sessions without a confirmed plate publish `UNKNOWN_OCR` (plate
  detected, OCR unconfirmed) or `UNKNOWN_DETECTION` (no plate region), with
  per-camera images nearest the first sustained peak.
- Images save locally before MQTT enqueue. MinIO failures retry from
  `storage/upload_pending.jsonl`; MQTT events stay in `storage/publish_pending.jsonl`
  until acknowledged. Publish does not wait for MinIO upload.
- Diagnostics (`storage/no-stable`, `storage/no-plate`, peak candidates) are local-only.
- Optional mock duplicate review (`DUPLICATE_REVIEW_ENABLED`) only writes intents to
  `storage/review-mock/`; it never contacts MQTT/MinIO or deletes anything.

## LPR Models

Production bundle (RK3588, RKNNLite 2.3.2), hash- and contract-verified at startup:

```text
models/lpr/license_plate_detector.rknn     YOLOv8-OBB plate detector
models/lpr/license_plate_recognizer.rknn   PP-OCR CTC recognizer
```

OBB is best on normal/close plates but misses on high-resolution full scenes. An
optional axis-YOLO fallback (`LPR_FALLBACK_DETECTOR_MODEL`, empty by default) runs
only when primary OCR confidence is low; enable it per host in `config.local.py`.
Camera 1 2K rollout: `LPR_2K_DEPLOYMENT.md`.

Benchmark RKNN variants without touching production paths:

```bash
python3 benchmark_lpr.py storage/lpr-samples \
  --variant NAME DETECTOR RECOGNIZER CHARSET IMGSZ \
  --labels-from-filenames --output storage/lpr-benchmark.json
```

## Runtime Data

Untracked: `storage/`, `logs/`, `scale_data/`, `captures/`, `*.db*`.

```text
logs/YYYY-MM-DD/weighing_service.log   service logs
scale_data/YYYY-MM-DD.db               daily raw scale reads (365 days)
storage/session-finalization.db        finalized-session ledger
storage/upload_pending.jsonl           MinIO retry queue
storage/publish_pending.jsonl          MQTT retry queue
storage/weighbridge/YYYY/MM/DD/        published evidence
storage/undetectable/                  unknown-plate evidence
storage/no-stable/, storage/no-plate/  local-only diagnostics
storage/peak-candidates/               shadow peak evidence
```

Retention is host-scoped (see `docs/storage.md`). Files pending upload are never
removed. Below 7 GiB free, oldest uploaded evidence is removed first; session capture
keeps a 5 GiB reserve. Legacy root scale DB migrates once with
`scripts/migrate_scale_data.py` while the service is stopped.

## Deploy

```text
Git authority  /home/son/Projects/weighing_meter
Remote         git@github.com:NDSern/weighing_meter.git
HP-01          cang-hp1:/home/aibox-vnpay2/apps/weighing_meter
HP-02          cang-hp2:/home/vta-giavu-weightbridge2/apps/weighing_meter
```

```bash
cd /home/<device-user>/apps/weighing_meter
git fetch origin
python3 -m compileall weighing_service.py services scripts
sudo systemctl restart weighing_service.service
```

Restart only at an empty scale with both queues at 0. Never `git reset`, `git clean`,
force pull, or broad checkout on a production host.

## Verify

```bash
make verify   # model checksums, secret scan, lint, unit tests
wc -l storage/upload_pending.jsonl storage/publish_pending.jsonl   # expect 0 / 0
```

CI runs the same on Python 3.10 and 3.12. Manual MQTT probe:
`python3 tests/mqtt_probe.py [--publish]`. Audited-miss recovery
(`scripts/recover_missed_session.py`) is dry-run by default and restricted; see
`docs/operations.md`.

HEVC warnings such as `PPS id out of range` are usually camera stream noise.
`XMinioStorageFull` means uploads retry until MinIO has space.

## Docs

```text
docs/architecture.md   lifecycle, finalization, outbox, evidence flow
docs/operations.md     secrets, safe restart, queues, rollback, serial stall
docs/storage.md        retention classes, local-only evidence
SECURITY.md            credential handling and rotation
LPR_2K_DEPLOYMENT.md   camera 1 2K rollout and rollback
```

## Follow-ups

- Consumer-side replay visibility for re-published audited events.
- Review `wip/lpr-camera-experiments-2026-09-29` separately; not in production.
- Credential rotation and root-owned env file on both hosts (`SECURITY.md`).
