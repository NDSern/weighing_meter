# Architecture

Weighbridge capture, LPR, and MQTT publication service. One session per
confirmed loaded scale cycle. The raw scale is authoritative; LPR only
labels a session.

## Module map

| Module | Responsibility |
| --- | --- |
| `weighing_service.py` | Entry point: logging, signals, start/stop loop only. |
| `services/runtime/bootstrap.py` | `construct_service()` / `shutdown_service()`: build and tear down every worker. |
| `services/session/session_manager.py` | Session lifecycle orchestration: start, promotion, empty dwell, end, finalization, publication. |
| `services/session/weight_state.py` | `WeighingSessionState`: raw/stable weight candidates, peaks, trend history. |
| `services/session/plate_registry.py` | Confirmed-plate registry, normalisation, OCR correction, plate DB counts. |
| `services/session/evidence_selection.py` | Pure selection of UNKNOWN_* result frames (target time, candidates, camera sync, detector boxes). |
| `services/session/result_builder.py` | Result payload and image-path/caption construction. |
| `services/scale/evidence.py` | Raw `weight_log` reads, loaded-cycle grouping, loaded-stable selection. |
| `services/review/duplicate_review.py` | Mock-only duplicate-cycle reviewer. |
| `services/storage/publish_outbox.py` | Durable publish queue with MQTT acknowledgement tracking. |
| `services/storage/retention_cleaner.py` | Image retention, verified MinIO cache, diagnostic archive, log compression. |

## Session lifecycle

1. **Trigger.** A session starts from a plate candidate or from rising scale
   weight (`scale_owned=True` for the latter).
2. **Promotion.** A plate-owned session is promoted to scale-owned as soon as a
   loaded frame arrives. Promotion runs **before** plate-loss completion, so a
   lost plate track can never finalize a loaded cycle early.
3. **Weight selection.** Only readings above `WEIGHT_THRESHOLD` (100 kg) may
   become the selected stable weight. Trailing empty readings cannot evict a
   loaded plateau. If no loaded stable value exists, the filtered peak then the
   raw peak is used.
4. **Termination.** A scale-owned session ends only after the scale stays
   `<= WEIGHT_THRESHOLD` for `SESSION_END_EMPTY_DWELL_SECONDS` (2 s), recorded
   as `scale_empty`. Loaded sessions ignore falling-trend termination.
5. **Finalization.** The terminal record is written once (idempotent on session
   id). Publication is queued in the outbox and acknowledged over MQTT.

There is no plate/time duplicate suppression. Two raw cycles separated by a
confirmed empty dwell are always two sessions; durable session ids and outbox
event ids provide retry idempotency. OCR plate equality never merges cycles.

## Publication

`publish_result()` builds the MQTT payload and result images, saves them
locally, enqueues an outbox event keyed by the session id, then activates it.
Because the event id is the session id, re-finalizing or replaying the same
session cannot publish twice. Acknowledged events are recorded in
`storage/publish_completed.db`.

Recognised plates publish merged/front/rear images. `UNKNOWN_OCR` and
`UNKNOWN_DETECTION` publish one image per available camera with an adaptive
caption and, for `UNKNOWN_OCR`, a plate bounding box.

## Unknown classification

`classify_unknown_plate(diagnostics)` returns `UNKNOWN_OCR` when any plate
detection/OCR stage produced data (detected regions, tracked regions, crop or
OCR attempts, candidates), otherwise `UNKNOWN_DETECTION`.

## Evidence classes

- **Published** result images: uploaded to MinIO, referenced by the MQTT payload.
- **Local-only diagnostics** (`no-stable`, `no-plate`, `peak-candidates`): never
  uploaded; kept as the sole copy on the bridge host.
- **Mock review / recovery audits**: local JSONL only; never touch the broker or
  MinIO objects.

## Duplicate review and recovery

`services/review/duplicate_review.py` compares adjacent finalized sessions using
raw scale continuity. Only a proven continuous above-threshold load produces an
action, and actions are written as **mock** MQTT/MinIO intents under
`storage/review-mock/`. Disabled unless `DUPLICATE_REVIEW_ENABLED` is set.

`scripts/recover_missed_session.py` replays an allowlisted audited miss. It is
dry-run by default and requires an explicit stopped-service assertion, an empty
scale, and frontend-absence confirmation. See `docs/operations.md`.

## Host policy

`config.py` `_HOST_POLICIES` keys per-host overrides by `WEIGHBRIDGE_ID`
(identity, cameras, retention). Defaults stay safe for any host; HP-02 narrows
diagnostic/log retention, enables verified log compression, and suppresses
redundant absorbed-active-session peak evidence.