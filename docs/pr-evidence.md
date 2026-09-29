# Canonical Production Rollout — PR Evidence

Branch: `refactor/canonical-production`
Base: `origin/main` (`3e88504`) — confirmed ancestor of this branch (fast-forwardable).
Repository: https://github.com/NDSern/weighing_meter

## 1. Feature coverage matrix

Consolidation base: `d5310f0` (shared ancestor of all production feature branches).

| Source branch | Commit | Feature | Disposition |
|---|---|---|---|
| `fix/scale-authoritative-session-publish` | `679b272` | Trust loaded scale cycles (empty readings never enter stable candidate; snapshot threshold; remove plate/time dedup) | Kept (canonical core) |
| `fix/scale-authoritative-session-publish` | `350f186` | Stage audited missed sessions (recovery tool) | Kept |
| `fix/scale-authoritative-session-publish` | `b8cf2bd` | Duplicate session review for any host | Kept — supersedes HP-01 `5b77c1f` |
| `fix/scale-authoritative-session-publish` | `9341a62` | Recovery: explicit image-less audit replay | Kept |
| `fix/hp1-loaded-session-lifecycle` | `9430244`,`d19dbce` | Camera white-light control per LPR session + API variants | Ported (controller, hooks, wiring, tests) |
| `fix/hp1-loaded-session-lifecycle` | `5b77c1f` | HP-01 mock duplicate removal | Superseded by `b8cf2bd` |
| `fix/hp2-storage-reduction` / `fix/storage-retention` | `e0f79b5`,`1eb3d9c` | Bounded diagnostics, drop `photo-unchosen`, verified gzip logs, absorbed-evidence suppression, cleanup script, journald 512M | Ported |

### Deliberately excluded

- Old (pre-`679b272`) host lifecycle code: restored `_should_skip_duplicate_publish`, sub-threshold stable selection, and empty-reading admission are all rejected in favour of scale-authoritative behaviour.
- HP-01-specific docstrings in `services/review/duplicate_review.py` (generic version kept).
- The dirty LPR/camera experiment work in `/home/son/Projects/weighing_meter` is preserved on local-only branch `wip/lpr-camera-experiments-2026-09-29` (commit `8d6c410`) and is **absent** from this PR.

## 2. Module responsibility map (before → after)

| Module | Before | After | Responsibility |
|---|---|---|---|
| `weighing_service.py` | 392 lines | 88 lines | entry point, signals, main loop |
| `services/runtime/bootstrap.py` | — | 369 lines | construction + ordered shutdown |
| `services/session/session_manager.py` | 2606 lines | 2025 lines | lifecycle orchestration, finalization, publication |
| `services/session/weight_state.py` | — | 136 lines | loaded-weight candidate state |
| `services/session/plate_registry.py` | — | 173 lines | plate registry + correction |
| `services/session/evidence_selection.py` | — | 329 lines | unknown-plate frame selection |
| `services/session/result_builder.py` | — | 94 lines | payload + result-image construction |
| `services/scale/evidence.py` | — | 159 lines | shared raw-scale read/continuity |

`session_manager._load_unknown_publish_frames` fell from 344 to 96 lines. Largest remaining methods are lifecycle/finalization orchestration (`finalize_deferred_session` 226, `_end_session` 115, `_start_session` 97), intentionally not split.

## 3. Test matrix

`make verify` (models checksum + secret scan + Ruff + suite with `ResourceWarning` as error), Python 3.10 and 3.12 in CI.

- Full suite: **302 tests, OK**, zero ResourceWarnings.
- 23 test modules, including new coverage: `test_camera_light_controller`, `test_evidence_selection`, `test_peak_candidate_cleanup`, `test_scale_evidence_fixtures`, `test_scan_secrets`, `test_storage_retention`, `test_recover_missed_session`, `test_duplicate_review`.
- Characterization covered: loaded stable surviving trailing empty readings; 2-second confirmed-empty termination; plate promotion before plate-loss completion; same OCR plate in separate cycles publishing as distinct session IDs; finalization/outbox retry idempotency; `UNKNOWN_OCR`/`UNKNOWN_DETECTION` image and box behaviour; camera-light lifecycle; host-specific retention + absorbed-evidence policy.

## 4. Secret scan

`python3 scripts/scan_secrets.py` → `OK: no tracked credentials found`.

Tracked MQTT/MinIO/RTSP credentials were replaced with empty defaults; secrets load from root-owned `/etc/weighing-meter/weighing.env` (mode 0600) via a systemd drop-in, with `config.local.py` for device policy. `validate_runtime_config()` fails fast before workers start. Real credential rotation and env-file installation remain an operations step (Phase 7) and are not code-complete.

## 5. Rollout checklist (Phase 7)

- [ ] Host drift inventory for `cang-hp1` and `cang-hp2` (tracked/untracked vs canonical vs WIP) — any unexplained delta blocks that file.
- [ ] Common preflight per host: fresh valid raw scale; weight ≤100 kg for ≥2 s; no active session/spool finalization; MQTT pending 0; upload pending 0; service active, NRestarts stable; checksums recorded; timestamped backups; syntax/import check against host-local config.
- [ ] HP-01 canary (preserve camera overrides, light policy, duplicate-review state): reviewed hunks only, rotated secrets/drop-in, restart once at empty scale, ≥20 min + one real weighted transaction (fresh ingestion, correct stable/peak, `scale_empty`, single queued+acked publish, upload success, light lifecycle, no traceback/backlog).
- [ ] HP-02 canary only after HP-01 acceptance (preserve storage policy): additionally verify 14-day cleaner, gzip verify, absorbed `images=0`, no `photo-unchosen` return, disk trend safe, serial feed fresh after restart.
- [ ] 24-hour observation on both hosts via raw-cycle ↔ finalization/ack correlation.

## 6. Rollback

- Restore only timestamped pre-deploy files and systemd secret drop-ins; restart only at confirmed empty scale with empty queues.
- Never `git reset`/`git clean`/force pull/broad checkout on a host.
- Do not restore revoked credentials; roll code back independently of credential rotation.
- If an outbox event was staged, inspect completed/pending IDs before retry; never duplicate a published session.

## 7. Known follow-ups

- Serial-reader watchdog (HP-02 freeze: thread alive but no valid frames persisted; restart restored ingestion).
- Consumer-side replay visibility / compensation protocol.
- Separate LPR experiment review (`wip/lpr-camera-experiments-2026-09-29`).
- Credential rotation + env-file install on both hosts.