# Code Review — Remediation Plan

Source: 4 parallel read-only reviews on branch `refactor/code-cleanup` @ `31416e7`
(lifecycle/session, capture/LPR/tracking, storage/publish/retention/runtime,
scale/config/scripts). Severity is impact-on-production, not effort.

Legend: **C** critical, **H** high, **M** medium, **L** low.
Each fix = one atomic commit with a regression test, `make verify` green.

---

## Phase 1 — Correctness blockers (fix before next deploy)

### C1. CTC prefix-beam decode drops repeated adjacent characters
`services/pipeline/license_plate_recognition.py:224-234`

The `ch == end` branch writes `next_beams[prefix]` then immediately overwrites the
same key, so the `p_nb + p` term is lost and the `p_b + p` mass is credited to the
wrong key. Top-K can never emit a doubled character: `"11"`→`"1"`, `"122"`→`"12"`,
`"444"`→`"4"`. Greedy merge in `recognize_combined` (line 353) masks it only when
greedy already agrees; top-K can never *correct* a greedy miss on repeated characters,
which are common on VN plates.

Fix — canonical CTC prefix beam search, no overwrite:
```python
if ch == end:
    # p_b path: a genuine repeat -> prefix + ch
    nb_b, nb_nb = next_beams.get(prefix + ch, (-np.inf, -np.inf))
    next_beams[prefix + ch] = (nb_b, _logsumexp(nb_nb, p_b + p))
    # p_nb path: repeat without blank -> same prefix
    nb_b, nb_nb = next_beams.get(prefix, (-np.inf, -np.inf))
    next_beams[prefix] = (nb_b, _logsumexp(nb_nb, p_nb + p))
```
Test: synthetic logits for `1,blank,1,blank` must decode `"11"`; `"122"` and `"444"`
cases too. Add a real captured plate fixture with a doubled digit.

### C2. One exception permanently kills scale ingestion
`d2008_scale_reader.py:587-624` (calls 597-598, 622-623); catch 468; finally 481-485

`on_frame(frame)` and `_db.save(frame)` are unguarded. `sqlite3.Error` is not an
`OSError`, so it escapes the `except (SerialException, TypeError, OSError)`, the
`finally` sets `_running=False`/`state="stopped"`, and the reader thread dies with no
log and no `on_health`. The service stays "up" with zero scale input.

Fix:
- Wrap `on_frame` and `_db.save` in try/except with logging (mirror the `on_weight` guard).
- Add a broad `except Exception` in `_run` that records `last_error`, emits
  `on_health("stalled", …)`, sets `state="failed"` (see M10), and keeps the reconnect loop alive.

### C3. `stop()` races `release()` against an in-flight grab
`services/capture/frame_source.py:85-95` (release 88-90); loop 161-195

`_capture_lock` guards only the `self._capture` pointer; the grab loop calls
`cap.grab()`/`retrieve()` (165, 174, 177) unlocked. Concurrent `release()` on a live
`cv2.VideoCapture` is undefined behavior (crash in the FFmpeg backend).

Fix: the grab loop owns the capture lifetime. `stop()` sets `_stop_event` and joins;
it never touches `_capture`. The loop releases on exit. Add
`CAP_PROP_OPEN_TIMEOUT_MSEC`/`CAP_PROP_READ_TIMEOUT_MSEC` (see M6) so a hung read can't
block shutdown indefinitely.

### C4. Duplicate MQTT publishes (idempotency fails open)
`services/storage/publish_outbox.py:129-137`, compounded by `140-148`/`239-240`

`_is_published` catches `(OSError, sqlite3.Error)` and returns `False`, so a transient
DB read failure makes a published id look new → re-sent. Separately `_publish_event`
calls `_mark_completed` after a successful ack; if that raises, the loop requeues a
duplicate.

Fix:
- `_is_published` must fail closed: raise (or return `True`) when the completed-DB read fails.
- Persist a write-ahead "publishing" intent before the network call.
- On `_mark_completed` failure do NOT requeue — leave the event pending so the DB row is
  retried, never the MQTT send.

### C5. Night camera lights always authenticate with an empty password
`services/capture/camera_light_controller.py:14-26` + `services/runtime/bootstrap.py:273-275`

Bootstrap builds `CameraLightController([RTSP_URL, RTSP_URL_2, RTSP_URL_3], datetime.now)`
with no `api_password`; `_parse_target` takes the username from the RTSP URL but the
password from that arg, so every request sends Basic `<user>:`. All requests 401 →
lights never activate → degraded night LPR.

Fix: pass `api_password=RTSP_PASSWORD` at construction, or fall back to the URL
credential in `_parse_target` (`api_password or unquote(parsed.password or "")`).

---

## Phase 2 — High: silent failures and lost work

### H1. `mask_url_secret` leaks passwords containing `@`
`services/runtime/bootstrap.py:82-83` — regex stops at the first `@`. Fix: `urlsplit`
reconstruction or `re.sub(r"://[^/@]*@", "://***@", url)`.

### H2. Re-entry can dead-letter an already-published session
`services/session/session_manager.py:1314-1327` — `expected_outbox_id` is fabricated
(`outbox_event_id or (session_id if outcome=="published" else None)`). Activate only when
an outbox id was actually recorded; a missing id means success.

### H3. `finalization_store` swallows read errors as "not finalized"
`services/session/finalization_store.py:19-46` — distinguish not-found from read failure
(transient lock must not re-run the publish path); commit the schema DDL.

### H4. Upload worker can die permanently
`services/storage/image_save_worker.py:395-412` — wrap the whole loop body (including
retry/persist at 409-411, 449-457) so it survives; reset `_worker_started` on exit.

### H5. Retention tagger can rename an empty day dir mid-write
`services/storage/retention_cleaner.py:114-137` — add a mtime/ctime grace window; skip the
current day and yesterday.

### H6. `_bytes_written` accounting race
`services/capture/session_frame_spool.py:203,304,325` — guard every mutation with
`self._lock` (as the `+=` sites at 257/423 already do).

### H7. Session state touched off the lifecycle lock
`services/session/session_manager.py:804-814` (`on_status_change` write) and `297-319`
(`on_weight` read/format) — take `_lifecycle_lock`, or snapshot to locals.

### H8. OCR result attributed to a newly started session
`services/capture/detect_coordinator.py:411-435` — re-verify `session_context` immediately
before the first `add_observation`/`update_image` (a new session clears the tracker at
`session_manager.py:838`).

### H9. First detector miss resets a confirmed track
`services/capture/detect_coordinator.py:46-64` vs `80-94` — `observe([])` zeroes
`hits`/`valid`/`bbox`, making `PLATE_TRACK_STALE_SECONDS=3.0` miss-tolerance unreachable
and ending plate-owned sessions after a 1 s dwell (`session_manager.py:370-374`). Keep last
state on empty regions and let `expire()` handle staleness, or add a miss counter with
hysteresis.

---

## Phase 3 — Medium: timers, supervisors, validation

- **M1 wall-clock dwell** `session_manager.py:602,670,785-787,1006` — use
  `time.monotonic()` (as 210/372/393 already do); an NTP step can hold a session open forever.
- **M2 inert re-arm guard** `session_manager.py:162-165,401-426,819-833` +
  `weight_state.py:36-38` — `rearm_block_until`/`rearm_reference_weight` are only ever
  `0.0`/`None` in prod. Set them in `_archive_no_stable` or delete the dead fields/branches.
- **M3 `BackgroundWorker` lifecycle** `background_worker.py:35-45` — `stop()` must return
  whether the join succeeded; `start()` must refuse to double-start while a thread is alive.
- **M4 archive cleaner** `retention_cleaner.py:381-400` (listing outside try) and
  `_archive_day` `FileExistsError` wedge (418-435) — wrap listing per root; recover a
  half-archived day dir instead of failing.
- **M5 no I/O timeouts** — MinIO client (`image_save_worker.py:141-152`) has no HTTP
  timeout; OpenCV capture (`frame_source.py:146-165`) has no read timeout. Add both.
- **M6 silent thread deaths** — `session_frame_spool._capture_loop` (267-285) wrap loop
  body; `deferred_lpr_worker.py:300-305` count partial frame errors and feed the
  classification.
- **M7 migration duplicates** `scripts/migrate_scale_data.py:61-66,96-97,163-167` — refuse a
  non-empty destination per date or `INSERT OR IGNORE` with a deterministic key; add `--dry-run`.
- **M8 unvalidated scale config** `config.py:259-331` — positive-numeric checks for
  `SCALE_READER_STALL_SECONDS`, `BAUD_RATE`, `WEIGHT_THRESHOLD`; non-empty `SERIAL_PORT`.
- **M9 paho-mqtt pinning** `mqtt_service.py:110-118`, `requirements.txt:4` — pin the version
  and pass `callback_api_version=VERSION2` with the matching handler signatures.
- **M10 dead `"failed"` state** `d2008_scale_reader.py` + `weighing_service.py:65-66` — set
  `state="failed"` on fatal/after N reconnects, or delete the dead branch and drive failure
  via `on_health`.
- **M11 `AsyncLogger`** `async_logging.py:95-108,67-75,110-125` — warn once on overflow,
  report `_dropped` in `close()`, re-check `_closing` before restart, clear `_thread` on close.
- **M12 partial-startup leaks** `bootstrap.py:203-206,317-321` — set started flags so
  shutdown can always stop a half-started resource.
- **M13 `dead_letter` never expires** `dead_letter.py:30-37` — normalize aware/naive inside
  the `try`, return on `TypeError`.
- **M14 outbox weight formatting** `publish_outbox.py:245` — guard non-numeric
  `stable_weight`; **M15** `wait_for_pending` must also count `_pending_tasks`
  (`image_save_worker.py:154-163`); **M16** fsync the parent dir in `_persist_pending_locked`
  (335-343).
- **M17 reader shutdown/reset** `d2008_scale_reader.py:414-419` (always close DB/lock in
  `finally`); 435-442 reset stability history + parser on reconnect.

---

## Phase 4 — Low: hardening and dead code

- Sanitize plate-derived filenames/keys: `result_builder.py:20`,
  `recover_missed_session.py:354-358`.
- Remove dead code: write-only dedup state file (`session_manager.py:155-157,248-269,1627-1636`);
  unreachable `MAX_STABLE_WEIGHT_CANDIDATES` trim (`weight_state.py:91-102`);
  `frame_source.get_latest_frame` (97-101); unreachable `VehicleDetectCoordinator`/
  `VehicleTracker` (`bootstrap.py:224,267`); `del plates`/`del full_frame` (367,439-440);
  duplicated `mask_url_secret` (`frame_source.py:43` + `bootstrap.py:82`).
- `_start_session` weight window: clear then extend (`session_manager.py:866`).
- `unknown_capture.py:193,211` fall back when `metadata["started_at"]` is absent.
- `plate_registry.py:36-38` log via `log_fn`, not `print`.
- `_end_session` null `stable_weight`/`stable_decimal_pos` (1142-1256) and don't clear an
  unrelated `fatal_error` (1157); guard `_get_vehicle_summary` for `None` (288-295,302,800).
- `valid_candidates[1:]` → filter by `display != plate_text`
  (`detect_coordinator.py:341`, `deferred_lpr_worker.py:595`).
- `camera_light_controller.py:53-54` close the HTTP response; guard the target sets (18-19,92,103-106).
- `dated_tree.py:12,24-33` allow end-of-string/`.` after the compact day group.
- `d2008_scale_reader.py:146` accept only `+`/`-` sign bytes; `.db` suffix via `splitext` (225);
  restore `row_factory` in `finally` (322-328).
- `mqtt_service.py:57` reject blank/whitespace plates.
- `export_sessions.py:63` (URI-encode path), 159-171 (sanitize/truncate sheet title),
  88-96 (parse timestamps before comparing).
- `recover_missed_session.py:517-530` use a pidfile/flock rather than `/proc` substring matching.
- `cleanup_absorbed_peak_candidates.py:99` catch the `stat()` TOCTOU.

---

## Verified clean (no action)

`services/runtime/inference_lock.py`, `rknn_models.py`, `lpr_bundle.py`,
`services/scale/evidence.py`, `detector_obb_decode.py`; retention's pending-upload
protection; config precedence; MQTT QoS/ack outbox retry; path-containment in the spool,
deferred worker, and recovery scripts; `diagnostic_archive.save_frames` atomic write;
`plate_registry.editDistanceAtMostOne` / `preferDetailedLicensePlateCandidate`.

---

## Sequencing

1. Phase 1 (C1-C5) — one commit each, each with a regression test. C1 and C2 first.
2. Phase 2 (H1-H9).
3. Phase 3 (M1-M17).
4. Phase 4 (low) — batch cleanup commit(s).

After Phase 1 + 2: re-run `make verify`, then treat `refactor/code-cleanup` as the release
candidate for the pending git-hygiene deploy.
