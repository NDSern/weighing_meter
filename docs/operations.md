# Operations

## Secrets

Tracked code contains **no** credentials. The resolution order is:

1. Non-secret defaults in `config.py`
2. Host-local `config.local.py` (untracked) for identity and device policy
3. Environment values for secrets, loaded by systemd, highest precedence

Install the environment file as root with mode 0600:

```bash
sudo install -d -m 0755 /etc/weighing-meter
sudo install -m 0600 systemd/weighing.env.example /etc/weighing-meter/weighing.env
sudo chown root:root /etc/weighing-meter/weighing.env
sudo editor /etc/weighing-meter/weighing.env
```

Install the drop-in so the unit reads that file:

```bash
sudo install -d -m 0755 /etc/systemd/system/weighing_service.service.d
sudo install -m 0644 systemd/weighing_service.service.d/10-secrets.conf \
  /etc/systemd/system/weighing_service.service.d/10-secrets.conf
sudo systemctl daemon-reload
```

Recognized variables are listed in `systemd/weighing.env.example`. A value set
in the environment overrides `config.local.py`; an unset value falls back to it.

Startup fails before any worker starts when required secrets, identity,
endpoints, or RTSP credentials are missing or malformed
(`config.validate_runtime_config`). URL credentials are masked in logs.

Rotate credentials in this order: provision the new value, install it in the
environment file on every host, verify MQTT connect and MinIO read/write, and
revoke the old value only after both bridges pass a weighted transaction.

Detect secrets accidentally added to tracked files:

```bash
python3 scripts/scan_secrets.py
```

## Safe restart

Restarting during a loaded cycle can split one vehicle into two sessions.

1. Confirm the raw scale is at or below `WEIGHT_THRESHOLD` (100 kg) for at least
   2 seconds (`SESSION_END_EMPTY_DWELL_SECONDS`).
2. Confirm the MQTT pending queue, image upload queue, and deferred LPR
   `storage/lpr-spool/pending/` and `processing/` queues are empty.
3. Restart, then verify: service active, `NRestarts` stable, fresh raw scale
   rows, and no traceback or import error.

```bash
systemctl restart weighing_service.service
systemctl show weighing_service.service -p ActiveState -p SubState -p NRestarts
```

## Queue and evidence checks

```bash
wc -l storage/publish_pending.jsonl        # should be 0
ls storage/weighbridge/$(date +%Y/%m/%d)   # result images for today
```

Local-only evidence (`storage/no-stable`, `storage/no-plate`) is never uploaded
to object storage; it is bounded by `DIAGNOSTIC_ARCHIVE_AFTER_DAYS` and
`DIAGNOSTIC_ARCHIVE_RETENTION_DAYS`.

## Serial scale stall

If raw scale rows stop advancing while the service stays up, the reader thread
can stall without a logged error. Diagnose before restarting:

```bash
# newest persisted reading (hosts have no sqlite3 CLI)
python3 - <<'PY'
import sqlite3, glob
db = sorted(glob.glob("scale_data/*.db"))[-1]
with sqlite3.connect(db) as c:
    print(db, c.execute("SELECT max(timestamp) FROM weight_log").fetchone()[0])
PY
```

If the feed is frozen, restart at confirmed empty scale. A short raw serial
capture decoded with `d2008_scale_reader.D2008Parser` distinguishes an
indicator/cable fault (no valid checksums) from a reader stall (valid frames
throughout).

## Deployment

Deploy reviewed files or hunks only; never `git pull`, `git reset`, `git clean`,
or a broad `checkout` on a production host:

```bash
git fetch origin
# copy reviewed files/hunks from the canonical branch, then:
python3 -m compileall -q weighing_service.py services scripts config.py
```

Preflight before restart:

- raw scale fresh and `<= 100 kg` for at least 2 seconds
- no active session or spool finalization in progress
- deferred LPR `pending/` and `processing/` queues drained
- `storage/publish_pending.jsonl` empty and image upload queue empty
- service active and `NRestarts` stable
- timestamped backup of every file to be replaced

Observe for at least 20 minutes and through one real loaded transaction: fresh
scale ingestion, correct stable/peak selection, `scale_empty` finalization, one
queued-and-acknowledged publication, image upload success, and no traceback,
restart, backlog, or memory/spool growth. Run one host as canary before the
other.

## Inference timing and fairness

Each camera's inference lock admits waiters FIFO within the live and deferred
classes, retaining live/deferred class arbitration. A detector cannot repeatedly
reacquire ahead of an already-waiting live OCR job. OCR checks session context
and enablement again after acquiring the lock, before expensive inference.

Rate-limited timing logs use monotonic durations: `queue_ms` is time before the
OCR worker takes the job, `wait_ms` is inference-lock admission time,
`inference_ms` includes recognition and any fallback, and `total_ms` includes
queueing, lock wait, and inference. Detection logs split lock wait and inference
too. Compare these fields rather than interpreting the old combined OCR elapsed
time as model compute time. Observation timestamps remain wall-clock values.

## Spool cleanup ownership

Acknowledgement cleanup and the background cleanup sweep share a cleanup lock.
The sampler skips a sweep when acknowledgement owns it; directory deletion does
not hold the active-frame capture lock. Missing cleanup markers are idempotent,
unreadable markers remain for retry, and corrupt markers go to `failed/`.
Partial deletion accounts only for removed bytes before retrying the remainder.
Keep failed/orphan evidence for normal quarantine retention unless an explicit
recovery decision calls for moving it to a backup.

## Camera light failures

Light-control failures back off per camera: 60, 120, 240, 480, then 900 seconds
(capped). Retry occurs on a subsequent controller call after the deadline;
there is no autonomous retry timer. Successful control clears the backoff and
logs recovery. A failed off request retains the known-on state for retry.
HTTP 404 disables that target for the process; HTTP 500 remains retryable.
Errors log HTTP status or exception type, never credential-bearing error text.

HTTP light credentials can differ from RTSP credentials. Configure
`CAM1_HTTP_USERNAME` / `CAM1_HTTP_PASSWORD` through `CAM3_HTTP_USERNAME` /
`CAM3_HTTP_PASSWORD` in host-local configuration, or use the corresponding
`WEIGHING_`-prefixed environment variables. Slots follow `RTSP_URL`,
`RTSP_URL_2`, and `RTSP_URL_3`, respectively. Each unset (`None`) field falls back
to its RTSP counterpart; an explicit empty string remains blank. Environment
values override local configuration even when empty for these HTTP fields.
Leave unused entries commented in the environment template to preserve fallback.

On 2026-11-02, read-only GET probes confirmed `/Images/1/IrCutFilter` and variable
white-light capability on HP2 cameras `.181` and `.177` using device HTTP
credentials, while service RTSP credentials returned HTTP 401. Camera `.20`
returned HTTP 500; its light route remains unresolved. Keep credentials outside
tracked files and logs. Successful GET access does not establish write
permission; brightness changes require a separate operational check.

## Recovery tooling (restricted)

`scripts/recover_missed_session.py` replays an audited missed transaction. It is
**operationally restricted**: hard allowlist of session ids, dry-run by default,
and controls that must all be satisfied before apply.

```bash
# dry run — prints the plan and changes nothing
python3 scripts/recover_missed_session.py <session_id>
```

Apply requires: `--apply`, `--service-stopped` (the tool verifies no
`weighing_service.py` process is running), the scale at confirmed empty, and
`--frontend-absence-confirmed <session_id>` after checking the consumer really
did not receive the event. Sessions without attributable evidence additionally
require `--allow-image-less <session_id>` and explicit operator approval.

Recovery preserves the original terminal outcome, records an append-only audit
in `storage/recovery-audit.jsonl`, and reuses the original session id as the
outbox event id so a retry can never publish twice. Before any retry, inspect
`storage/publish_pending.jsonl` and `storage/publish_completed.db`.

Duplicate review actions under `storage/review-mock/` are **mock only** and
never touch the broker or MinIO objects.

## Rollback

Restore only timestamped pre-deploy files and restart at confirmed empty scale.
Never `git reset`/`git clean` on a host. Keep new secrets in place and roll code
back independently; do not restore revoked credentials.

## Known follow-ups

- Serial-reader watchdog: detect a stalled reader (fresh lines expected but not
  arriving) and restart ingestion without a full service restart.
- Consumer-side replay visibility: a compensation protocol so a retracted or
  replayed event is observable to the backend.
- LPR/camera experiments: preserved separately on
  `wip/lpr-camera-experiments-2026-09-29`, not part of this change.
- Credential rotation and env-file installation on both hosts (see
  `SECURITY.md`).
## Offline replay (read-only)

`scripts/replay_weight_sessions.py` re-runs the live session lifecycle over
recorded `scale_data/YYYY-MM-DD.db` logs without touching a host. It is
strictly read-only (sqlite `mode=ro`) and writes nothing outside `--json`:

```bash
python3 -B scripts/replay_weight_sessions.py scale_data/2026-10-24.db \
    --csv /path/to/can_hang_2026-10-24.csv --json /tmp/replay.json
```

It prints a summary (session count, end reasons, loaded sessions) and, with
`--csv`, a one-to-one match of replayed sessions against CAN HANG gross/tare
targets. Matches are an upper-coverage estimate, not a recall claim.
