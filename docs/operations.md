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
2. Confirm the MQTT pending queue and image upload queue are both empty.
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
- `storage/publish_pending.jsonl` empty and image upload queue empty
- service active and `NRestarts` stable
- timestamped backup of every file to be replaced

Observe for at least 20 minutes and through one real loaded transaction: fresh
scale ingestion, correct stable/peak selection, `scale_empty` finalization, one
queued-and-acknowledged publication, image upload success, and no traceback,
restart, backlog, or memory/spool growth. Run one host as canary before the
other.

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