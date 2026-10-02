# Storage

Two classes of files, with different durability rules.

## Published result images (MinIO-backed)

Written under `storage/weighbridge/YYYY/MM/DD/` and uploaded to MinIO. The
`VerifiedMinioCacheCleaner` deletes a local copy only after confirming the
exact object key exists remotely; a missing remote object leaves the local file
in place and logs a failure. Cache window is 3 days.

`photo-unchosen-cam{1,3}.jpg` files are **no longer produced**; the local-only
copy is never uploaded and previously caused MinIO cache mismatches.

## Local-only diagnostics (sole copy)

| Directory | Content |
| --- | --- |
| `storage/no-stable` | Sessions that ended without a usable weight. |
| `storage/no-plate` | Sessions that ended without a confirmed plate. |
| `storage/peak-candidates` | Shadow peak evidence for loads. |

These are never uploaded. Retention must therefore be chosen with forensic
value in mind:

- Each completed day directory is archived to `YYYY/MM/DD.tar.zst` after
  `DIAGNOSTIC_ARCHIVE_AFTER_DAYS` (default 2).
- Archives are deleted once older than `DIAGNOSTIC_ARCHIVE_RETENTION_DAYS`
  (default 30; HP-02 uses 14), matched by **evidence date in the path**, not
  file mtime.

Peak candidates carry a `category`. `absorbed_active_session` records describe
a load already covered by a real session and are redundant:
`SAVE_ABSORBED_PEAK_CANDIDATE_EVIDENCE=False` (HP-02) stops writing them while
still emitting the `images=0` metric. Older redundant records can be removed
with `scripts/cleanup_absorbed_peak_candidates.py` (dry-run by default, exact
recorded paths only, unsafe/malformed records left untouched).

## Logs

- Default `LOG_RETENTION_DAYS=30`; HP-02 uses `LOG_RETENTION_DAYS=14` with
  `LOG_COMPRESSION_ENABLED=True` and `LOG_COMPRESS_AFTER_DAYS=1`.
- Compression is verified: the gzip is decompressed and SHA-256 compared to the
  source before the source is removed. Open files are skipped.
- Expiry recognises both `*.log` and `*.log.gz` (and rotated watchdog `*.jsonl`).
- systemd journal is capped separately via
  `systemd/journald.conf.d/90-weighing-size.conf` (`SystemMaxUse=512M`).

## Free-space pressure

`IMAGE_STORAGE_TARGET_FREE_BYTES` (7 GiB) triggers pressure-driven deletion of
oldest peak/undetectable evidence first. `SESSION_FRAME_MIN_FREE_BYTES` (5 GiB)
reserves space for active session frames. Raw scale databases
(`scale_data/YYYY-MM-DD.db`) are retained 365 days and are not touched by image
retention.