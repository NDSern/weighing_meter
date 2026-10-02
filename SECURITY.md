# Security

## Credentials are not in the repository

Tracked code, examples, and docs contain **no** operational credentials. MQTT,
MinIO, and RTSP secrets must be supplied at runtime.

### Loading precedence

1. Non-secret code defaults in `config.py` (endpoints and camera addresses only).
2. `config.local.py` on the host for device policy (identity, cameras, policy).
3. `WEIGHING_*` environment variables for secrets (highest precedence).

Secrets are delivered through a root-owned env file, not through Git:

```
/etc/weighing-meter/weighing.env        # owner root, mode 0600
```

installed with the systemd drop-in `systemd/weighing_service.service.d/10-secrets.conf`
(`EnvironmentFile=-/etc/weighing-meter/weighing.env`). `systemd/weighing.env.example`
lists every supported variable. `config.local.example.py` shows device-config
precedence.

Relevant environment variables:

```
WEIGHING_WEIGHBRIDGE_ID
WEIGHING_RTSP_URL WEIGHING_RTSP_URL_2 WEIGHING_RTSP_URL_3
WEIGHING_RTSP_USERNAME WEIGHING_RTSP_PASSWORD
WEIGHING_MQTT_HOST WEIGHING_MQTT_PORT WEIGHING_MQTT_USERNAME WEIGHING_MQTT_PASSWORD
WEIGHING_MINIO_ENDPOINT WEIGHING_MINIO_ACCESS_KEY WEIGHING_MINIO_SECRET_KEY
```

`validate_runtime_config()` runs before any worker starts and fails fast when a
required endpoint, identity, or credential is missing; URL credentials are
masked in logs.

## Secret scanning

`scripts/scan_secrets.py` scans tracked files for non-empty MQTT/MinIO/RTSP
credential assignments and `rtsp://user:pass@host` URLs. It is wired into
`make verify` and CI. The example files are skipped by path, never by value.

## Rotation

Removing secrets from the current tree does not revoke them: they remain in
history. Rotation is an operational action owned with the backend team:

1. Provision new MQTT and MinIO credentials.
2. Rotate camera credentials with camera consumers coordinated.
3. Install the new env file on each host and validate connections.
4. Revoke the old credentials only after both canaries pass.

Do not rewrite Git history as part of routine cleanup; that is a separate,
coordinated decision (see `docs/operations.md`).

## Reporting

Do not paste credentials, host env files, or full configuration into issues,
chat, or logs. Report only variable names and fingerprints.