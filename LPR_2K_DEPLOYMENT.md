# Fine-Tuned LPR 2K Deployment

This branch prepares the RK3588 service for the fine-tuned detector/OCR bundle and an optional 2K cam1 stream. Camera `.181` route discovery and live acceptance remain pending while the camera is offline.

## Bundle

Required files and SHA-256 values:

```text
models/lpr/license_plate_detector.rknn
4c580314148bde47920e20bd9e13969d96017fdbf64a0ed49a1679c45b1d3be8

models/lpr/license_plate_recognizer.rknn
ec88eff23206cf8c6fa609e3c9130e7b2d4caba31c846cc03969680ca2ce4eb3

models/lpr/charset.txt
d798cc4724b12e455112439aca5b51ee29da71a264663e5dbfda25ed24a391f4

services/pipeline/detector_obb_decode.py
d370fa5aabb1f4e9d17349b09a5095a42689bbe709b2c55a416d684822c727ba
```

Verify from repository root:

```bash
(cd models/lpr && sha256sum -c SHA256SUMS)
```

## Host Preflight

Record before deployment:

```bash
python3 -c 'import sys; assert sys.version_info >= (3, 10); print(sys.version)'
git rev-parse HEAD
sha256sum models/lpr/license_plate_detector.rknn
sha256sum models/lpr/license_plate_recognizer.rknn
sha256sum models/lpr/charset.txt
python3 -c 'import rknnlite; print(rknnlite.__version__)'
systemctl is-active weighing_service.service
```

Preserve host-local configuration and runtime queues:

```bash
cp config.local.py config.local.py.pre-lpr-2k
```

Do not delete or replace `storage/lpr-spool`, publish outbox files, or finalization databases.

## Offline Camera Deployment

When `.181` remains offline, keep existing route and disable the resolution gate:

```python
CAM1_EXPECTED_RESOLUTION = None
```

The service can start without cam1. Model startup validation still runs before MQTT and camera workers. A shape, value-domain, dynamic-width, or hash mismatch aborts startup.

Deploy code and restart:

```bash
sudo systemctl restart weighing_service.service
systemctl is-active weighing_service.service
journalctl -u weighing_service.service -n 200 --no-pager
```

Required startup evidence:

```text
Fine-tuned LPR runtime contract passed
RKNN lpr_detector_cam1 ready
RKNN lpr_recognizer_cam1 ready
RKNN lpr_detector_cam3 ready
RKNN lpr_recognizer_cam3 ready
```

Absence of cam1 frames is expected while `.181` is offline. Cam3, scale reader, deferred queue, and MQTT/outbox health must remain unchanged.

## 2K Route Discovery

Activate only after `.181` is reachable at the intended camera MAC:

```text
5a:5a:00:d0:ca:e8
```

Probe candidate routes without writing credentials to logs. Accept a route only after three open/decode/close cycles and 100 consecutive frames with stable dimensions.

Record exact verified values for later use in untracked `config.local.py`:

```python
RTSP_URL = "rtsp://user:pass@192.168.1.181:554/<verified-main-route>"
CAM1_EXPECTED_RESOLUTION = (2880, 1624)  # Replace with exact decoded tuple.
```

Do not activate the route in production until clean full-scene samples confirm detector recall, session OCR accuracy, latency, memory, and deferred queue stability. On an isolated test instance, restart and require:

```text
[cam1] RTSP decoded source=<expected width>x<expected height>
```

Any mismatch logs `RTSP resolution rejected`, discards the frame, and reconnects. It does not silently process the wrong stream.

## New No-Result Evidence

Weight-backed sessions without a confirmed plate retain the existing `session_no_plate` metric and add `weight_backed_lpr_no_result` with one classification:

```text
no_plate_detection
lpr_frames_unavailable
detector_inference_error
crop_failed
plate_detected_ocr_blank
plate_detected_ocr_low_confidence
plate_detected_ocr_invalid_format
ocr_inference_error
no_confirmed_plate_after_voting
```

Per-camera counters, tracker rejection reason, and representative frame paths are written to the no-plate archive JSON and finalization ledger. Total frame-processing failures retry, then enter the existing failed spool with a typed dead-letter metric.

`LPR_OCR_MIN_CONFIDENCE` defaults to `None`. Do not enable it until labeled data calibrates the conditional 38-class confidence.

## Rollback

Rollback as one unit:

1. Stop service.
2. Restore previous Git revision.
3. Restore `config.local.py.pre-lpr-2k`.
4. Confirm previous detector, recognizer, and charset hashes.
5. Start service.
6. Confirm scale, cam3, MQTT/outbox, and deferred spool health.

Commands:

```bash
sudo systemctl stop weighing_service.service
git switch --detach <previous-sha>
cp config.local.py.pre-lpr-2k config.local.py
sudo systemctl start weighing_service.service
systemctl is-active weighing_service.service
journalctl -u weighing_service.service -n 200 --no-pager
```

Do not clear pending runtime data during rollback.

## Deferred Validation

Test code is included but not executed in this implementation pass. Live `.181` validation, clean 2K corpus capture, benchmark comparison, and soak testing remain required before claiming 2K accuracy or performance acceptance or activating the 2K route in production.
