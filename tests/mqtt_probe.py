#!/usr/bin/env python3
"""Publish a test MQTT event using the latest stored images and scale row."""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

from config import (
    CAPTURE_DIR,
    MQTT_HOST,
    MQTT_KEEPALIVE,
    MQTT_PASSWORD,
    MQTT_PORT,
    MQTT_QOS,
    MQTT_TOPIC,
    MQTT_USERNAME,
    SERVICE_DIR,
    WEIGHT_THRESHOLD,
)
from mqtt_service import build_weighbridge_payload


MQTT_CLIENT_ID = f"smartport-weighbridge-test-{uuid.uuid4()}"

# Default transaction type
DEFAULT_TRANSACTION_TYPE = "gate_in"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PHOTO_SUFFIXES = {
    "photo-merged": "merged",
    "photo-front": "front",
    "photo-rear": "rear",
}
FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d+)_(?P<plate>.+?)_(?P<suffix>photo-(?:merged|front|rear))\.[^.]+$",
    re.IGNORECASE,
)


def log(message):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish a test MQTT weighbridge event using the latest local image set and latest scale DB row."
    )
    parser.add_argument("--publish", action="store_true", help="Actually publish to MQTT. Defaults to dry-run.")
    parser.add_argument("--plate", help="Override plate. Defaults to parsed plate from latest image filename.")
    parser.add_argument("--weight", type=float, help="Override weight in kg. Defaults to latest valid DB row.")
    parser.add_argument("--event-id", help="Stable edge event ID. Reuse to test backend deduplication.")
    parser.add_argument(
        "--transaction-type",
        default=DEFAULT_TRANSACTION_TYPE,
        help=f"MQTT transaction type. Default: {DEFAULT_TRANSACTION_TYPE}.",
    )
    parser.add_argument("--db", default=os.path.join(SERVICE_DIR, "scale_data.db"), help="SQLite DB path.")
    parser.add_argument("--capture-dir", default=CAPTURE_DIR, help="Image storage root.")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="Seconds to wait for MQTT connection.")
    return parser.parse_args()


def get_latest_scale_row(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Scale DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, timestamp, weight_kg, sign, decimal_pos, checksum_ok, status
            FROM weight_log
            WHERE checksum_ok = 1 AND status = 'STABLE' AND weight_kg > ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (WEIGHT_THRESHOLD,),
        ).fetchone()
        if row is None:
            row = conn.execute(
                """
                SELECT id, timestamp, weight_kg, sign, decimal_pos, checksum_ok, status
                FROM weight_log
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

    if row is None:
        raise RuntimeError(f"No rows found in weight_log: {db_path}")
    return dict(row)


def iter_images(root):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Capture directory not found: {root}")
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_image_name(path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    data = match.groupdict()
    group_key = f"{data['date']}_{data['time']}_{data['micro']}_{data['plate']}"
    try:
        captured_at = datetime.strptime(
            f"{data['date']}{data['time']}{data['micro'][:6].ljust(6, '0')}",
            "%Y%m%d%H%M%S%f",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        captured_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "group_key": group_key,
        "plate": data["plate"],
        "suffix": data["suffix"].lower(),
        "captured_at": captured_at,
    }


def url_for_image(path, capture_root):
    rel = path.resolve().relative_to(Path(capture_root).resolve())
    return "/storage/weighbridge/" + rel.as_posix()


def find_latest_image_set(capture_root):
    groups = {}
    fallback_latest = None

    for path in iter_images(capture_root):
        stat = path.stat()
        if fallback_latest is None or stat.st_mtime > fallback_latest.stat().st_mtime:
            fallback_latest = path

        parsed = parse_image_name(path)
        if not parsed:
            continue

        group = groups.setdefault(
            parsed["group_key"],
            {
                "plate": parsed["plate"],
                "captured_at": parsed["captured_at"],
                "latest_mtime": stat.st_mtime,
                "files": {},
            },
        )
        group["latest_mtime"] = max(group["latest_mtime"], stat.st_mtime)
        group["files"][parsed["suffix"]] = path

    if groups:
        return max(groups.values(), key=lambda item: item["latest_mtime"])

    if fallback_latest is None:
        raise RuntimeError(f"No images found under {capture_root}")

    return {
        "plate": None,
        "captured_at": datetime.fromtimestamp(fallback_latest.stat().st_mtime, tz=timezone.utc),
        "latest_mtime": fallback_latest.stat().st_mtime,
        "files": {"photo-merged": fallback_latest},
    }


def build_photos(image_set, capture_root):
    captured_at = image_set["captured_at"].strftime("%Y-%m-%dT%H:%M:%SZ")
    photos = []
    for suffix, photo_type in PHOTO_SUFFIXES.items():
        path = image_set["files"].get(suffix)
        if path is not None:
            photos.append({"url": url_for_image(path, capture_root), "type": photo_type, "captured_at": captured_at})
    return photos


def build_session_result(args, scale_row, image_set):
    plate = args.plate or image_set.get("plate") or "TEST-PLATE"
    weight = args.weight if args.weight is not None else float(scale_row["weight_kg"])
    timestamp = scale_row.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "start": timestamp,
        "end": timestamp,
        "duration_s": 0,
        "stable_weight": weight,
        "official_plate": plate,
        "official_plate_count": 1,
        "all_plates": {plate: 1},
        "photos": build_photos(image_set, args.capture_dir),
        "offline_event_id": args.event_id or str(uuid.uuid4()),
    }


def build_mqtt_payload(session_result, transaction_type):
    return build_weighbridge_payload(session_result, transaction_type)


def publish(session_result, transaction_type, connect_timeout):
    connected = False
    published = False
    publish_error = None

    def on_connect(client, userdata, flags, rc):
        nonlocal connected
        if rc == 0:
            connected = True
            log(f"Connected to MQTT broker {MQTT_HOST}:{MQTT_PORT}")
        else:
            log(f"MQTT connect failed rc={rc}")

    def on_publish(client, userdata, mid):
        nonlocal published
        published = True
        log(f"MQTT message published mid={mid}")

    client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311, clean_session=True)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    client.connect_async(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
    client.loop_start()
    deadline = time.time() + connect_timeout
    while not connected and time.time() < deadline:
        time.sleep(0.1)

    if not connected:
        client.loop_stop()
        client.disconnect()
        raise RuntimeError("MQTT connection timed out")

    try:
        payload = build_mqtt_payload(session_result, transaction_type)
        info = client.publish(MQTT_TOPIC, json.dumps(payload, ensure_ascii=False), qos=MQTT_QOS)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            publish_error = f"MQTT publish returned rc={info.rc}"
        else:
            log(
                f"Published to {MQTT_TOPIC}: plate={payload['vehicle_plate']}, "
                f"weight={payload['gross_weight_kg']:.3f} kg, type={transaction_type} (mid={info.mid})"
            )

        publish_deadline = time.time() + 5.0
        while not published and time.time() < publish_deadline:
            time.sleep(0.1)
    finally:
        client.disconnect()
        client.loop_stop()

    if publish_error:
        raise RuntimeError(publish_error)
    return published


def main():
    args = parse_args()
    scale_row = get_latest_scale_row(args.db)
    image_set = find_latest_image_set(args.capture_dir)
    session_result = build_session_result(args, scale_row, image_set)

    log(f"MQTT topic: {MQTT_TOPIC}")
    log(f"Scale row: {json.dumps(scale_row, ensure_ascii=False)}")
    log("Images:")
    for suffix, path in sorted(image_set["files"].items()):
        log(f"  {suffix}: {path}")
    log("Payload session_result:")
    log(json.dumps(session_result, ensure_ascii=False, indent=2))
    log("Payload MQTT:")
    log(json.dumps(build_mqtt_payload(session_result, args.transaction_type), ensure_ascii=False, indent=2))

    if not args.publish:
        log("Dry-run only. Re-run with --publish to send this MQTT event.")
        return 0

    ok = publish(session_result, args.transaction_type, args.connect_timeout)
    log(f"Publish result: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
