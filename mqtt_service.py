"""
MQTT Service — publishes weighbridge events to SmartPort MQTT broker.

Follows format 4.8 (Weighbridge Events) from mqtt-integration-guide.md.
Topic: smartport/weighbridge/{weighbridge_id}/events

Usage:
    Imported by weighing_service.py to publish session results via MQTT.
    Can also run standalone for testing:
        python3 mqtt_service.py
"""

import json
import time
import uuid
import threading
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import (
    DEFAULT_TRANSACTION_TYPE,
    MQTT_CLIENT_ID,
    MQTT_HOST,
    MQTT_KEEPALIVE,
    MQTT_PASSWORD,
    MQTT_PORT,
    MQTT_QOS,
    MQTT_TOPIC,
    MQTT_USERNAME,
    WEIGHBRIDGE_ID,
)


def _log(level: str, msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
    print(f"{ts} [MQTT/{level:<7}] {msg}", flush=True)


class MqttService:
    """Manages MQTT connection and publishes weighbridge events."""

    def __init__(self, on_log=None):
        self._log = on_log or _log
        self._connected = False
        self._lock = threading.Lock()

        self._client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )
        self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            self._log("INFO", f"Connected to MQTT broker {MQTT_HOST}:{MQTT_PORT}")
        else:
            self._connected = False
            self._log("ERROR", f"MQTT connect failed (rc={rc})")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            self._log("WARNING", f"MQTT disconnected unexpectedly (rc={rc}), will reconnect...")

    def _on_publish(self, client, userdata, mid):
        self._log("INFO", f"MQTT message published (mid={mid})")

    def start(self):
        """Connect to broker in background with auto-reconnect."""
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        try:
            self._client.connect_async(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
            self._client.loop_start()
            self._log("INFO", f"MQTT client starting — broker {MQTT_HOST}:{MQTT_PORT}")
        except Exception as exc:
            self._log("ERROR", f"MQTT connect error: {exc}")

    def stop(self):
        """Disconnect and stop the network loop."""
        self._client.loop_stop()
        self._client.disconnect()
        self._log("INFO", "MQTT client stopped.")

    @property
    def connected(self) -> bool:
        return self._connected

    def publish_weighbridge_event(self, session_result: dict,
                                  transaction_type: str = DEFAULT_TRANSACTION_TYPE,
                                  wait_for_ack: bool = False,
                                  timeout: float = 10.0) -> bool:
        """Publish a weighbridge event from a finalized WeighingSession result.

        Args:
            session_result: dict from WeighingSession.finalize() with keys:
                start, end, duration_s, stable_weight, official_plate,
                official_plate_count, all_plates
            transaction_type: one of gate_in, gate_out, vgm, reweigh, spot_check

        Returns:
            True if publish was queued successfully, or acknowledged when wait_for_ack=True.
        """
        plate = session_result.get("official_plate", "none")
        weight = session_result.get("stable_weight")

        if plate == "none" or weight is None or weight <= 0:
            self._log("WARNING", f"Skipping MQTT publish — plate={plate}, weight={weight}")
            return False

        payload = {
            "event_id": session_result.get("offline_event_id"),
            "weighbridge_id": WEIGHBRIDGE_ID,
            "vehicle_plate": plate,
            "transaction_type": transaction_type,
            "gross_weight_kg": round(weight, 3),
            "ocr_plate_read": plate,
            "photos": session_result.get("photos", []),
        }

        payload_json = json.dumps(payload, ensure_ascii=False)

        try:
            info = self._client.publish(MQTT_TOPIC, payload_json, qos=MQTT_QOS)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self._log("ERROR", f"MQTT publish rejected rc={info.rc}: plate={plate}, weight={weight:.3f} kg")
                return False
            if wait_for_ack:
                info.wait_for_publish(timeout=timeout)
                if not info.is_published():
                    self._log("ERROR", f"MQTT publish ack timeout: plate={plate}, weight={weight:.3f} kg, mid={info.mid}")
                    return False
            self._log("INFO", f"Published to {MQTT_TOPIC}: "
                               f"plate={plate}, weight={weight:.3f} kg, "
                               f"type={transaction_type} (mid={info.mid})")
            return True
        except Exception as exc:
            self._log("ERROR", f"MQTT publish failed: {exc}")
            return False


# ── Standalone test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("MQTT Service — standalone test")
    print(f"Broker: {MQTT_HOST}:{MQTT_PORT}")
    print(f"Topic:  {MQTT_TOPIC}")
    print()

    svc = MqttService()
    svc.start()

    # Wait for connection
    for _ in range(10):
        if svc.connected:
            break
        time.sleep(0.5)

    if not svc.connected:
        print("ERROR: Could not connect to MQTT broker.")
        svc.stop()
        exit(1)

    # Send a test event
    test_result = {
        "start": "2026-03-26T08:00:00",
        "end": "2026-03-26T08:01:30",
        "duration_s": 90.0,
        "stable_weight": 28500.0,
        "official_plate": "51A-12345",
        "official_plate_count": 3,
        "all_plates": {"51A-12345": 3},
    }

    ok = svc.publish_weighbridge_event(test_result, transaction_type="gate_in")
    print(f"\nPublish result: {'OK' if ok else 'FAILED'}")

    time.sleep(2)
    svc.stop()
