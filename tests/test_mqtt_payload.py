import os
import json
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("paho", Mock())
sys.modules.setdefault("paho.mqtt", Mock())
sys.modules.setdefault("paho.mqtt.client", Mock())

from mqtt_service import MqttService, build_weighbridge_payload, serialize_weighbridge_payload


class MqttPayloadTests(unittest.TestCase):
    def test_builds_canonical_identity_and_timestamp(self):
        with patch("mqtt_service.WEIGHBRIDGE_ID", "07ad7452-a66f-442b-9029-3d7abd2443f5"):
            payload = build_weighbridge_payload({
                "offline_event_id": "session-001",
                "timestamp": "2026-08-22T10:15:30.123+00:00",
                "official_plate": "15C-123.45",
                "stable_weight": 28500,
                "photos": [{"url": "/storage/photo.jpg"}],
            })

        self.assertEqual(payload["edge_event_id"], "session-001")
        self.assertNotIn("event_id", payload)
        self.assertEqual(payload["weighbridge_id"], "07ad7452-a66f-442b-9029-3d7abd2443f5")
        self.assertEqual(payload["timestamp"], "2026-08-22T10:15:30.123Z")
        self.assertEqual(payload["gross_weight_kg"], 28500)

    def test_hp1_guide_payload_with_real_tare(self):
        with patch("mqtt_service.WEIGHBRIDGE_ID", "07ad7452-a66f-442b-9029-3d7abd2443f5"):
            payload = build_weighbridge_payload({
                "offline_event_id": "hp1-20260824-110000-000001",
                "timestamp": "2026-08-24T11:00:00.000+07:00",
                "official_plate": "15C-240.84",
                "stable_weight": 28400,
                "tare_weight_kg": 8500,
            }, "gate_in")

        self.assertEqual(payload, {
            "edge_event_id": "hp1-20260824-110000-000001",
            "weighbridge_id": "07ad7452-a66f-442b-9029-3d7abd2443f5",
            "timestamp": "2026-08-24T04:00:00.000Z",
            "vehicle_plate": "15C-240.84",
            "transaction_type": "gate_in",
            "gross_weight_kg": 28400,
            "ocr_plate_read": "15C-240.84",
            "photos": [],
            "tare_weight_kg": 8500,
        })

    def test_hp2_guide_payload_with_real_tare(self):
        with patch("mqtt_service.WEIGHBRIDGE_ID", "a095b1a9-65a8-4802-a9dc-a617412e96f1"):
            payload = build_weighbridge_payload({
                "offline_event_id": "hp2-20260824-110500-000001",
                "timestamp": "2026-08-24T11:05:00.000+07:00",
                "official_plate": "15C-326.77",
                "stable_weight": 28500,
                "tare_weight_kg": 9000,
            }, "gate_out")

        self.assertEqual(payload["weighbridge_id"], "a095b1a9-65a8-4802-a9dc-a617412e96f1")
        self.assertEqual(payload["transaction_type"], "gate_out")
        self.assertEqual(payload["tare_weight_kg"], 9000)

    def test_builds_weight_event_when_plate_is_unreadable(self):
        with patch("mqtt_service.WEIGHBRIDGE_ID", "100ecc11-dbcb-4c23-8e89-d41ccefcda37"):
            payload = build_weighbridge_payload({
                "offline_event_id": "session-no-plate",
                "timestamp": "2026-08-24T15:20:37.532+00:00",
                "official_plate": "UNKNOWN",
                "ocr_plate_read": None,
                "stable_weight": 39220,
                "metadata": {
                    "plate_status": "unreadable",
                    "lpr_classification": "no_plate_detection",
                },
            })

        self.assertEqual(payload["vehicle_plate"], "UNKNOWN")
        self.assertIsNone(payload["ocr_plate_read"])
        self.assertEqual(payload["gross_weight_kg"], 39220)
        self.assertEqual(payload["metadata"]["plate_status"], "unreadable")

    def test_serializes_strict_raw_utf8_json(self):
        payload = {
            "weighbridge_id": "07ad7452-a66f-442b-9029-3d7abd2443f5",
            "edge_event_id": "event-001",
            "vehicle_plate": "15C-240.84",
            "gross_weight_kg": 28400,
            "remarks": "Cân tự động",
        }

        serialized = serialize_weighbridge_payload(payload)

        self.assertEqual(json.loads(serialized), payload)
        self.assertIn("Cân tự động", serialized)
        self.assertNotIn("NaN", serialized)

    def test_rejects_non_json_number(self):
        with self.assertRaises(ValueError):
            serialize_weighbridge_payload({"gross_weight_kg": float("nan")})

    def test_publish_explicitly_disables_retain(self):
        info = Mock(rc=0)
        info.is_published.return_value = True
        service = MqttService.__new__(MqttService)
        service._client = Mock()
        service._client.publish.return_value = info
        service._log = Mock()

        with patch("mqtt_service.mqtt.MQTT_ERR_SUCCESS", 0), patch(
            "mqtt_service.MQTT_TOPIC", "smartport/weighbridge/test/events",
        ), patch("mqtt_service.MQTT_QOS", 1):
            result = service.publish_weighbridge_event({
                "offline_event_id": "event-002",
                "timestamp": "2026-08-24T11:00:00.000+07:00",
                "official_plate": "15C-240.84",
                "stable_weight": 28400,
            })

        self.assertTrue(result)
        _, kwargs = service._client.publish.call_args
        self.assertEqual(kwargs, {"qos": 1, "retain": False})
        self.assertEqual(json.loads(service._client.publish.call_args.args[1])["edge_event_id"], "event-002")

    def test_legacy_local_end_timestamp_is_normalized(self):
        payload = build_weighbridge_payload({
            "offline_event_id": "session-002",
            "end": "2026-08-22 17:15:30",
            "official_plate": "15C-123.45",
            "stable_weight": 28500,
        })

        self.assertRegex(payload["timestamp"], r"^2026-08-22T\d{2}:15:30\.000Z$")

    def test_rejects_nonpositive_weight(self):
        with self.assertRaises(ValueError):
            build_weighbridge_payload({
                "offline_event_id": "session-003",
                "official_plate": "15C-123.45",
                "stable_weight": 0,
            })


if __name__ == "__main__":
    unittest.main()
