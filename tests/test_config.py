import importlib.util
import os
import shutil
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConfigTests(unittest.TestCase):
    def load_config(self, local_config):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        config_path = os.path.join(root.name, "config.py")
        shutil.copyfile(os.path.join(ROOT, "config.py"), config_path)
        with open(os.path.join(root.name, "config.local.py"), "w", encoding="utf-8") as handle:
            handle.write(local_config)
        spec = importlib.util.spec_from_file_location("test_host_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_hp01_identity_derives_mqtt_fields_after_local_override(self):
        config = self.load_config(
            'WEIGHBRIDGE_ID = "100ecc11-dbcb-4c23-8e89-d41ccefcda37"\n'
        )

        self.assertEqual(config.MQTT_WEIGHBRIDGE_TOPIC_ID, config.WEIGHBRIDGE_ID)
        self.assertEqual(
            config.MQTT_CLIENT_ID,
            "smartport-weighbridge-100ecc11-dbcb-4c23-8e89-d41ccefcda37",
        )
        self.assertTrue(config.MQTT_TOPIC.endswith(
            "/weighbridge/100ecc11-dbcb-4c23-8e89-d41ccefcda37/events"
        ))
        self.assertEqual(config.DEFAULT_TRANSACTION_TYPE, "gate_in")
        self.assertFalse(config.SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT)
        config.validate_runtime_config()

    def test_hp02_identity_derives_mqtt_fields_after_local_override(self):
        config = self.load_config(
            'WEIGHBRIDGE_ID = "9aa29a10-6605-47dd-9460-970d66c3d1c3"\n'
        )

        self.assertEqual(config.MQTT_WEIGHBRIDGE_TOPIC_ID, config.WEIGHBRIDGE_ID)
        self.assertEqual(
            config.MQTT_CLIENT_ID,
            "smartport-weighbridge-9aa29a10-6605-47dd-9460-970d66c3d1c3",
        )
        self.assertTrue(config.MQTT_TOPIC.endswith(
            "/weighbridge/9aa29a10-6605-47dd-9460-970d66c3d1c3/events"
        ))
        self.assertEqual(config.DEFAULT_TRANSACTION_TYPE, "gate_out")
        self.assertTrue(config.SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT)
        config.validate_runtime_config()

    def test_explicit_local_mqtt_identity_fields_win(self):
        config = self.load_config(
            'WEIGHBRIDGE_ID = "bridge-id"\n'
            'MQTT_WEIGHBRIDGE_TOPIC_ID = "bridge-id"\n'
            'MQTT_CLIENT_ID = "custom-client-bridge-id"\n'
            'MQTT_TOPIC = "custom/bridge-id/topic"\n'
        )

        self.assertEqual(config.MQTT_WEIGHBRIDGE_TOPIC_ID, "bridge-id")
        self.assertEqual(config.MQTT_CLIENT_ID, "custom-client-bridge-id")
        self.assertEqual(config.MQTT_TOPIC, "custom/bridge-id/topic")
        config.validate_runtime_config()

    def test_topic_identity_must_match_payload_identity(self):
        config = self.load_config(
            'WEIGHBRIDGE_ID = "bridge-id"\n'
            'MQTT_WEIGHBRIDGE_TOPIC_ID = "truncated"\n'
        )

        with self.assertRaisesRegex(ValueError, "must equal WEIGHBRIDGE_ID"):
            config.validate_runtime_config()

    def test_known_host_rejects_conflicting_policy(self):
        config = self.load_config(
            'WEIGHBRIDGE_ID = "100ecc11-dbcb-4c23-8e89-d41ccefcda37"\n'
            'DEFAULT_TRANSACTION_TYPE = "gate_out"\n'
            'SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT = True\n'
        )

        with self.assertRaisesRegex(ValueError, "canonical host policy"):
            config.validate_runtime_config()

    def test_disabled_mqtt_still_validates_session_policy(self):
        config = self.load_config(
            'WEIGHBRIDGE_ID = "100ecc11-dbcb-4c23-8e89-d41ccefcda37"\n'
            'MQTT_ENABLED = False\n'
            'SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT = True\n'
        )

        with self.assertRaisesRegex(ValueError, "canonical host policy"):
            config.validate_runtime_config()

    def test_enabled_mqtt_requires_valid_identity_and_credentials(self):
        config = self.load_config('MQTT_PASSWORD = ""\nMQTT_PORT = 0\n')

        with self.assertRaisesRegex(ValueError, "MQTT_PASSWORD.*MQTT_PORT"):
            config.validate_runtime_config()

    def test_minio_fields_are_required(self):
        config = self.load_config('MINIO_SECRET_KEY = ""\n')

        with self.assertRaisesRegex(ValueError, "MINIO_SECRET_KEY"):
            config.validate_runtime_config()


if __name__ == "__main__":
    unittest.main()
