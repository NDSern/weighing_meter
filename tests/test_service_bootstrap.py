"""Smoke tests that the production entry point can be imported and constructed.

These guard against config/handler drift that module-level unit tests miss
because they never import the runtime bootstrap.
"""

import contextlib
import sys
import unittest
from unittest import mock

for _name in ("cv2", "minio", "minio.error", "serial", "rknnlite", "rknnlite.api",
              "paho", "paho.mqtt", "paho.mqtt.client"):
    sys.modules.setdefault(_name, mock.MagicMock())


class BootstrapSmokeTests(unittest.TestCase):
    def test_entry_points_import(self):
        from services.runtime import bootstrap

        self.assertTrue(callable(bootstrap.construct_service))
        self.assertTrue(callable(bootstrap.shutdown_service))
        self.assertTrue(callable(bootstrap.configure_module_logging))

    def test_service_resources_fields(self):
        from services.runtime import bootstrap

        res = bootstrap.ServiceResources()
        for attr in ("session_manager", "reader", "mqtt_svc", "frame_spool", "deferred_lpr"):
            self.assertTrue(hasattr(res, attr), attr)

    def test_lpr_bundle_paths_leave_fallback_disabled_by_default(self):
        from services.runtime import bootstrap

        self.assertNotIn("fallback_detector", bootstrap._lpr_bundle_paths(""))

    def test_lpr_bundle_paths_include_configured_fallback(self):
        from services.runtime import bootstrap

        self.assertEqual(
            bootstrap._lpr_bundle_paths("/models/fallback.rknn")["fallback_detector"],
            "/models/fallback.rknn",
        )


class MaskUrlSecretTests(unittest.TestCase):
    def _masks(self):
        from services.capture import frame_source
        from services.runtime import bootstrap

        return (("bootstrap", bootstrap.mask_url_secret), ("frame_source", frame_source.mask_url_secret))

    def test_masks_password_containing_at_sign(self):
        url = "rtsp://admin:p@ss@192.168.1.181:554/live/0/MAIN"
        for name, mask in self._masks():
            masked = mask(url)
            self.assertNotIn("ss@192", masked, name)
            self.assertNotIn("p@ss", masked, name)
            self.assertEqual(masked, "rtsp://admin:***@192.168.1.181:554/live/0/MAIN", name)

    def test_masks_simple_password(self):
        url = "rtsp://admin:123456@192.168.1.20:554/ch01/0"
        for name, mask in self._masks():
            self.assertEqual(mask(url), "rtsp://admin:***@192.168.1.20:554/ch01/0", name)

    def test_leaves_url_without_credentials_unchanged(self):
        url = "rtsp://192.168.1.20:554/ch01/0"
        for name, mask in self._masks():
            self.assertEqual(mask(url), url, name)


class PartialStartupShutdownTests(unittest.TestCase):
    """M12: a resource flagged started before start() must always be stopped."""

    def _construction_patchers(self):
        return [
            mock.patch("services.runtime.bootstrap.MQTT_ENABLED", True),
            mock.patch("services.runtime.bootstrap.verify_lpr_bundle", return_value={}),
            mock.patch("services.runtime.bootstrap.validate_runtime_config"),
            mock.patch("services.runtime.bootstrap.os.makedirs"),
            mock.patch("services.runtime.RknnModelSet"),
            mock.patch("services.pipeline.license_plate_recognition.detect_axis_plate_regions"),
            mock.patch("services.pipeline.license_plate_recognition.detect_plate_regions"),
            mock.patch("services.pipeline.license_plate_recognition.load_lpr_charset", return_value=[]),
            mock.patch("services.pipeline.license_plate_recognition.recognize_plate_regions"),
            mock.patch("services.pipeline.license_plate_recognition.validate_lpr_runtime"),
            mock.patch("services.tracking.PlateTracker"),
            mock.patch("services.capture.FrameGrabber"),
            mock.patch("services.capture.CameraGrabber"),
            mock.patch("services.capture.DetectCoordinator"),
            mock.patch("services.capture.camera_light_controller.CameraLightController"),
            mock.patch("services.capture.session_frame_spool.SessionFrameSpool"),
            mock.patch("services.pipeline.deferred_lpr_worker.DeferredLprWorker"),
            mock.patch("services.storage.image_save_worker.ImageSaveWorker"),
            mock.patch("services.storage.publish_outbox.PublishOutbox"),
            mock.patch("services.storage.retention_cleaner.DiagnosticArchiveCleaner"),
            mock.patch("services.storage.retention_cleaner.ImageRetentionCleaner"),
            mock.patch("services.storage.retention_cleaner.StorageMaintenance"),
            mock.patch("services.storage.retention_cleaner.VerifiedMinioCacheCleaner"),
            mock.patch("services.review.duplicate_review.DuplicateReviewer"),
            mock.patch("services.session.SessionManager"),
            mock.patch("d2008_scale_reader.D2008Reader"),
            mock.patch("mqtt_service.MqttService"),
        ]

    def test_started_flag_is_set_before_start_so_shutdown_can_stop(self):
        from services.runtime import bootstrap

        cases = (
            ("mqtt_service.MqttService", "start", "mqtt_started"),
            ("services.storage.image_save_worker.ImageSaveWorker", "start_upload_worker", "image_worker_started"),
            ("services.storage.publish_outbox.PublishOutbox", "start", "outbox_started"),
        )
        for target, method, flag in cases:
            with self.subTest(flag=flag):
                res = bootstrap.ServiceResources()
                with contextlib.ExitStack() as stack:
                    for patcher in self._construction_patchers():
                        stack.enter_context(patcher)
                    owner = stack.enter_context(mock.patch(target))
                    getattr(owner, method).side_effect = RuntimeError("boom")
                    getattr(owner.return_value, method).side_effect = RuntimeError("boom")
                    with self.assertRaises(RuntimeError):
                        bootstrap.construct_service(mock.Mock(), res)
                self.assertTrue(getattr(res, flag), flag)

    def test_shutdown_stops_a_half_started_resource(self):
        from services.runtime import bootstrap

        res = bootstrap.ServiceResources()
        res.mqtt_svc = mock.Mock()
        res.mqtt_started = True
        res.outbox_started = True
        res.image_worker_started = True
        log = mock.Mock()
        with mock.patch("services.storage.publish_outbox.PublishOutbox.stop") as outbox_stop, mock.patch(
            "services.storage.image_save_worker.ImageSaveWorker.stop"
        ) as worker_stop, mock.patch(
            "services.storage.image_save_worker.ImageSaveWorker.wait_for_pending", return_value=True
        ):
            bootstrap.shutdown_service(res, log)
        outbox_stop.assert_called_once()
        worker_stop.assert_called_once()
        res.mqtt_svc.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
