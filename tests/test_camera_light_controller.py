import base64
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request
from xml.etree import ElementTree

sys.modules.setdefault("cv2", Mock())

from services.capture.camera_light_controller import CameraLightController


CURRENT_XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<IrCutFillter xmlns="http://www.zwcloud.wang/ver10/XMLSchema">
  <Mode>variablewhitelight</Mode>
  <VarWhiteControlMode>auto</VarWhiteControlMode>
  <VarWhiteWorkMode>threshold</VarWhiteWorkMode>
  <VarWhiteBrightness>0</VarWhiteBrightness>
  <Unchanged>preserve-me</Unchanged>
</IrCutFillter>'''
OK_XML = b"<Response><statusCode>0</statusCode></Response>"


class CameraLightControllerTests(unittest.TestCase):
    def make_controller(self, now):
        self.requests = []

        def open_fn(request, timeout):
            self.requests.append(request)
            response = Mock()
            response.read.return_value = CURRENT_XML if request.get_method() == "GET" else OK_XML
            return response

        return CameraLightController(
            ["rtsp://admin:secret@192.168.1.181:554/ch01/0"],
            lambda: now,
            open_fn,
            api_password="camera-api-secret",
        )

    def put_body(self):
        return self.requests[-1].data.decode()

    def put_values(self):
        root = ElementTree.fromstring(self.requests[-1].data)
        return {
            element.tag.rsplit("}", 1)[-1]: element.text
            for element in root.iter()
        }

    def test_night_start_turns_light_on_and_preserves_xml(self):
        controller = self.make_controller(datetime(2026, 9, 22, 19, 0))
        log = Mock()

        controller.set_lpr_active(True, log)

        self.assertEqual([request.get_method() for request in self.requests], ["GET", "PUT"])
        values = self.put_values()
        self.assertEqual(values["VarWhiteControlMode"], "custom")
        self.assertEqual(values["VarWhiteWorkMode"], "timing")
        self.assertEqual(values["VarWhiteBrightness"], "100")
        self.assertEqual(values["Unchanged"], "preserve-me")
        authorization = self.requests[0].get_header("Authorization")
        self.assertEqual(
            authorization,
            "Basic " + base64.b64encode(b"admin:camera-api-secret").decode(),
        )

    def test_url_credentials_are_used_when_api_password_missing(self):
        self.requests = []

        def open_fn(request, timeout):
            self.requests.append(request)
            response = Mock()
            response.read.return_value = CURRENT_XML if request.get_method() == "GET" else OK_XML
            return response

        controller = CameraLightController(
            ["rtsp://admin:url-secret@192.168.1.181:554/ch01/0"],
            lambda: datetime(2026, 9, 22, 20, 0),
            open_fn,
        )

        controller.set_lpr_active(True, Mock())

        authorization = self.requests[0].get_header("Authorization")
        self.assertEqual(
            authorization,
            "Basic " + base64.b64encode(b"admin:url-secret").decode(),
        )

    def test_day_start_does_not_change_light(self):
        controller = self.make_controller(datetime(2026, 9, 22, 12, 0))

        controller.set_lpr_active(True, Mock())

        self.assertEqual(self.requests, [])

    def test_end_turns_off_light_started_at_night_even_after_schedule(self):
        controller = self.make_controller(datetime(2026, 9, 22, 6, 29))
        log = Mock()
        controller.set_lpr_active(True, log)
        controller._now_fn = lambda: datetime(2026, 9, 22, 6, 31)

        controller.set_lpr_active(False, log)

        self.assertEqual([request.get_method() for request in self.requests], ["GET", "PUT", "GET", "PUT"])
        self.assertEqual(self.put_values()["VarWhiteBrightness"], "0")

    def test_camera_failure_does_not_raise_or_mark_light_on(self):
        def open_fn(_request, timeout):
            raise OSError("unreachable")

        controller = CameraLightController(
            ["rtsp://admin:secret@192.168.1.181:554/ch01/0"],
            lambda: datetime(2026, 9, 22, 20, 0),
            open_fn,
        )
        log = Mock()

        controller.set_lpr_active(True, log)

        self.assertEqual(controller._on_targets, set())
        log.assert_called_once()

    def test_unsupported_camera_is_logged_once_and_skipped(self):
        request = Request("http://192.168.1.179/Images/1/IrCutFilter")

        def open_fn(_request, timeout):
            raise HTTPError(request.full_url, 404, "unsupported", None, None)

        controller = CameraLightController(
            ["rtsp://admin:secret@192.168.1.179:554/ch01/0"],
            lambda: datetime(2026, 9, 22, 20, 0),
            open_fn,
        )
        log = Mock()

        controller.set_lpr_active(True, log)
        controller.set_lpr_active(True, log)

        self.assertEqual(len(log.call_args_list), 1)
        self.assertIn("unsupported", log.call_args.args[1])

    def test_http_failures_back_off_and_server_error_remains_retryable(self):
        for status in (401, 403, 500):
            with self.subTest(status=status):
                clock = [0.0]
                opener = Mock(side_effect=lambda *_a, **_kw: (_ for _ in ()).throw(
                    HTTPError("http://camera/", status, "secret must not appear", None, None)))
                controller = CameraLightController(
                    ["rtsp://admin:secret@192.168.1.181:554/ch01/0"],
                    lambda: datetime(2026, 9, 22, 20, 0), opener,
                    monotonic_fn=lambda: clock[0],
                )
                log = Mock()
                for delay in (60, 120, 240, 480, 900, 900):
                    before = opener.call_count
                    controller.set_lpr_active(True, log)
                    self.assertEqual(opener.call_count, before + 1)
                    clock[0] += delay - 0.1
                    controller.set_lpr_active(True, log)
                    self.assertEqual(opener.call_count, before + 1)
                    clock[0] += 0.1
                self.assertEqual(controller._unsupported_targets, set())
                self.assertNotIn("secret", str(log.call_args_list))

    def test_failed_off_retains_state_and_success_resets_backoff(self):
        controller = self.make_controller(datetime(2026, 9, 22, 20, 0))
        clock = [0.0]
        controller._monotonic = lambda: clock[0]
        log = Mock()
        controller.set_lpr_active(True, log)
        successful_open = controller._open
        controller._open = Mock(side_effect=TimeoutError("secret-url"))
        controller.set_lpr_active(False, log)
        self.assertTrue(controller._on_targets)
        controller.set_lpr_active(False, log)
        self.assertEqual(controller._open.call_count, 1)
        clock[0] = 60.0
        controller._open = successful_open
        controller.set_lpr_active(False, log)
        self.assertEqual(controller._on_targets, set())
        self.assertEqual(controller._retry_state, {})
        self.assertTrue(any("recovered" in c.args[1] for c in log.call_args_list))
        self.assertNotIn("secret-url", str(log.call_args_list))

    def test_failed_target_does_not_block_other_camera(self):
        def opener(request, timeout):
            if request.host == "192.168.1.181":
                raise TimeoutError()
            response = Mock()
            response.read.return_value = CURRENT_XML if request.get_method() == "GET" else OK_XML
            return response

        controller = CameraLightController(
            ["rtsp://admin:secret@192.168.1.181/x", "rtsp://admin:secret@192.168.1.177/x"],
            lambda: datetime(2026, 9, 22, 20, 0), opener,
        )
        controller.set_lpr_active(True, Mock())
        self.assertEqual({t[0] for t in controller._on_targets}, {"192.168.1.177"})


if __name__ == "__main__":
    unittest.main()
