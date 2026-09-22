import base64
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock
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
        self.assertEqual(authorization, "Basic " + base64.b64encode(b"admin:secret").decode())

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


if __name__ == "__main__":
    unittest.main()
