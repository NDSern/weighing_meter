"""Control camera white lights while a nighttime LPR session is active."""

import base64
from datetime import time as clock_time
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from xml.etree import ElementTree


class CameraLightController:
    """Use each camera's vendor XML API without changing unrelated settings."""

    def __init__(self, rtsp_urls, now_fn, open_fn=None):
        self._targets = [self._parse_target(url) for url in rtsp_urls]
        self._now_fn = now_fn
        self._open = open_fn or build_opener(ProxyHandler({})).open
        self._on_targets = set()

    @staticmethod
    def _parse_target(rtsp_url):
        parsed = urlsplit(rtsp_url)
        if not parsed.hostname:
            raise ValueError("Camera RTSP URL must include a host")
        return parsed.hostname, unquote(parsed.username or ""), unquote(parsed.password or "")

    @staticmethod
    def _is_night(now):
        current = now.timetz().replace(tzinfo=None)
        return current >= clock_time(19, 0) or current <= clock_time(6, 30)

    @staticmethod
    def _element(root, name):
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == name:
                return element
        raise ValueError(f"Camera light response is missing {name}")

    def _request(self, target, method, body=None):
        host, username, password = target
        request = Request(
            f"http://{host}/Images/1/IrCutFilter",
            data=body,
            method=method,
            headers={
                "Authorization": "Basic " + base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode(),
                "Content-Type": "application/xml",
            },
        )
        response = self._open(request, timeout=3.0)
        return response.read()

    def _set_brightness(self, target, brightness):
        current = self._request(target, "GET")
        root = ElementTree.fromstring(current)
        self._element(root, "VarWhiteControlMode").text = "custom"
        self._element(root, "VarWhiteWorkMode").text = "timing"
        self._element(root, "VarWhiteBrightness").text = str(brightness)
        response = self._request(
            target,
            "PUT",
            ElementTree.tostring(root, encoding="utf-8", xml_declaration=True),
        )
        result = ElementTree.fromstring(response)
        status = next(
            (element.text for element in result.iter()
             if element.tag.rsplit("}", 1)[-1] == "statusCode"),
            None,
        )
        if status not in (None, "0"):
            raise RuntimeError(f"camera rejected white-light update statusCode={status}")

    def set_lpr_active(self, active, log_fn):
        if active and not self._is_night(self._now_fn()):
            return
        targets = self._targets if active else [
            target for target in self._targets if target in self._on_targets
        ]
        brightness = 100 if active else 0
        for target in targets:
            if active and target in self._on_targets:
                continue
            host = target[0]
            try:
                self._set_brightness(target, brightness)
            except Exception as exc:
                log_fn("ERROR", f"LPR light control failed host={host}: {exc}")
                continue
            if active:
                self._on_targets.add(target)
            else:
                self._on_targets.discard(target)
            log_fn("EVENT", f"LPR light {'on' if active else 'off'} host={host}")
