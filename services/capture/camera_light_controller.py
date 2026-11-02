"""Control camera white lights while a nighttime LPR session is active."""

import base64
import threading
import time
from datetime import time as clock_time
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from xml.etree import ElementTree


class CameraLightController:
    """Use each camera's vendor XML API without changing unrelated settings."""

    def __init__(self, rtsp_urls, now_fn, open_fn=None, api_password="", monotonic_fn=None):
        self._targets = [self._parse_target(url, api_password) for url in rtsp_urls]
        self._now_fn = now_fn
        self._monotonic = monotonic_fn or time.monotonic
        self._open = open_fn or build_opener(ProxyHandler({})).open
        self._lock = threading.Lock()
        self._on_targets = set()
        self._unsupported_targets = set()
        self._retry_state = {}

    @staticmethod
    def _parse_target(rtsp_url, api_password):
        parsed = urlsplit(rtsp_url)
        if not parsed.hostname:
            raise ValueError("Camera RTSP URL must include a host")
        # Fall back to the RTSP URL credential so the nightly white-light API
        # does not silently authenticate with an empty password.
        password = api_password or unquote(parsed.password or "")
        return parsed.hostname, unquote(parsed.username or ""), password

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
        response = self._open(request, timeout=1.0)
        try:
            return response.read()
        finally:
            response.close()

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
        with self._lock:
            on_targets = set(self._on_targets)
        targets = self._targets if active else [
            target for target in self._targets if target in on_targets
        ]
        brightness = 100 if active else 0
        for target in targets:
            with self._lock:
                if target in self._unsupported_targets or (active and target in self._on_targets):
                    continue
                retry_at, _delay = self._retry_state.get(target, (0.0, 0.0))
                if self._monotonic() < retry_at:
                    continue
            host = target[0]
            try:
                self._set_brightness(target, brightness)
            except HTTPError as exc:
                exc.close()
                if exc.code == 404:
                    with self._lock:
                        self._unsupported_targets.add(target)
                    log_fn(
                        "WARNING",
                        f"LPR light control unsupported host={host} status={exc.code}",
                    )
                    continue
                self._record_failure(target, f"HTTP {exc.code}", log_fn)
                continue
            except Exception as exc:
                # Exception text from HTTP clients can contain credentials.
                self._record_failure(target, type(exc).__name__, log_fn)
                continue
            with self._lock:
                recovered = self._retry_state.pop(target, None) is not None
                if active:
                    self._on_targets.add(target)
                else:
                    self._on_targets.discard(target)
            if recovered:
                log_fn("EVENT", f"LPR light control recovered host={host}")
            log_fn("EVENT", f"LPR light {'on' if active else 'off'} host={host}")

    def _record_failure(self, target, reason, log_fn):
        with self._lock:
            _retry_at, previous_delay = self._retry_state.get(target, (0.0, 0.0))
            delay = min(900.0, previous_delay * 2) if previous_delay else 60.0
            self._retry_state[target] = (self._monotonic() + delay, delay)
        log_fn("ERROR", f"LPR light control failed host={target[0]}: {reason}; retry_in={delay:.0f}s")
