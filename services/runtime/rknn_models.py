"""Rollback-safe ownership for RKNN model runtimes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RknnHandles:
    cam1_detector: object
    cam1_ocr: object
    cam3_detector: object
    cam3_ocr: object
    vehicle: object | None


class RknnModelSet:
    def __init__(self, handles, release_order, log_fn=None):
        self.handles = handles
        self._release_order = release_order
        self._released = set()
        self._log_fn = log_fn

    @classmethod
    def open(
        cls,
        detector_path,
        ocr_path,
        vehicle_path,
        vehicle_enabled,
        rknn_class,
        log_fn=None,
    ):
        specs = [
            ("cam1_detector", detector_path, "lpr_detector_cam1", rknn_class.NPU_CORE_0),
            ("cam1_ocr", ocr_path, "lpr_recognizer_cam1", rknn_class.NPU_CORE_0),
            ("cam3_detector", detector_path, "lpr_detector_cam3", rknn_class.NPU_CORE_1),
            ("cam3_ocr", ocr_path, "lpr_recognizer_cam3", rknn_class.NPU_CORE_1),
        ]
        if vehicle_enabled:
            specs.append(("vehicle", vehicle_path, "YOLO26_vehicle", rknn_class.NPU_CORE_2))

        opened = {}
        release_order = []
        try:
            for key, path, name, core_mask in specs:
                opened[key] = cls._open_one(rknn_class, path, name, core_mask, log_fn)
                release_order.append((key, opened[key]))
        except BaseException:
            cls._release_many(reversed(release_order), log_fn)
            raise

        handles = RknnHandles(
            cam1_detector=opened["cam1_detector"],
            cam1_ocr=opened["cam1_ocr"],
            cam3_detector=opened["cam3_detector"],
            cam3_ocr=opened["cam3_ocr"],
            vehicle=opened.get("vehicle"),
        )
        return cls(handles, release_order, log_fn)

    @staticmethod
    def _open_one(rknn_class, path, name, core_mask, log_fn):
        if log_fn:
            log_fn("INFO", f"Loading RKNN {name}: {path}")
        model = rknn_class()
        try:
            ret = model.load_rknn(path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model {path} (ret={ret})")
            ret = model.init_runtime(core_mask=core_mask)
            if ret != 0:
                raise RuntimeError(f"Failed to init RKNN runtime for {path} (ret={ret})")
        except BaseException:
            try:
                model.release()
            except Exception:
                pass
            raise
        if log_fn:
            log_fn("INFO", f"RKNN {name} ready (core_mask={core_mask}).")
        return model

    @staticmethod
    def _release_many(resources, log_fn):
        errors = []
        for name, model in resources:
            try:
                model.release()
            except Exception as exc:
                errors.append((name, exc))
                if log_fn:
                    log_fn("ERROR", f"RKNN release failed model={name}: {exc}")
        return errors

    def close(self, release_lpr=True, release_vehicle=True):
        errors = []
        for name, model in reversed(self._release_order):
            is_vehicle = name == "vehicle"
            if name in self._released:
                continue
            if (is_vehicle and not release_vehicle) or (not is_vehicle and not release_lpr):
                continue
            try:
                model.release()
                self._released.add(name)
            except Exception as exc:
                errors.append((name, exc))
                if self._log_fn:
                    self._log_fn("ERROR", f"RKNN release failed model={name}: {exc}")
        return errors
