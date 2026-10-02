"""Benchmark explicit LPR RKNN variants without changing production config."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(paths):
    images = []
    for value in paths:
        path = Path(value)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        elif path.is_dir():
            images.extend(
                candidate for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            raise ValueError(f"Image path not found or unsupported: {path}")
    return list(dict.fromkeys(images))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def open_rknn(rknn_class, path, core_mask):
    model = rknn_class()
    try:
        result = model.load_rknn(path)
        if result != 0:
            raise RuntimeError(f"load_rknn failed for {path}: {result}")
        result = model.init_runtime(core_mask=core_mask)
        if result != 0:
            raise RuntimeError(f"init_runtime failed for {path}: {result}")
        return model
    except BaseException:
        try:
            model.release()
        except Exception:
            pass
        raise


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_variant(variant, images, labels, warmup, runs, rknn_class, core_mask):
    import cv2

    from services.pipeline.license_plate_recognition import (
        detect_plate_regions,
        load_lpr_charset,
        recognize_plate_regions,
    )

    label, detector_path, ocr_path, charset_path, image_size_value = variant
    charset = load_lpr_charset(charset_path)
    image_size = int(image_size_value)
    if image_size != 960:
        raise ValueError(f"Fine-tuned detector requires image size 960, got {image_size}")
    detector = open_rknn(rknn_class, detector_path, core_mask)
    ocr = None
    try:
        ocr = open_rknn(rknn_class, ocr_path, core_mask)

        def infer(frame):
            started = time.perf_counter()
            regions = detect_plate_regions(frame, detector=detector, imgsz=image_size)
            detected = time.perf_counter()
            plates = recognize_plate_regions(regions, ocr=ocr, charset=charset)
            finished = time.perf_counter()
            return plates, (detected - started) * 1000, (finished - detected) * 1000

        for _ in range(warmup):
            for image_path in images:
                frame = cv2.imread(os.fspath(image_path))
                if frame is None:
                    raise ValueError(f"JPEG decode failed: {image_path}")
                infer(frame)

        detector_ms = []
        ocr_ms = []
        results = {}
        for _ in range(runs):
            for image_path in images:
                frame = cv2.imread(os.fspath(image_path))
                if frame is None:
                    raise ValueError(f"JPEG decode failed: {image_path}")
                plates, detect_elapsed, ocr_elapsed = infer(frame)
                detector_ms.append(detect_elapsed)
                ocr_ms.append(ocr_elapsed)
                path_key = str(image_path)
                expected = labels.get(path_key, {}).get("expected_plate")
                results.setdefault(path_key, []).append({
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "expected_plate": expected,
                    "exact_match": expected in [plate["plate"] for plate in plates]
                    if expected else None,
                    "plates": [
                        {"plate": plate["plate"], "det_conf": plate["det_conf"],
                         "ocr_status": plate["ocr_status"],
                         "ocr_outcome": plate.get("ocr_outcome"),
                         "ocr_confidence": plate.get("ocr_confidence")}
                        for plate in plates
                    ],
                })

        total_ms = [detect + ocr for detect, ocr in zip(detector_ms, ocr_ms)]
        return {
            "label": label,
            "detector_model": detector_path,
            "ocr_model": ocr_path,
            "charset": charset_path,
            "detector_sha256": sha256(detector_path),
            "ocr_sha256": sha256(ocr_path),
            "charset_sha256": sha256(charset_path),
            "image_size": image_size,
            "samples": len(total_ms),
            "mean_detect_ms": sum(detector_ms) / len(detector_ms),
            "mean_ocr_ms": sum(ocr_ms) / len(ocr_ms),
            "mean_total_ms": sum(total_ms) / len(total_ms),
            "p50_total_ms": percentile(total_ms, 0.50),
            "p95_total_ms": percentile(total_ms, 0.95),
            "fps": 1000.0 / (sum(total_ms) / len(total_ms)),
            "results": results,
        }
    finally:
        try:
            if ocr is not None:
                ocr.release()
        finally:
            detector.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument(
        "--variant", nargs=5, action="append", required=True,
        metavar=("LABEL", "DETECTOR_RKNN", "OCR_RKNN", "CHARSET", "IMAGE_SIZE"),
        help="Repeat to compare compatible 960-input fine-tuned model pairs",
    )
    parser.add_argument(
        "--manifest",
        help="Optional JSON object keyed by image path with expected_plate/session metadata",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--core", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be non-negative and --runs must be positive")

    images = collect_images(args.images)
    if not images:
        parser.error("No benchmark images found")

    from rknnlite.api import RKNNLite
    core_masks = (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)
    labels = {}
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as handle:
            labels = json.load(handle)
        if not isinstance(labels, dict):
            parser.error("--manifest must contain a JSON object")
    report = {
        "images": [str(path) for path in images],
        "warmup": args.warmup,
        "runs": args.runs,
        "variants": [
            run_variant(
                variant, images, labels, args.warmup, args.runs,
                RKNNLite, core_masks[args.core],
            )
            for variant in args.variant
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
