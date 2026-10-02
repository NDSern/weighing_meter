#!/usr/bin/env python3
"""Remove peak-candidate evidence duplicated by active sessions."""

import argparse
import json
import os
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from config import PEAK_CANDIDATE_DIR


CATEGORY = "absorbed_active_session"
PENDING_SUFFIX = ".json.cleanup-pending"
RETAINED_CATEGORIES = {
    "blocked_waiting_for_empty",
    "stable_session",
    "unstable_local_peak",
}


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _load_record(metadata_path, root, allow_missing_images=False):
    root = root.resolve()
    metadata_path = Path(os.path.abspath(metadata_path))
    try:
        if metadata_path.is_symlink() or not metadata_path.is_file():
            return None, "metadata_not_regular_file"
        if not _within(metadata_path.resolve(strict=True), root):
            return None, "metadata_outside_root"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "invalid_metadata"
    if not isinstance(data, dict):
        return None, "invalid_metadata"
    category = data.get("category")
    if category in RETAINED_CATEGORIES:
        return None, "retained_category"
    if category != CATEGORY:
        return None, "unknown_category"
    images = data.get("images")
    if not isinstance(images, list) or not images:
        return None, "invalid_images"
    resolved = []
    for raw_path in images:
        if not isinstance(raw_path, str) or not raw_path:
            return None, "invalid_image_path"
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = metadata_path.parent / image_path
        image_path = Path(os.path.abspath(image_path))
        if not _within(image_path, root):
            return None, "image_outside_root"
        if image_path.is_symlink():
            return None, "image_symlink"
        try:
            resolved_image = image_path.resolve(strict=False)
        except OSError:
            return None, "image_path_unresolvable"
        if not _within(resolved_image, root):
            return None, "image_outside_root"
        if image_path.exists() and not image_path.is_file():
            return None, "image_not_regular_file"
        if not image_path.exists() and not allow_missing_images:
            return None, "image_missing"
        resolved.append(image_path)
    return {"metadata": metadata_path, "images": resolved}, None


def inventory(root):
    root = Path(root).resolve()
    candidates = []
    unsafe = []
    for metadata_path in sorted(root.rglob("*.json")):
        record, reason = _load_record(metadata_path, root)
        if record:
            candidates.append(record)
        elif reason != "retained_category":
            unsafe.append((metadata_path, reason))
    pending = []
    for metadata_path in sorted(root.rglob(f"*{PENDING_SUFFIX}")):
        record, reason = _load_record(metadata_path, root, allow_missing_images=True)
        if record:
            pending.append(record)
        else:
            unsafe.append((metadata_path, reason))
    files = [path for record in candidates for path in record["images"]]
    files += [record["metadata"] for record in candidates]
    bytes_total = 0
    for path in files:
        try:
            bytes_total += path.stat().st_size
        except OSError:
            # File removed by a concurrent writer between listing and stat.
            continue
    return {
        "root": root,
        "candidates": candidates,
        "pending": pending,
        "unsafe": unsafe,
        "files": len(files),
        "bytes": bytes_total,
    }


def _remove_empty_parents(paths, root):
    root = root.resolve()
    for parent in sorted({path.parent for path in paths}, key=lambda p: len(p.parts), reverse=True):
        while parent != root and _within(parent, root):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def apply_cleanup(report):
    deleted_records = deleted_images = reclaimed = failed = 0
    affected = []
    for record in report["pending"] + report["candidates"]:
        metadata_path = record["metadata"]
        if metadata_path.name.endswith(PENDING_SUFFIX):
            pending_path = metadata_path
        else:
            pending_path = metadata_path.with_name(metadata_path.name + ".cleanup-pending")
            try:
                os.replace(metadata_path, pending_path)
            except OSError:
                failed += 1
                continue
        record_failed = False
        for image_path in record["images"]:
            try:
                if image_path.exists():
                    size = image_path.stat().st_size
                    image_path.unlink()
                    reclaimed += size
                    deleted_images += 1
                affected.append(image_path)
            except OSError:
                failed += 1
                record_failed = True
                break
        if record_failed:
            continue
        try:
            size = pending_path.stat().st_size
            pending_path.unlink()
            reclaimed += size
            deleted_records += 1
            affected.append(pending_path)
        except OSError:
            failed += 1
    _remove_empty_parents(affected, report["root"])
    return {
        "deleted_records": deleted_records,
        "deleted_images": deleted_images,
        "reclaimed": reclaimed,
        "failed": failed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=PEAK_CANDIDATE_DIR)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)

    report = inventory(args.root)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "root": str(report["root"]),
        "candidate_records": len(report["candidates"]),
        "pending_records": len(report["pending"]),
        "candidate_files": report["files"],
        "candidate_bytes": report["bytes"],
        "unsafe_records": len(report["unsafe"]),
    }
    manifest = {
        **summary,
        "candidates": [
            {
                "metadata": str(record["metadata"]),
                "images": [str(path) for path in record["images"]],
            }
            for record in report["candidates"]
        ],
        "pending": [
            {
                "metadata": str(record["metadata"]),
                "images": [str(path) for path in record["images"]],
            }
            for record in report["pending"]
        ],
        "unsafe": [
            {"path": str(path), "reason": reason}
            for path, reason in report["unsafe"]
        ],
    }
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if args.apply:
        result = apply_cleanup(report)
        print(json.dumps(result, sort_keys=True))
        return 1 if result["failed"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
