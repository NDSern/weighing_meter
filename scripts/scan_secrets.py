#!/usr/bin/env python3
"""Fail when tracked files contain operational credentials.

Secrets belong in /etc/weighing-meter/weighing.env or an untracked
config.local.py. This scanner looks for non-empty assignments to known
credential names and for RTSP URLs with embedded user:password.

Only template/example paths and documentation are skipped; there is no
allowlist for real credential values.
"""

import argparse
import os
import re
import subprocess
import sys


ASSIGNMENT_PATTERN = re.compile(
    r'(?i)\b(MQTT_PASSWORD|MINIO_SECRET_KEY|MINIO_ACCESS_KEY|RTSP_PASSWORD)'
    r'\s*=\s*(["\'])(?P<value>[^"\']+)\2'
)
RTSP_CREDENTIALS_PATTERN = re.compile(
    r'rtsp://[^\s"\'/@:{}]+:[^\s"\'/@{}]+@'
)
# Example configs, tests, and documentation legitimately show placeholder
# URLs and dummy values. Production code, config, and systemd files are scanned.
SKIP_PATHS = {
    "config.local.example.py",
    "systemd/weighing.env.example",
    "scripts/scan_secrets.py",
    "tests/test_scan_secrets.py",
}
SKIP_PREFIXES = ("docs/", "tests/")
SKIP_SUFFIXES = (".md",)


def scan_text(text):
    """Return ``(line_number, kind)`` for each credential-looking match."""
    findings = []
    for index, line in enumerate(text.splitlines(), start=1):
        if ASSIGNMENT_PATTERN.search(line):
            findings.append((index, "credential assignment"))
        if RTSP_CREDENTIALS_PATTERN.search(line):
            findings.append((index, "rtsp credentials"))
    return findings


def _skip(relative_path):
    normalized = relative_path.replace(os.sep, "/")
    return (
        normalized in SKIP_PATHS
        or normalized.startswith(SKIP_PREFIXES)
        or normalized.endswith(SKIP_SUFFIXES)
    )


def tracked_files(root):
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def scan_tree(root="."):
    findings = []
    for relative_path in tracked_files(root):
        if _skip(relative_path):
            continue
        try:
            with open(os.path.join(root, relative_path), encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, kind in scan_text(text):
            findings.append((relative_path, line_number, kind))
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to scan")
    args = parser.parse_args(argv)

    findings = scan_tree(args.root)
    for path, line_number, kind in findings:
        print(f"{path}:{line_number}: {kind}")
    if findings:
        print(f"FAIL: {len(findings)} credential finding(s) in tracked files")
        return 1
    print("OK: no tracked credentials found")
    return 0


if __name__ == "__main__":
    sys.exit(main())