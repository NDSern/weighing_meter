"""Helpers for YYYY/MM/DD dated storage trees.

Retention code walks root/YYYY/MM/DD directories and parses dates from either
the path layout or an embedded ``YYYYMMDD`` / ``YYYY_MM_DD`` filename token.
These helpers keep that parsing and the deepest-first walk in one place.
"""

import os
import re
from datetime import date

COMPACT_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]")
SEPARATED_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)")


def make_date(year, month, day):
    """Return a date from string parts, or None when they are not a real date."""
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def date_from_filename(filename):
    """Parse a ``YYYYMMDD`` or ``YYYY_MM_DD`` token from a filename."""
    for regex in (COMPACT_DATE_RE, SEPARATED_DATE_RE):
        match = regex.search(filename)
        if not match:
            continue
        parsed = make_date(*match.groups())
        if parsed:
            return parsed
    return None


def date_from_relative_path(relative_path, separator=os.sep):
    """Parse the leading YYYY/MM/DD segments of a path relative to a root."""
    parts = relative_path.split(separator)
    if len(parts) < 4:
        return None
    year, month, day = parts[:3]
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        return None
    return make_date(year, month, day)


def iter_dirs_deepest_first(root):
    """Yield every (non-symlink) subdirectory under root, deepest first."""
    dirs = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=False):
        dirnames[:] = [
            name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))
        ]
        for dirname in dirnames:
            dirs.append(os.path.join(dirpath, dirname))
    dirs.sort(key=lambda path: path.count(os.sep), reverse=True)
    return dirs
