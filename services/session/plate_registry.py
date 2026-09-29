"""Registered license plate registry and plate normalization helpers."""

import json
import os
import threading

from services.session import plate_store

from config import SERVICE_DIR

_log_fn = None
_registry_lock = threading.Lock()
_registry_loaded = False
_registry_mtime = None
_registry_exact = {}
_registry_family = {}
_registry_active_count = 0


def set_log_fn(log_fn):
    """Set the logging function used by registry operations."""
    global _log_fn
    _log_fn = log_fn


def log(level: str, msg: str):
    if _log_fn:
        _log_fn(level, msg)


def saveConfirmedLicensePlate(license_plate, session_id=None):
    """Persist confirmed plate count once per confirmed session."""
    db_file = os.path.join(SERVICE_DIR, "confirmed_license_plates.db")
    try:
        return plate_store.increment(db_file, license_plate, session_id)
    except Exception as exc:
        print(f"[PLATE_DB] saveConfirmedLicensePlate failed: {exc}", flush=True)
        return None


def normalizeLicensePlate(license_plate):
    return "".join(ch for ch in (license_plate or "").upper() if ch.isalnum())


def licensePlatePrefix(normalized_plate):
    if len(normalized_plate) < 3:
        return normalized_plate
    return normalized_plate[:3]


def registeredFamilyKeys(normalized_plate):
    keys = {normalized_plate}
    if len(normalized_plate) >= 8:
        keys.add(normalized_plate[:-1])
    return keys


def editDistanceAtMostOne(left, right):
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


def loadRegisteredLicensePlates():
    global _registry_loaded, _registry_mtime, _registry_exact, _registry_family, _registry_active_count

    registry_file = os.path.join(SERVICE_DIR, "registered_license_plates.json")
    try:
        mtime = os.path.getmtime(registry_file)
    except OSError:
        mtime = None

    with _registry_lock:
        if _registry_loaded and _registry_mtime == mtime:
            return

        exact = {}
        family = {}
        active_count = 0
        if mtime is not None:
            try:
                with open(registry_file, "r", encoding="utf-8") as fh:
                    rows = json.load(fh)
                for row in rows:
                    if not isinstance(row, dict) or not row.get("active", True):
                        continue
                    plate = row.get("plate")
                    normalized = normalizeLicensePlate(plate)
                    if not normalized:
                        continue
                    active_count += 1
                    exact[normalized] = plate
                    for key in registeredFamilyKeys(normalized):
                        family.setdefault(key, []).append(plate)
            except Exception as exc:
                log("ERROR", f"[REGISTRY] Failed to load registered_license_plates.json: {exc}")

        _registry_loaded = True
        _registry_mtime = mtime
        _registry_exact = exact
        _registry_family = family
        _registry_active_count = active_count
        if mtime is None:
            log("REGISTRY", "No registered_license_plates.json found")
        else:
            log("REGISTRY", f"Loaded {active_count} active registered plates")


def correctWithRegisteredLicensePlate(license_plate):
    if not license_plate or license_plate == "none":
        return license_plate, None

    loadRegisteredLicensePlates()
    normalized = normalizeLicensePlate(license_plate)
    if not normalized:
        return license_plate, None

    with _registry_lock:
        exact = dict(_registry_exact)
        family = {key: list(value) for key, value in _registry_family.items()}

    if normalized in exact:
        registered = exact[normalized]
        if registered != license_plate:
            return registered, "exact"
        return license_plate, None

    family_matches = list(dict.fromkeys(family.get(normalized, [])))
    if len(family_matches) == 1:
        return family_matches[0], "family_unique"

    fuzzy_matches = []
    for registered_norm, registered_plate in exact.items():
        if editDistanceAtMostOne(normalized, registered_norm):
            fuzzy_matches.append(registered_plate)

    fuzzy_matches = list(dict.fromkeys(fuzzy_matches))
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], "fuzzy_distance_1"
    if len(family_matches) > 1 or len(fuzzy_matches) > 1:
        log("REGISTRY", f"Kept plate {license_plate} reason=ambiguous registered_matches={family_matches + fuzzy_matches}")
    return license_plate, None


def preferDetailedLicensePlateCandidate(license_plate, all_plates):
    """Prefer 5-digit VN plate over same shortened 4-digit variant when seen in session."""
    cleaned = "".join(ch for ch in (license_plate or "").upper() if ch.isalnum())
    if len(cleaned) != 7:
        return license_plate

    detailed = []
    for plate, count in all_plates.items():
        candidate = "".join(ch for ch in plate.upper() if ch.isalnum())
        if len(candidate) == 8 and candidate.endswith("0") and candidate[:-1] == cleaned:
            detailed.append((plate, count))
    if not detailed:
        return license_plate
    return max(detailed, key=lambda item: item[1])[0]