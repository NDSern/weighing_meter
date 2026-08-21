import hashlib


EXPECTED_LPR_SHA256 = {
    "detector": "4c580314148bde47920e20bd9e13969d96017fdbf64a0ed49a1679c45b1d3be8",
    "recognizer": "ec88eff23206cf8c6fa609e3c9130e7b2d4caba31c846cc03969680ca2ce4eb3",
    "charset": "d798cc4724b12e455112439aca5b51ee29da71a264663e5dbfda25ed24a391f4",
    "decoder": "d370fa5aabb1f4e9d17349b09a5095a42689bbe709b2c55a416d684822c727ba",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lpr_bundle(paths):
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    for name, expected in EXPECTED_LPR_SHA256.items():
        if hashes.get(name) != expected:
            raise ValueError(f"Unexpected {name} SHA256: {hashes.get(name)}")
    return hashes
