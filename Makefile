PYTHON ?= python3
WARNINGS ?= -W error::ResourceWarning

.PHONY: test compile scan-secrets lint verify-models verify probes

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. $(PYTHON) $(WARNINGS) -m unittest discover -s tests -p 'test_*.py'

compile:
	$(PYTHON) -m compileall -q weighing_service.py services scripts config.py

scan-secrets:
	$(PYTHON) scripts/scan_secrets.py

lint:
	$(PYTHON) -m ruff check .

verify-models:
	cd models/lpr && sha256sum -c SHA256SUMS

verify: verify-models scan-secrets lint test

# Hardware/manual probes. Not part of CI: they need a live broker or device.
probes:
	$(PYTHON) tests/mqtt_probe.py