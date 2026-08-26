PYTHON ?= python3

.PHONY: test verify-models verify

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

verify-models:
	cd models/lpr && sha256sum -c SHA256SUMS

verify: verify-models test
