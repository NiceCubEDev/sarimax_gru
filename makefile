# ============================================
# ClusterApp offline forecasting pipeline
# ============================================

PYTHON = .venv/bin/python
STORE_ID ?= 1

.PHONY: lint lint-fix fmt fmt-check initial-methods clean help

lint:
	$(PYTHON) -m ruff check app

lint-fix:
	$(PYTHON) -m ruff check app --fix

fmt:
	$(PYTHON) -m ruff format app

fmt-check:
	$(PYTHON) -m ruff format app --check

initial-methods:
	$(PYTHON) -m app.cli initial-methods --store-id $(STORE_ID)

clean:
	rm -rf build/reports/*

help:
	@echo ""
	@echo "  ClusterApp offline pipeline - available commands:"
	@echo ""
	@echo "  make lint                    - Ruff check"
	@echo "  make lint-fix                - Ruff auto-fix"
	@echo "  make fmt                     - Ruff format"
	@echo "  make fmt-check               - Ruff format check"
	@echo "  make initial-methods         - Run validation, SARIMAX, GRU and build PDF"
	@echo "  make initial-methods STORE_ID=5"
	@echo "  make clean                   - Remove generated reports"
	@echo ""
