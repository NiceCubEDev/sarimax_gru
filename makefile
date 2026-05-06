# ============================================
# ClusterApp offline-пайплайн прогнозирования
# ============================================

PYTHON = .venv/bin/python
STORE_ID ?= 1
HORIZON ?= 12

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
	$(PYTHON) -m app.cli initial-methods --store-id $(STORE_ID) --horizon $(HORIZON)

clean:
	rm -rf build/reports/*

help:
	@echo ""
	@echo "  ClusterApp offline-пайплайн - доступные команды:"
	@echo ""
	@echo "  make lint                    - Проверка Ruff"
	@echo "  make lint-fix                - Автоисправления Ruff"
	@echo "  make fmt                     - Форматирование Ruff"
	@echo "  make fmt-check               - Проверка форматирования Ruff"
	@echo "  make initial-methods         - Валидация, SARIMAX, GRU и сборка PDF"
	@echo "  make initial-methods STORE_ID=5 HORIZON=12"
	@echo "  make clean                   - Удалить сгенерированные отчеты"
	@echo ""
