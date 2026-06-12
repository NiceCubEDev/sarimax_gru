# ============================================
# ClusterApp offline-пайплайн прогнозирования
# ============================================

PYTHON = .venv/bin/python
STORE_ID ?= 1
STORE ?= $(STORE_ID)
HORIZON ?= 12

.DEFAULT_GOAL := help

.PHONY: lint lint-fix fmt fmt-check run report app initial-methods clean help

lint:
	$(PYTHON) -m ruff check app

lint-fix:
	$(PYTHON) -m ruff check app --fix

fmt:
	$(PYTHON) -m ruff format app

fmt-check:
	$(PYTHON) -m ruff format app --check

run:
	@$(PYTHON) -m app.cli initial-methods --store-id $(STORE) --horizon $(HORIZON)

report: run

app: run

initial-methods: run

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
	@echo "  make run                     - Запустить прогноз для магазина 1"
	@echo "  make run STORE=5             - Запустить прогноз для магазина 5"
	@echo "  make run STORE=5 HORIZON=12  - Запустить прогноз с горизонтом 12 недель"
	@echo "  make initial-methods         - Старый алиас для make run"
	@echo "  make clean                   - Удалить сгенерированные отчеты"
	@echo ""
