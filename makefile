# ============================================
# ClusterApp Backend — Makefile
# ============================================

COMPOSE = docker compose -f build/docker-compose.yml --project-directory .

# --------------- Docker --------------------

.PHONY: build up down restart logs ps

## Собрать образы
build:
	$(COMPOSE) build

## Запустить все сервисы
up:
	$(COMPOSE) up -d

## Запустить с пересборкой
up-build:
	$(COMPOSE) up -d --build

## Остановить все сервисы
down:
	$(COMPOSE) down

## Остановить и удалить volumes
down-v:
	$(COMPOSE) down -v

## Перезапустить сервисы
restart:
	$(COMPOSE) restart

## Показать логи (follow)
logs:
	$(COMPOSE) logs -f

## Логи только backend
logs-app:
	$(COMPOSE) logs -f app

## Статус контейнеров
ps:
	$(COMPOSE) ps

# --------------- Development ---------------

.PHONY: shell db-shell redis-cli

## Зайти в shell backend-контейнера
shell:
	$(COMPOSE) exec app bash

## Зайти в psql
db-shell:
	$(COMPOSE) exec db psql -U $${POSTGRES_USER:-clusterapp} -d $${POSTGRES_DB:-clusterapp}

## Зайти в redis-cli
redis-cli:
	$(COMPOSE) exec redis redis-cli

# --------------- Lint / Format ---------------

.PHONY: lint lint-fix fmt fmt-check

## Проверка кода (ruff check)
lint:
	uv run ruff check app/

## Автоисправление ошибок линтера
lint-fix:
	uv run ruff check app/ --fix

## Форматирование кода
fmt:
	uv run ruff format app/

## Проверка форматирования (без изменений)
fmt-check:
	uv run ruff format app/ --check

# --------------- Help ----------------------

.PHONY: help
help:
	@echo ""
	@echo "  ClusterApp Backend — доступные команды:"
	@echo ""
	@echo "  make build       — Собрать Docker-образы"
	@echo "  make up          — Запустить все сервисы"
	@echo "  make up-build    — Запустить с пересборкой"
	@echo "  make down        — Остановить сервисы"
	@echo "  make down-v      — Остановить + удалить volumes"
	@echo "  make restart     — Перезапустить"
	@echo "  make logs        — Логи всех сервисов"
	@echo "  make logs-app    — Логи backend"
	@echo "  make ps          — Статус контейнеров"
	@echo "  make shell       — Shell в backend"
	@echo "  make db-shell    — psql в PostgreSQL"
	@echo "  make redis-cli   — redis-cli"
	@echo "  make lint        — Проверка кода (ruff)"
	@echo "  make lint-fix    — Автоисправление (ruff --fix)"
	@echo "  make fmt         — Форматирование (ruff format)"
	@echo "  make fmt-check   — Проверка форматирования"
	@echo ""
