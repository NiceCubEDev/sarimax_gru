# ============================================
# ClusterApp Backend — Multi-stage Dockerfile
# ============================================

# --------------- BUILD STAGE ----------------
FROM python:3.12-slim AS builder

WORKDIR /tmp

# Poetry
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Системные зависимости для сборки Python-пакетов
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main --no-cache

# ------------- PRODUCTION STAGE -------------
FROM python:3.12-slim AS production

LABEL maintainer="clusterapp-team"
LABEL description="ClusterApp Backend — FastAPI async API"

# Непривилегированный пользователь
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Runtime-зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Копируем код и данные
COPY ./app ./app
COPY ./data ./data

# Права
RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
