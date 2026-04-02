# ============================================
# ClusterApp Backend — Multi-stage Dockerfile
# ============================================

# --------------- BUILD STAGE ----------------
FROM python:3.12-slim AS builder

WORKDIR /tmp

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Системные зависимости для сборки Python-пакетов
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости в системный Python (без venv)
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml uv.lock ./
RUN uv pip install --requirement pyproject.toml

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
