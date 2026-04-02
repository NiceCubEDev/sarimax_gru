"""
ClusterApp Backend — FastAPI application.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.rest import router as api_router
from app.config import settings
from app.service.data_loader import data_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка данных при старте сервера."""
    data_loader.load()
    # Создаём директорию для графиков
    settings.plots_full_path.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="ClusterApp Backend",
    description="API для прогнозирования продаж Walmart (SARIMAX)",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — для React-фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика для графиков
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# API роуты
app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health():
    """Health check."""
    return {"status": "ok"}
