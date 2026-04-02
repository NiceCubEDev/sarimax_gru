"""
ClusterApp Backend — FastAPI application.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.rest import router as api_router
from app.service.data_loader import data_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка данных при старте сервера."""
    data_loader.load()
    yield


app = FastAPI(
    title="ClusterApp Backend",
    description="API для прогнозирования продаж Walmart (SARIMAX + GRU)",
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

# API роуты
app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health():
    """Health check."""
    return {"status": "ok"}
