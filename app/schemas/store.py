"""
ClusterApp Backend — Pydantic-схемы для магазинов.
"""

from pydantic import BaseModel


class SalesRecord(BaseModel):
    """Запись из CSV — одна неделя одного магазина."""

    store: int
    date: str
    weekly_sales: float
    holiday_flag: int
    temperature: float
    fuel_price: float
    cpi: float
    unemployment: float


class StoreAggregate(BaseModel):
    """Агрегированные метрики магазина."""

    store_id: int
    mean_sales: float
    std_sales: float
    mean_temperature: float
    mean_cpi: float
    mean_unemployment: float
    mean_fuel_price: float
