"""
ClusterApp Backend — API v1 роуты.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.forecast import ForecastRequest, SarimaxResponse
from app.schemas.store import SalesRecord, StoreAggregate
from app.service.data_loader import data_loader
from app.analytics.sarimax import run_sarimax_pipeline

router = APIRouter(prefix="/api/v1", tags=["API v1"])


# ─────────────────── Stores ───────────────────


@router.get("/stores", response_model=list[StoreAggregate])
async def get_stores():
    """Список магазинов с агрегированными метриками."""
    agg = data_loader.get_store_aggregates()
    return agg.to_dict(orient="records")


@router.get("/stores/{store_id}/sales", response_model=list[SalesRecord])
async def get_store_sales(store_id: int):
    """Все еженедельные продажи конкретного магазина."""
    try:
        store_df = data_loader.get_store_data(store_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    store_df = store_df.reset_index()
    records = []
    for _, row in store_df.iterrows():
        records.append(
            SalesRecord(
                store=store_id,
                date=str(row["Date"].date()),
                weekly_sales=row["Weekly_Sales"],
                holiday_flag=int(row["Holiday_Flag"]),
                temperature=row["Temperature"],
                fuel_price=row["Fuel_Price"],
                cpi=row["CPI"],
                unemployment=row["Unemployment"],
            )
        )
    return records


# ─────────────────── Forecast ───────────────────


@router.post("/forecast/sarimax", response_model=SarimaxResponse)
async def forecast_sarimax(request: ForecastRequest):
    """
    Полный SARIMAX-пайплайн:
    стационарность → подбор порядков → обучение → постпрогноз → прогноз → графики.
    """
    try:
        store_df = data_loader.get_store_data(request.store_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = run_sarimax_pipeline(store_df, horizon=request.horizon, store_id=request.store_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка SARIMAX: {str(e)}")

    return result
