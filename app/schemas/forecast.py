"""
ClusterApp Backend — Pydantic-схемы для прогнозирования.
"""

from pydantic import BaseModel


class ForecastRequest(BaseModel):
    """Запрос на прогнозирование."""

    store_id: int
    horizon: int = 12


class StationarityResult(BaseModel):
    """Результат ADF-теста на стационарность."""

    adf_statistic: float
    p_value: float
    is_stationary: bool
    differencing_order: int


class ForecastPoint(BaseModel):
    """Одна точка прогноза."""

    date: str
    actual: float | None = None
    predicted: float
    lower_ci: float | None = None
    upper_ci: float | None = None


class ModelMetrics(BaseModel):
    """Метрики качества модели."""

    mae: float
    rmse: float
    mape: float


class ForecastPlots(BaseModel):
    """URL-ы графиков."""

    time_series_url: str
    forecast_url: str
    residuals_url: str | None = None
    acf_pacf_url: str | None = None
    training_loss_url: str | None = None


class SarimaxResponse(BaseModel):
    """Полный ответ SARIMAX: стационарность, параметры, прогнозы, метрики, графики."""

    stationarity: StationarityResult
    order: list[int]  # [p, d, q]
    seasonal_order: list[int]  # [P, D, Q, s]
    train_forecast: list[ForecastPoint]
    test_forecast: list[ForecastPoint]
    metrics: ModelMetrics
    plots: ForecastPlots


class GruHyperparams(BaseModel):
    """Гиперпараметры GRU-модели."""

    hidden_size: int
    num_layers: int
    epochs_trained: int
    learning_rate: float
    sequence_length: int
    features: list[str]


class GruResponse(BaseModel):
    """Полный ответ GRU: гиперпараметры, прогнозы, метрики, графики."""

    hyperparams: GruHyperparams
    train_forecast: list[ForecastPoint]
    test_forecast: list[ForecastPoint]
    metrics: ModelMetrics
    plots: ForecastPlots

