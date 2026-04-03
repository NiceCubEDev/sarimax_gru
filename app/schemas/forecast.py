"""
ClusterApp Backend - schemas for forecasting and PDF reporting.
"""

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request for generating a comparison report."""

    store_id: int = Field(gt=0)


class DataValidationSummary(BaseModel):
    """Summary of validated input data."""

    row_count: int
    start_date: str
    end_date: str
    frequency_days: int
    missing_values: int
    duplicated_dates: int


class DataSplitSummary(BaseModel):
    """Chronological train/validation/test split summary."""

    train_size: int
    validation_size: int
    test_size: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


class ForecastPoint(BaseModel):
    """One prediction point."""

    date: str
    actual: float
    predicted: float


class ModelMetrics(BaseModel):
    """Forecast quality metrics."""

    mae: float
    rmse: float
    mape: float
    smape: float
    wape: float


class StationarityResult(BaseModel):
    """ADF-based stationarity check result."""

    adf_statistic: float
    p_value: float
    is_stationary: bool
    differencing_order: int


class SarimaxResult(BaseModel):
    """SARIMAX report section."""

    stationarity: StationarityResult
    order: list[int]
    seasonal_order: list[int]
    validation_metrics: ModelMetrics
    test_metrics: ModelMetrics
    validation_forecast: list[ForecastPoint]
    test_forecast: list[ForecastPoint]
    residuals: list[float]


class GruHyperparams(BaseModel):
    """GRU hyperparameters used in training."""

    hidden_size: int
    num_layers: int
    learning_rate: float
    sequence_length: int
    selected_epochs: int
    features: list[str]


class GruResult(BaseModel):
    """GRU report section."""

    hyperparams: GruHyperparams
    validation_metrics: ModelMetrics
    test_metrics: ModelMetrics
    validation_forecast: list[ForecastPoint]
    test_forecast: list[ForecastPoint]
    training_losses: list[float]


class ComparisonReport(BaseModel):
    """Complete comparison result for both models."""

    store_id: int
    data_validation: DataValidationSummary
    split: DataSplitSummary
    sarimax: SarimaxResult
    gru: GruResult
    pdf_path: str
