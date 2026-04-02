import numpy as np

from app.schemas.forecast import ModelMetrics


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> ModelMetrics:
    """Вычисление MAE, RMSE, MAPE."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # MAPE с защитой от деления на 0
    nonzero = actual != 0
    if nonzero.any():
        abs_pct_error = np.abs(
            (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
        )
        mape = float(np.mean(abs_pct_error) * 100)
    else:
        mape = 0.0

    return ModelMetrics(mae=round(mae, 2), rmse=round(rmse, 2), mape=round(mape, 2))
