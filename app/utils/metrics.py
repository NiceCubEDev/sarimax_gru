import numpy as np

from app.schemas.forecast import ModelMetrics


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> ModelMetrics:
    """Compute forecast metrics on aligned arrays."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")

    mae = float(np.mean(np.abs(actual_arr - predicted_arr)))
    mse = float(np.mean((actual_arr - predicted_arr) ** 2))
    rmse = float(np.sqrt(mse))

    nonzero_mask = actual_arr != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual_arr[nonzero_mask] - predicted_arr[nonzero_mask])
                    / actual_arr[nonzero_mask]
                )
            )
            * 100
        )
        wape = float(
            np.sum(np.abs(actual_arr - predicted_arr))
            / np.sum(np.abs(actual_arr[nonzero_mask]))
            * 100
        )
    else:
        mape = 0.0
        wape = 0.0

    denominator = np.abs(actual_arr) + np.abs(predicted_arr)
    smape_mask = denominator != 0
    if smape_mask.any():
        smape = float(
            np.mean(
                2
                * np.abs(predicted_arr[smape_mask] - actual_arr[smape_mask])
                / denominator[smape_mask]
            )
            * 100
        )
    else:
        smape = 0.0

    return ModelMetrics(
        mae=round(mae, 2),
        mse=round(mse, 2),
        rmse=round(rmse, 2),
        mape=round(mape, 2),
        smape=round(smape, 2),
        wape=round(wape, 2),
    )
