"""
Honest SARIMAX pipeline with train/validation/test evaluation.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from app.config import settings
from app.pipeline.forecasting import TimeSeriesSplit
from app.schemas.forecast import ForecastPoint, SarimaxResult, StationarityResult
from app.utils.metrics import compute_metrics


def check_stationarity(series: pd.Series, max_diff: int = 2) -> StationarityResult:
    """Run an ADF test on progressively differenced train data."""
    diff_order = 0
    current = series.dropna()

    for _ in range(max_diff + 1):
        adf_stat, p_value, *_rest = adfuller(current, autolag="AIC")
        if p_value <= 0.05:
            return StationarityResult(
                adf_statistic=round(float(adf_stat), 4),
                p_value=round(float(p_value), 4),
                is_stationary=True,
                differencing_order=diff_order,
            )
        diff_order += 1
        current = current.diff().dropna()

    return StationarityResult(
        adf_statistic=round(float(adf_stat), 4),
        p_value=round(float(p_value), 4),
        is_stationary=False,
        differencing_order=diff_order - 1,
    )


def _forecast_to_points(
    actual_series: pd.Series,
    predicted_values: np.ndarray,
) -> list[ForecastPoint]:
    return [
        ForecastPoint(
            date=str(actual_series.index[position].date()),
            actual=float(actual_series.iloc[position]),
            predicted=float(predicted_values[position]),
        )
        for position in range(len(actual_series))
    ]


def _fit_candidate(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=200)


def _select_order(
    split: TimeSeriesSplit,
    differencing_order: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], np.ndarray]:
    """Select SARIMAX order by validation RMSE without touching the test set."""
    seasonal_period = settings.sarimax_seasonal_period
    validation_values = split.validation["Weekly_Sales"].to_numpy(dtype=float)

    best_score = float("inf")
    best_order = (1, differencing_order, 1)
    best_seasonal_order = (0, 0, 0, seasonal_period)
    best_predictions = np.empty(len(validation_values), dtype=float)

    for p, q, seasonal_p, seasonal_q in itertools.product(range(3), range(3), range(2), range(2)):
        if p == 0 and q == 0 and seasonal_p == 0 and seasonal_q == 0:
            continue

        order = (p, differencing_order, q)
        seasonal_order = (seasonal_p, 0, seasonal_q, seasonal_period)
        try:
            fitted_model = _fit_candidate(split.train["Weekly_Sales"], order, seasonal_order)
            validation_pred = fitted_model.forecast(
                steps=len(split.validation),
            ).to_numpy(dtype=float)
            validation_rmse = compute_metrics(validation_values, validation_pred).rmse
        except Exception:
            continue

        if validation_rmse < best_score:
            best_score = validation_rmse
            best_order = order
            best_seasonal_order = seasonal_order
            best_predictions = validation_pred

    if not np.isfinite(best_score):
        raise ValueError("SARIMAX could not fit any candidate configuration")

    return best_order, best_seasonal_order, best_predictions


def run_sarimax_pipeline(store_df: pd.DataFrame, split: TimeSeriesSplit) -> SarimaxResult:
    """Train, validate and test SARIMAX without leaking test data."""
    train_series = split.train["Weekly_Sales"]
    validation_series = split.validation["Weekly_Sales"]
    combined_train_series = pd.concat([train_series, validation_series])
    test_series = split.test["Weekly_Sales"]

    stationarity = check_stationarity(train_series)
    order, seasonal_order, validation_predictions = _select_order(
        split,
        stationarity.differencing_order,
    )

    final_model = _fit_candidate(combined_train_series, order, seasonal_order)
    test_predictions = final_model.forecast(steps=len(test_series)).to_numpy(dtype=float)

    validation_metrics = compute_metrics(
        validation_series.to_numpy(dtype=float),
        validation_predictions,
    )
    test_metrics = compute_metrics(
        test_series.to_numpy(dtype=float),
        test_predictions,
    )

    return SarimaxResult(
        stationarity=stationarity,
        order=list(order),
        seasonal_order=list(seasonal_order),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_forecast=_forecast_to_points(validation_series, validation_predictions),
        test_forecast=_forecast_to_points(test_series, test_predictions),
        residuals=[float(value) for value in final_model.resid.to_numpy(dtype=float)],
    )
