"""
Honest SARIMAX pipeline with train/test evaluation.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from app.config import settings
from app.pipeline.forecasting import (
    CONTINUOUS_FUTURE_COLUMNS,
    TimeSeriesSplit,
    build_future_dates,
    build_future_feature_frame,
)
from app.schemas.forecast import ForecastPoint, SarimaxResult, StationarityResult
from app.utils.metrics import compute_metrics


EXOG_COLUMNS = [*CONTINUOUS_FUTURE_COLUMNS, "Holiday_Flag"]


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


def _future_forecast_to_points(
    future_dates: pd.DatetimeIndex,
    predicted_values: np.ndarray,
) -> list[ForecastPoint]:
    return [
        ForecastPoint(
            date=str(future_dates[position].date()),
            predicted=float(predicted_values[position]),
        )
        for position in range(len(future_dates))
    ]


def _fit_candidate(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog: pd.DataFrame | None = None,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=200)


def _select_order(
    split: TimeSeriesSplit,
    differencing_order: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Select SARIMAX order by AIC on the training set without touching the test set."""
    seasonal_period = settings.sarimax_seasonal_period

    best_score = float("inf")
    best_order = (1, differencing_order, 1)
    best_seasonal_order = (0, 0, 0, seasonal_period)

    for p, q, seasonal_p, seasonal_q in itertools.product(range(3), range(3), range(2), range(2)):
        if p == 0 and q == 0 and seasonal_p == 0 and seasonal_q == 0:
            continue

        order = (p, differencing_order, q)
        seasonal_order = (seasonal_p, 0, seasonal_q, seasonal_period)
        try:
            fitted_model = _fit_candidate(
                split.train["Weekly_Sales"],
                order,
                seasonal_order,
                split.train[EXOG_COLUMNS],
            )
        except Exception:
            continue

        if fitted_model.aic < best_score:
            best_score = fitted_model.aic
            best_order = order
            best_seasonal_order = seasonal_order

    if not np.isfinite(best_score):
        raise ValueError("SARIMAX could not fit any candidate configuration")

    return best_order, best_seasonal_order


def run_sarimax_pipeline(
    store_df: pd.DataFrame,
    split: TimeSeriesSplit,
    horizon: int,
) -> SarimaxResult:
    """Train and test SARIMAX without leaking test data."""
    train_series = split.train["Weekly_Sales"]
    test_series = split.test["Weekly_Sales"]
    full_series = store_df["Weekly_Sales"]
    train_exog = split.train[EXOG_COLUMNS]
    test_exog = split.test[EXOG_COLUMNS]
    full_exog = store_df[EXOG_COLUMNS]

    stationarity = check_stationarity(train_series)
    order, seasonal_order = _select_order(
        split,
        stationarity.differencing_order,
    )

    fitted_model = _fit_candidate(train_series, order, seasonal_order, train_exog)
    training_predictions = fitted_model.predict(
        start=0,
        end=len(train_series) - 1,
        exog=train_exog,
    ).to_numpy(dtype=float, copy=True)
    training_predictions[0] = float(train_series.iloc[0])

    test_predictions = fitted_model.forecast(
        steps=len(test_series),
        exog=test_exog,
    ).to_numpy(dtype=float)
    future_model = _fit_candidate(full_series, order, seasonal_order, full_exog)
    history_predictions = future_model.predict(
        start=0,
        end=len(full_series) - 1,
        exog=full_exog,
    ).to_numpy(dtype=float, copy=True)
    history_predictions[0] = float(full_series.iloc[0])
    future_dates = build_future_dates(store_df.index.max(), horizon)
    future_exog = build_future_feature_frame(store_df, future_dates)
    future_predictions = future_model.forecast(
        steps=horizon,
        exog=future_exog[EXOG_COLUMNS],
    ).to_numpy(dtype=float)

    training_metrics = compute_metrics(
        train_series.to_numpy(dtype=float),
        training_predictions,
    )
    test_metrics = compute_metrics(
        test_series.to_numpy(dtype=float),
        test_predictions,
    )

    return SarimaxResult(
        stationarity=stationarity,
        order=list(order),
        seasonal_order=list(seasonal_order),
        training_metrics=training_metrics,
        test_metrics=test_metrics,
        history_forecast=_forecast_to_points(full_series, history_predictions),
        test_forecast=_forecast_to_points(test_series, test_predictions),
        future_forecast=_future_forecast_to_points(future_dates, future_predictions),
        residuals=[float(value) for value in fitted_model.resid.to_numpy(dtype=float)],
    )
