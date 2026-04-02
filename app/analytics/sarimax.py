"""
ClusterApp Backend — SARIMAX аналитический модуль.

Пайплайн:
1. Проверка стационарности (ADF-тест)
2. ACF/PACF анализ
3. Выбор порядков модели (p,d,q)(P,D,Q,s)
4. Обучение SARIMAX
5. Постпрогноз (in-sample)
6. Прогноз (out-of-sample)
7. Метрики (MAE, RMSE, MAPE)
8. Графики
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from app.config import settings
from app.schemas.forecast import (
    ForecastPlots,
    ForecastPoint,
    SarimaxResponse,
    StationarityResult,
)
from app.utils.metrics import compute_metrics
from app.utils.plots import (
    plot_acf_pacf,
    plot_forecast,
    plot_residuals,
    plot_time_series,
)


# Кэш результатов: ключ (store_id, horizon) → SarimaxResponse
_cache: dict[tuple[int, int], SarimaxResponse] = {}


def check_stationarity(series: pd.Series, max_diff: int = 2) -> StationarityResult:
    """
    Проверка стационарности с помощью ADF-теста.
    Если ряд нестационарен — дифференцируем и проверяем снова.
    """
    diff_order = 0
    current = series.dropna()

    for _d in range(max_diff + 1):
        result = adfuller(current, autolag="AIC")
        adf_stat, p_value = result[0], result[1]

        if p_value <= 0.05:
            return StationarityResult(
                adf_statistic=round(adf_stat, 4),
                p_value=round(p_value, 4),
                is_stationary=True,
                differencing_order=diff_order,
            )

        diff_order += 1
        current = current.diff().dropna()

    # Если после max_diff ряд всё ещё нестационарен
    return StationarityResult(
        adf_statistic=round(adf_stat, 4),
        p_value=round(p_value, 4),
        is_stationary=False,
        differencing_order=diff_order - 1,
    )


def _fit_seasonal_variants(
    series: pd.Series,
    p: int,
    d: int,
    q: int,
    seasonal_period: int,
    best_aic: float,
    best_order: tuple[int, int, int],
    best_seasonal: tuple[int, int, int, int],
) -> tuple[float, tuple[int, int, int], tuple[int, int, int, int]]:
    """Перебор сезонных параметров (P, Q) для фиксированных (p, d, q)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for P, Q in itertools.product(range(2), range(2)):
            try:
                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, 0, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=50)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_seasonal = (P, 0, Q, seasonal_period)
            except Exception:
                continue
    return best_aic, best_order, best_seasonal


def _select_order(
    series: pd.Series,
    d: int,
    seasonal_period: int,
    max_p: int = 2,
    max_q: int = 2,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """
    Подбор порядков SARIMAX по минимальному AIC.
    Перебирает комбинации (p, d, q) и (P, D, Q, s).
    """
    best_aic = np.inf
    best_order = (1, d, 1)
    best_seasonal = (0, 0, 0, seasonal_period)

    # Упрощённый перебор для скорости
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            best_aic, best_order, best_seasonal = _fit_seasonal_variants(
                series, p, d, q, seasonal_period,
                best_aic, best_order, best_seasonal,
            )

    return best_order, best_seasonal


def run_sarimax_pipeline(
    store_df: pd.DataFrame,
    horizon: int = 12,
    store_id: int | None = None,
) -> SarimaxResponse:
    """
    Полный SARIMAX-пайплайн для одного магазина.
    Результат кэшируется по (store_id, horizon).
    """
    # Проверяем кэш
    if store_id is not None:
        cache_key = (store_id, horizon)
        if cache_key in _cache:
            return _cache[cache_key]
    series = store_df["Weekly_Sales"]
    seasonal_period = settings.sarimax_seasonal_period
    train_ratio = settings.sarimax_train_ratio

    # --- 1. Стационарность ---
    stationarity = check_stationarity(series)

    # --- 2. ACF / PACF ---
    acf_pacf_url = plot_acf_pacf(series.values, lags=min(40, len(series) // 2 - 1))

    # --- 3. Выбор порядков ---
    order, seasonal_order = _select_order(series, stationarity.differencing_order, seasonal_period)

    # --- 4. Train/test split ---
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    # --- 5. Обучение ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted_model = model.fit(disp=False, maxiter=200)

    # --- 6. Постпрогноз (in-sample) ---
    in_sample_pred = fitted_model.get_prediction(start=0, end=len(train) - 1)
    in_sample_mean = in_sample_pred.predicted_mean

    train_forecast = [
        ForecastPoint(
            date=str(train.index[i].date()),
            actual=float(train.iloc[i]),
            predicted=float(in_sample_mean.iloc[i]),
        )
        for i in range(len(train))
    ]

    # --- 7. Прогноз (out-of-sample) ---
    forecast_steps = len(test) + horizon
    forecast_result = fitted_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()

    test_forecast = []
    for i in range(len(test)):
        test_forecast.append(
            ForecastPoint(
                date=str(test.index[i].date()),
                actual=float(test.iloc[i]),
                predicted=float(forecast_mean.iloc[i]),
                lower_ci=float(forecast_ci.iloc[i, 0]),
                upper_ci=float(forecast_ci.iloc[i, 1]),
            )
        )

    # Будущие точки (после test)
    for i in range(len(test), forecast_steps):
        test_forecast.append(
            ForecastPoint(
                date=(
                    str(forecast_mean.index[i].date())
                    if hasattr(forecast_mean.index[i], "date")
                    else str(forecast_mean.index[i])
                ),
                actual=None,
                predicted=float(forecast_mean.iloc[i]),
                lower_ci=float(forecast_ci.iloc[i, 0]),
                upper_ci=float(forecast_ci.iloc[i, 1]),
            )
        )

    # --- 8. Метрики (на test) ---
    test_actual = test.values
    test_predicted = forecast_mean.iloc[: len(test)].values
    metrics = compute_metrics(test_actual, test_predicted)

    # --- 9. Графики ---
    ts_url = plot_time_series(series.index, series.values, title="Магазин — Weekly Sales")

    forecast_url = plot_forecast(
        train_dates=train.index,
        train_values=train.values,
        test_dates=test.index,
        test_actual=test.values,
        test_predicted=test_predicted,
        lower_ci=forecast_ci.iloc[: len(test), 0].values,
        upper_ci=forecast_ci.iloc[: len(test), 1].values,
        title=f"SARIMAX{order}x{seasonal_order}",
    )

    residuals = fitted_model.resid
    residuals_url = plot_residuals(residuals.values)

    plots = ForecastPlots(
        time_series_url=ts_url,
        forecast_url=forecast_url,
        residuals_url=residuals_url,
        acf_pacf_url=acf_pacf_url,
    )

    result = SarimaxResponse(
        stationarity=stationarity,
        order=list(order),
        seasonal_order=list(seasonal_order),
        train_forecast=train_forecast,
        test_forecast=test_forecast,
        metrics=metrics,
        plots=plots,
    )

    # Сохраняем в кэш
    if store_id is not None:
        _cache[(store_id, horizon)] = result

    return result
