"""
ClusterApp Backend — GRU (Gated Recurrent Unit) аналитический модуль.

Пайплайн:
1. Подготовка данных (многомерный вход, MinMaxScaler)
2. Создание скользящих окон (X, y)
3. Обучение GRU-модели (PyTorch)
4. Постпрогноз (in-sample)
5. Прогноз (out-of-sample)
6. Метрики (MAE, RMSE, MAPE)
7. Графики
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from app.config import settings
from app.schemas.forecast import (
    ForecastPlots,
    ForecastPoint,
    GruHyperparams,
    GruResponse,
)
from app.utils.metrics import compute_metrics
from app.utils.plots import plot_forecast, plot_time_series, plot_training_loss


# Фичи для многомерного входа
FEATURE_COLUMNS = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Holiday_Flag",
]
TARGET_COLUMN = "Weekly_Sales"

# Кэш результатов: ключ (store_id, horizon) → GruResponse
_cache: dict[tuple[int, int], GruResponse] = {}


class GRUModel(nn.Module):
    """GRU-модель для прогнозирования временных рядов."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        # Берём выход последнего шага
        out = self.fc(out[:, -1, :])
        return out


def _create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Создание скользящих окон для обучения."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)


def _train_model(
    model: GRUModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    learning_rate: float,
    patience: int = 15,
) -> list[float]:
    """Обучение GRU с early stopping. Возвращает список losses."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    best_loss = float("inf")
    patience_counter = 0

    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return losses


def run_gru_pipeline(
    store_df: pd.DataFrame,
    horizon: int = 12,
    store_id: int | None = None,
) -> GruResponse:
    """
    Полный GRU-пайплайн для одного магазина.
    Многомерный вход: Weekly_Sales + Temperature + Fuel_Price + CPI + Unemployment + Holiday_Flag.
    Результат кэшируется по (store_id, horizon).
    """
    # Проверяем кэш
    if store_id is not None:
        cache_key = (store_id, horizon)
        if cache_key in _cache:
            return _cache[cache_key]

    seq_length = settings.gru_sequence_length
    train_ratio = settings.gru_train_ratio

    # --- 1. Подготовка данных ---
    df = store_df[FEATURE_COLUMNS].copy()
    dates = store_df.index

    # Нормализация
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(df.values)
    scaled_target = target_scaler.fit_transform(
        df[[TARGET_COLUMN]].values,
    )

    # --- 2. Скользящие окна ---
    x_all, y_all = _create_sequences(scaled_features, scaled_target.ravel(), seq_length)

    # Train/test split
    split_idx = int(len(x_all) * train_ratio)
    x_train, x_test = x_all[:split_idx], x_all[split_idx:]
    y_train = y_all[:split_idx]
    # y_test не используется — test_actual берётся из исходных данных

    # В тензоры
    x_train_t = torch.FloatTensor(x_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    x_test_t = torch.FloatTensor(x_test)

    # --- 3. Обучение ---
    model = GRUModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=settings.gru_hidden_size,
        num_layers=settings.gru_num_layers,
    )

    losses = _train_model(
        model,
        x_train_t,
        y_train_t,
        epochs=settings.gru_epochs,
        learning_rate=settings.gru_learning_rate,
    )

    # --- 4. Предсказания ---
    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(x_train_t).numpy().ravel()
        test_pred_scaled = model(x_test_t).numpy().ravel()

    # Обратное масштабирование
    train_pred = target_scaler.inverse_transform(
        train_pred_scaled.reshape(-1, 1),
    ).ravel()
    test_pred = target_scaler.inverse_transform(
        test_pred_scaled.reshape(-1, 1),
    ).ravel()

    # Реальные значения (без масштабирования)
    # Даты сдвинуты на seq_length (первое окно)
    train_dates = dates[seq_length : seq_length + len(train_pred)]
    test_dates = dates[seq_length + split_idx : seq_length + split_idx + len(test_pred)]

    train_actual = df[TARGET_COLUMN].values[seq_length : seq_length + len(train_pred)]
    test_actual = df[TARGET_COLUMN].values[
        seq_length + split_idx : seq_length + split_idx + len(test_pred)
    ]

    # --- 5. Формирование ответа ---
    train_forecast = [
        ForecastPoint(
            date=(
                str(train_dates[i].date())
                if hasattr(train_dates[i], "date")
                else str(train_dates[i])
            ),
            actual=float(train_actual[i]),
            predicted=float(train_pred[i]),
        )
        for i in range(len(train_pred))
    ]

    test_forecast = [
        ForecastPoint(
            date=(
                str(test_dates[i].date())
                if hasattr(test_dates[i], "date")
                else str(test_dates[i])
            ),
            actual=float(test_actual[i]),
            predicted=float(test_pred[i]),
        )
        for i in range(len(test_pred))
    ]

    # --- 6. Прогноз в будущее (horizon шагов) ---
    last_sequence = scaled_features[-seq_length:]
    future_preds = []
    current_seq = torch.FloatTensor(last_sequence).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            pred = model(current_seq)
            pred_value = pred.item()
            future_preds.append(pred_value)

            # Сдвигаем окно: убираем первый шаг, добавляем новый
            new_row = current_seq[0, -1, :].clone()
            new_row[0] = pred_value  # Weekly_Sales = предсказание
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                new_row.unsqueeze(0).unsqueeze(0),
            ], dim=1)

    # Обратное масштабирование будущих предсказаний
    future_pred_values = target_scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1),
    ).ravel()

    # Генерация будущих дат (еженедельно)
    last_date = dates[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="W")[1:]

    for i in range(horizon):
        test_forecast.append(
            ForecastPoint(
                date=str(future_dates[i].date()),
                actual=None,
                predicted=float(future_pred_values[i]),
            )
        )

    # --- 7. Метрики (на test) ---
    metrics = compute_metrics(test_actual, test_pred)

    # --- 8. Графики ---
    series = store_df[TARGET_COLUMN]
    ts_url = plot_time_series(
        series.index, series.values, title="Магазин — Weekly Sales",
    )

    forecast_url = plot_forecast(
        train_dates=train_dates,
        train_values=train_actual,
        test_dates=test_dates,
        test_actual=test_actual,
        test_predicted=test_pred,
        title="GRU — прогноз",
    )

    training_loss_url = plot_training_loss(losses)

    plots = ForecastPlots(
        time_series_url=ts_url,
        forecast_url=forecast_url,
        training_loss_url=training_loss_url,
    )

    hyperparams = GruHyperparams(
        hidden_size=settings.gru_hidden_size,
        num_layers=settings.gru_num_layers,
        epochs_trained=len(losses),
        learning_rate=settings.gru_learning_rate,
        sequence_length=seq_length,
        features=FEATURE_COLUMNS,
    )

    result = GruResponse(
        hyperparams=hyperparams,
        train_forecast=train_forecast,
        test_forecast=test_forecast,
        metrics=metrics,
        plots=plots,
    )

    # Сохраняем в кэш
    if store_id is not None:
        _cache[(store_id, horizon)] = result

    return result
