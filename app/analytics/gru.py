"""
Honest GRU pipeline with train/validation/test evaluation.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from app.config import settings
from app.pipeline.forecasting import (
    REQUIRED_COLUMNS,
    TimeSeriesSplit,
    build_future_dates,
    build_future_feature_frame,
)
from app.schemas.forecast import ForecastPoint, GruHyperparams, GruResult
from app.utils.metrics import compute_metrics


DATE_FEATURE_COLUMNS = [
    "Week_Sin",
    "Week_Cos",
]
FEATURE_COLUMNS = [*REQUIRED_COLUMNS, *DATE_FEATURE_COLUMNS]
TARGET_COLUMN = "Weekly_Sales"


class GRUModel(nn.Module):
    """GRU model for one-step forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.gru(x)
        return self.fc(output[:, -1, :])


def _make_prediction_points(
    dates: pd.Index,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[ForecastPoint]:
    return [
        ForecastPoint(
            date=str(dates[position].date()),
            actual=float(actual[position]),
            predicted=float(predicted[position]),
        )
        for position in range(len(actual))
    ]


def _make_future_prediction_points(
    dates: pd.Index,
    predicted: np.ndarray,
) -> list[ForecastPoint]:
    return [
        ForecastPoint(
            date=str(dates[position].date()),
            predicted=float(predicted[position]),
        )
        for position in range(len(predicted))
    ]


def _with_date_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched_df = df.copy()
    weeks = enriched_df.index.isocalendar().week.astype(int).to_numpy()
    seasonal_weeks = np.minimum(weeks, 52)
    week_angle = 2 * np.pi * (seasonal_weeks - 1) / 52
    enriched_df["Week_Sin"] = np.sin(week_angle)
    enriched_df["Week_Cos"] = np.cos(week_angle)
    return enriched_df


def _build_window_dataset(
    features: np.ndarray,
    target: np.ndarray,
    start_target_idx: int,
    end_target_idx: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for target_idx in range(start_target_idx, end_target_idx):
        start_idx = target_idx - sequence_length
        if start_idx < 0:
            continue
        xs.append(features[start_idx:target_idx])
        ys.append(target[target_idx])
    if not xs:
        raise ValueError("Not enough observations to build sequences for the requested split")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _to_tensor_pair(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def _train_with_validation(
    model: GRUModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    epochs: int,
    learning_rate: float,
    patience: int = 15,
) -> tuple[list[float], int, dict[str, torch.Tensor]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()
        history.append(float(train_loss.item()))

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(x_validation), y_validation).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise ValueError("GRU validation training did not produce a valid checkpoint")

    return history, best_epoch, best_state


def _train_for_epochs(
    model: GRUModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()


def _predict(model: GRUModel, x_data: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x_data).cpu().numpy().ravel()


def _recursive_future_forecast(
    model: GRUModel,
    historical_feature_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    future_features: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    sequence_length: int,
) -> np.ndarray:
    history_features = list(
        feature_scaler.transform(historical_feature_df[FEATURE_COLUMNS].to_numpy(dtype=float)),
    )
    predictions: list[float] = []

    for forecast_date in future_dates:
        window = np.asarray(history_features[-sequence_length:], dtype=np.float32)[np.newaxis, :, :]
        predicted_scaled = _predict(model, torch.tensor(window, dtype=torch.float32))[0]
        predicted = float(target_scaler.inverse_transform([[predicted_scaled]])[0][0])
        predictions.append(predicted)

        future_row = future_features.loc[forecast_date].to_dict()
        future_row[TARGET_COLUMN] = predicted
        ordered_row = np.asarray([[future_row[column] for column in FEATURE_COLUMNS]], dtype=float)
        history_features.append(feature_scaler.transform(ordered_row)[0])

    return np.asarray(predictions, dtype=float)


def run_gru_pipeline(store_df: pd.DataFrame, split: TimeSeriesSplit, horizon: int) -> GruResult:
    """Train, validate and test GRU without target or scaler leakage."""
    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    sequence_length = settings.gru_sequence_length
    feature_df = _with_date_features(store_df)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = feature_df.iloc[: split.train_end_idx][FEATURE_COLUMNS].to_numpy(dtype=float)
    train_target = split.train[[TARGET_COLUMN]].to_numpy(dtype=float)
    feature_scaler.fit(train_features)
    target_scaler.fit(train_target)

    all_features = feature_scaler.transform(feature_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    all_target = target_scaler.transform(store_df[[TARGET_COLUMN]].to_numpy(dtype=float)).ravel()
    all_actual = store_df[TARGET_COLUMN].to_numpy(dtype=float)
    all_dates = store_df.index

    x_train, y_train = _build_window_dataset(
        all_features,
        all_target,
        start_target_idx=sequence_length,
        end_target_idx=split.train_end_idx,
        sequence_length=sequence_length,
    )
    x_validation, y_validation = _build_window_dataset(
        all_features,
        all_target,
        start_target_idx=split.train_end_idx,
        end_target_idx=split.validation_end_idx,
        sequence_length=sequence_length,
    )
    x_train_validation, y_train_validation = _build_window_dataset(
        all_features,
        all_target,
        start_target_idx=sequence_length,
        end_target_idx=split.validation_end_idx,
        sequence_length=sequence_length,
    )
    x_test, y_test = _build_window_dataset(
        all_features,
        all_target,
        start_target_idx=split.validation_end_idx,
        end_target_idx=len(store_df),
        sequence_length=sequence_length,
    )

    x_train_t, y_train_t = _to_tensor_pair(x_train, y_train)
    x_validation_t, y_validation_t = _to_tensor_pair(x_validation, y_validation)
    x_train_validation_t, y_train_validation_t = _to_tensor_pair(
        x_train_validation,
        y_train_validation,
    )
    x_test_t, _y_test_t = _to_tensor_pair(x_test, y_test)

    model = GRUModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=settings.gru_hidden_size,
        num_layers=settings.gru_num_layers,
    )
    losses, selected_epochs, best_state = _train_with_validation(
        model,
        x_train_t,
        y_train_t,
        x_validation_t,
        y_validation_t,
        epochs=settings.gru_epochs,
        learning_rate=settings.gru_learning_rate,
    )
    model.load_state_dict(best_state)

    validation_pred_scaled = _predict(model, x_validation_t)

    final_model = GRUModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=settings.gru_hidden_size,
        num_layers=settings.gru_num_layers,
    )
    _train_for_epochs(
        final_model,
        x_train_validation_t,
        y_train_validation_t,
        epochs=selected_epochs,
        learning_rate=settings.gru_learning_rate,
    )
    test_pred_scaled = _predict(final_model, x_test_t)

    future_feature_scaler = MinMaxScaler()
    future_target_scaler = MinMaxScaler()
    future_feature_scaler.fit(feature_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    future_target_scaler.fit(store_df[[TARGET_COLUMN]].to_numpy(dtype=float))
    full_features = future_feature_scaler.transform(
        feature_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    )
    full_target = future_target_scaler.transform(
        store_df[[TARGET_COLUMN]].to_numpy(dtype=float)
    ).ravel()
    x_full, y_full = _build_window_dataset(
        full_features,
        full_target,
        start_target_idx=sequence_length,
        end_target_idx=len(store_df),
        sequence_length=sequence_length,
    )
    x_full_t, y_full_t = _to_tensor_pair(x_full, y_full)
    future_model = GRUModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=settings.gru_hidden_size,
        num_layers=settings.gru_num_layers,
    )
    future_epochs = max(selected_epochs, min(settings.gru_future_min_epochs, settings.gru_epochs))
    _train_for_epochs(
        future_model,
        x_full_t,
        y_full_t,
        epochs=future_epochs,
        learning_rate=settings.gru_learning_rate,
    )
    history_pred_scaled = _predict(future_model, x_full_t)
    history_pred = future_target_scaler.inverse_transform(
        history_pred_scaled.reshape(-1, 1),
    ).ravel()
    history_pred = np.concatenate([all_actual[:sequence_length], history_pred])
    history_actual = all_actual
    history_dates = all_dates
    future_dates = build_future_dates(store_df.index.max(), horizon)
    future_features = _with_date_features(build_future_feature_frame(store_df, future_dates))
    future_pred = _recursive_future_forecast(
        future_model,
        feature_df,
        future_dates,
        future_features,
        future_feature_scaler,
        future_target_scaler,
        sequence_length,
    )

    validation_pred = target_scaler.inverse_transform(validation_pred_scaled.reshape(-1, 1)).ravel()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

    validation_actual = all_actual[split.train_end_idx : split.validation_end_idx]
    test_actual = all_actual[split.validation_end_idx :]
    validation_dates = all_dates[split.train_end_idx : split.validation_end_idx]
    test_dates = all_dates[split.validation_end_idx :]

    validation_metrics = compute_metrics(validation_actual, validation_pred)
    test_metrics = compute_metrics(test_actual, test_pred)

    hyperparams = GruHyperparams(
        hidden_size=settings.gru_hidden_size,
        num_layers=settings.gru_num_layers,
        learning_rate=settings.gru_learning_rate,
        sequence_length=sequence_length,
        selected_epochs=selected_epochs,
        future_epochs=future_epochs,
        features=FEATURE_COLUMNS,
    )

    return GruResult(
        hyperparams=hyperparams,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        history_forecast=_make_prediction_points(history_dates, history_actual, history_pred),
        validation_forecast=_make_prediction_points(
            validation_dates,
            validation_actual,
            validation_pred,
        ),
        test_forecast=_make_prediction_points(test_dates, test_actual, test_pred),
        future_forecast=_make_future_prediction_points(future_dates, future_pred),
        training_losses=losses,
    )
