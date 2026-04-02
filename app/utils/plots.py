import uuid

import matplotlib


matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from app.config import settings


def _save_figure(fig: plt.Figure, prefix: str) -> str:
    """Сохранить график и вернуть URL."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = settings.plots_full_path / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"/static/plots/{filename}"


def plot_time_series(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    title: str = "Временной ряд Weekly Sales",
) -> str:
    """График исходного временного ряда."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, values, color="#2196F3", linewidth=1.2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Weekly Sales")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return _save_figure(fig, "timeseries")


def plot_forecast(
    train_dates: pd.DatetimeIndex,
    train_values: np.ndarray,
    test_dates: pd.DatetimeIndex,
    test_actual: np.ndarray,
    test_predicted: np.ndarray,
    lower_ci: np.ndarray | None = None,
    upper_ci: np.ndarray | None = None,
    title: str = "Прогноз SARIMAX",
) -> str:
    """График прогноза: факт vs предсказание."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Train
    ax.plot(train_dates, train_values, color="#2196F3", linewidth=1, label="Train (факт)")

    # Test — факт
    ax.plot(test_dates, test_actual, color="#4CAF50", linewidth=1.5, label="Test (факт)")

    # Test — прогноз
    ax.plot(
        test_dates, test_predicted, color="#FF5722", linewidth=1.5, linestyle="--", label="Прогноз"
    )

    # Доверительный интервал
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(test_dates, lower_ci, upper_ci, color="#FF5722", alpha=0.15, label="95% ДИ")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Weekly Sales")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return _save_figure(fig, "forecast")


def plot_residuals(residuals: np.ndarray, title: str = "Остатки модели") -> str:
    """График остатков."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(residuals, color="#9C27B0", linewidth=0.8)
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title(f"{title} — временной ряд")
    axes[0].set_ylabel("Остаток")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, color="#9C27B0", edgecolor="white", alpha=0.7)
    axes[1].set_title(f"{title} — гистограмма")
    axes[1].set_xlabel("Остаток")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_figure(fig, "residuals")


def plot_acf_pacf(series: np.ndarray, lags: int = 40, title: str = "ACF / PACF") -> str:
    """Графики ACF и PACF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(series, lags=lags, ax=axes[0], title=f"{title} — ACF")
    plot_pacf(series, lags=lags, ax=axes[1], title=f"{title} — PACF", method="ywm")

    fig.tight_layout()
    return _save_figure(fig, "acf_pacf")


def plot_training_loss(
    losses: list[float],
    title: str = "Кривая обучения GRU",
) -> str:
    """График loss по эпохам."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, color="#9C27B0", linewidth=1.5, label="Train Loss (MSE)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, "training_loss")

