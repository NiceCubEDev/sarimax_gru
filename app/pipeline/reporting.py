"""
Генерация PDF-отчета для сравнения SARIMAX и GRU.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from app.analytics.gru import run_gru_pipeline
from app.analytics.sarimax import run_sarimax_pipeline
from app.config import settings
from app.pipeline.forecasting import split_time_series, validate_store_dataframe
from app.schemas.forecast import ComparisonReport


plt.switch_backend("Agg")


def _metrics_rows(report: ComparisonReport) -> list[tuple[str, str, str]]:
    sarimax = report.sarimax.test_metrics
    gru = report.gru.test_metrics
    return [
        ("MAE", f"{sarimax.mae:.2f}", f"{gru.mae:.2f}"),
        ("MSE", f"{sarimax.mse:.2f}", f"{gru.mse:.2f}"),
        ("RMSE", f"{sarimax.rmse:.2f}", f"{gru.rmse:.2f}"),
        ("MAPE, %", f"{sarimax.mape:.2f}", f"{gru.mape:.2f}"),
        ("sMAPE, %", f"{sarimax.smape:.2f}", f"{gru.smape:.2f}"),
        ("WAPE, %", f"{sarimax.wape:.2f}", f"{gru.wape:.2f}"),
    ]


def _plot_overview_page(pdf: PdfPages, report: ComparisonReport) -> None:
    fig, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")

    validation = report.data_validation
    split = report.split
    axis.text(
        0.02,
        0.95,
        f"Отчет по сравнению прогнозных моделей для магазина {report.store_id}",
        fontsize=15,
        weight="bold",
    )
    axis.text(
        0.02,
        0.90,
        f"Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=9,
    )
    axis.text(
        0.02,
        0.80,
        (
            "Проверка входных данных\n"
            f"Количество строк: {validation.row_count}\n"
            f"Период: {validation.start_date} - {validation.end_date}\n"
            f"Частота: {validation.frequency_days} дней\n"
            f"Пропуски: {validation.missing_values}\n"
            f"Дубликаты дат: {validation.duplicated_dates}\n"
            f"Горизонт прогноза: {report.horizon} недель"
        ),
        fontsize=9,
        va="top",
    )
    axis.text(
        0.37,
        0.80,
        (
            "Хронологическое разбиение\n"
            f"Обучение: {split.train_size} ({split.train_start} - {split.train_end})\n"
            f"Валидация: {split.validation_size} "
            f"({split.validation_start} - {split.validation_end})\n"
            f"Тест: {split.test_size} ({split.test_start} - {split.test_end})"
        ),
        fontsize=9,
        va="top",
    )
    axis.text(
        0.72,
        0.80,
        (
            "Выбранные модели\n"
            f"SARIMAX: {tuple(report.sarimax.order)} x {tuple(report.sarimax.seasonal_order)}\n"
            f"GRU размер скрытого слоя: {report.gru.hyperparams.hidden_size}\n"
            f"GRU количество слоев: {report.gru.hyperparams.num_layers}\n"
            f"GRU выбранные эпохи: {report.gru.hyperparams.selected_epochs}\n"
            f"GRU длина окна: {report.gru.hyperparams.sequence_length}"
        ),
        fontsize=9,
        va="top",
    )

    table = axis.table(
        cellText=_metrics_rows(report),
        colLabels=["Метрика", "SARIMAX", "GRU"],
        cellLoc="center",
        loc="lower center",
        bbox=[0.1, 0.15, 0.8, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_test_page(pdf: PdfPages, report: ComparisonReport) -> None:
    sarimax_df = pd.DataFrame([point.model_dump() for point in report.sarimax.test_forecast])
    gru_df = pd.DataFrame([point.model_dump() for point in report.gru.test_forecast])
    actual_dates = pd.to_datetime(sarimax_df["date"])

    fig, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.plot(actual_dates, sarimax_df["actual"], label="Факт", color="#111827", linewidth=2.2)
    axis.plot(
        actual_dates,
        sarimax_df["predicted"],
        label="SARIMAX прогноз на тесте",
        color="#2563eb",
        linewidth=2.0,
    )
    axis.plot(
        actual_dates,
        gru_df["predicted"],
        label="GRU прогноз на тесте",
        color="#dc2626",
        linewidth=2.0,
    )
    axis.set_title("Тестовый период: факт и прогнозы моделей", fontsize=11)
    axis.set_xlabel("Дата", fontsize=9)
    axis.set_ylabel("Недельные продажи", fontsize=9)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.autofmt_xdate()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_future_page(pdf: PdfPages, report: ComparisonReport, store_df: pd.DataFrame) -> None:
    sarimax_history = pd.DataFrame(
        [point.model_dump() for point in report.sarimax.history_forecast],
    )
    gru_history = pd.DataFrame([point.model_dump() for point in report.gru.history_forecast])
    sarimax_future = pd.DataFrame(
        [point.model_dump() for point in report.sarimax.future_forecast],
    )
    gru_future = pd.DataFrame([point.model_dump() for point in report.gru.future_forecast])
    sarimax_history_dates = pd.to_datetime(sarimax_history["date"])
    gru_history_dates = pd.to_datetime(gru_history["date"])
    future_dates = pd.to_datetime(sarimax_future["date"])

    last_actual_date = store_df.index.max()
    last_actual_value = float(store_df["Weekly_Sales"].iloc[-1])
    test_start_date = pd.Timestamp(report.split.test_start)
    sarimax_future_dates = pd.DatetimeIndex([last_actual_date, *future_dates])
    sarimax_future_values = [last_actual_value, *sarimax_future["predicted"].to_list()]
    gru_future_dates = pd.DatetimeIndex([last_actual_date, *future_dates])
    gru_future_values = [last_actual_value, *gru_future["predicted"].to_list()]

    fig, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.plot(
        store_df.index,
        store_df["Weekly_Sales"],
        label="Фактическая история",
        color="#111827",
        linewidth=2.0,
    )
    axis.plot(
        sarimax_history_dates,
        sarimax_history["predicted"],
        label="SARIMAX модельная линия на истории",
        color="#2563eb",
        linewidth=2.0,
    )
    axis.plot(
        sarimax_future_dates,
        sarimax_future_values,
        label="SARIMAX будущий прогноз",
        color="#2563eb",
        linewidth=2.0,
        linestyle="--",
    )
    axis.plot(
        gru_history_dates,
        gru_history["predicted"],
        label="GRU модельная линия на истории",
        color="#dc2626",
        linewidth=2.0,
    )
    axis.plot(
        gru_future_dates,
        gru_future_values,
        label="GRU будущий прогноз",
        color="#dc2626",
        linewidth=2.0,
        linestyle="--",
    )
    axis.axvspan(
        test_start_date,
        last_actual_date,
        color="#f59e0b",
        alpha=0.08,
        label="Тестовый промежуток",
    )
    axis.axvspan(
        last_actual_date,
        future_dates.max(),
        color="#10b981",
        alpha=0.06,
        label="Будущий прогноз",
    )
    axis.axvline(
        test_start_date,
        color="#92400e",
        linewidth=1.4,
        linestyle=":",
        label="Начало тестового промежутка",
    )
    axis.axvline(
        last_actual_date,
        color="#6b7280",
        linewidth=1.4,
        linestyle=":",
        label="Начало будущего прогноза",
    )
    axis.text(
        0.99,
        0.03,
        (
            "Ошибки на тестовом промежутке\n"
            f"SARIMAX MAPE: {report.sarimax.test_metrics.mape:.2f}%\n"
            f"GRU MAPE: {report.gru.test_metrics.mape:.2f}%\n"
            f"SARIMAX WAPE: {report.sarimax.test_metrics.wape:.2f}%\n"
            f"GRU WAPE: {report.gru.test_metrics.wape:.2f}%\n"
            f"SARIMAX RMSE: {report.sarimax.test_metrics.rmse:,.0f}\n"
            f"GRU RMSE: {report.gru.test_metrics.rmse:,.0f}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.78},
    )
    axis.set_title("Факт, тестовые прогнозы и будущий прогноз", fontsize=11)
    axis.set_xlabel("Дата", fontsize=9)
    axis.set_ylabel("Недельные продажи", fontsize=9)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.autofmt_xdate()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_error_page(pdf: PdfPages, report: ComparisonReport) -> None:
    sarimax_df = pd.DataFrame([point.model_dump() for point in report.sarimax.test_forecast])
    gru_df = pd.DataFrame([point.model_dump() for point in report.gru.test_forecast])
    dates = pd.to_datetime(sarimax_df["date"])

    sarimax_abs_error = (sarimax_df["actual"] - sarimax_df["predicted"]).abs()
    gru_abs_error = (gru_df["actual"] - gru_df["predicted"]).abs()
    epochs = range(1, len(report.gru.training_losses) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    axes[0].plot(dates, sarimax_abs_error, label="Абсолютная ошибка SARIMAX", color="#2563eb")
    axes[0].plot(dates, gru_abs_error, label="Абсолютная ошибка GRU", color="#dc2626")
    axes[0].set_title("Абсолютная ошибка на тестовом периоде", fontsize=11)
    axes[0].set_xlabel("Дата", fontsize=9)
    axes[0].set_ylabel("Абсолютная ошибка", fontsize=9)
    axes[0].tick_params(axis="both", labelsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, report.gru.training_losses, color="#ea580c")
    axes[1].set_title("Функция потерь GRU при обучении", fontsize=11)
    axes[1].set_xlabel("Эпоха", fontsize=9)
    axes[1].set_ylabel("MSE loss", fontsize=9)
    axes[1].tick_params(axis="both", labelsize=8)
    axes[1].grid(alpha=0.3)

    fig.autofmt_xdate()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_comparison_report(
    store_df: pd.DataFrame,
    store_id: int,
    horizon: int,
) -> ComparisonReport:
    """Построить честное сравнение моделей и сохранить единый PDF-отчет."""
    min_rows = max(settings.gru_sequence_length + 20, settings.sarimax_seasonal_period + 10)
    validated_df, validation_summary = validate_store_dataframe(store_df, min_rows=min_rows)
    split = split_time_series(validated_df)

    sarimax_result = run_sarimax_pipeline(validated_df, split, horizon)
    gru_result = run_gru_pipeline(validated_df, split, horizon)

    reports_dir = settings.reports_full_path
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f"store_{store_id}_forecast_comparison.pdf"

    report = ComparisonReport(
        store_id=store_id,
        horizon=horizon,
        data_validation=validation_summary,
        split=split.summary,
        sarimax=sarimax_result,
        gru=gru_result,
        pdf_path=str(pdf_path.resolve()),
    )

    with PdfPages(pdf_path) as pdf:
        _plot_overview_page(pdf, report)
        _plot_test_page(pdf, report)
        _plot_future_page(pdf, report, validated_df)
        _plot_error_page(pdf, report)

    return report
