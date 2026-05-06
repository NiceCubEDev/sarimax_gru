"""
Comparison report generation for SARIMAX and GRU.
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
        ("RMSE", f"{sarimax.rmse:.2f}", f"{gru.rmse:.2f}"),
        ("MAPE, %", f"{sarimax.mape:.2f}", f"{gru.mape:.2f}"),
        ("sMAPE, %", f"{sarimax.smape:.2f}", f"{gru.smape:.2f}"),
        ("WAPE, %", f"{sarimax.wape:.2f}", f"{gru.wape:.2f}"),
    ]


def _plot_overview_page(pdf: PdfPages, report: ComparisonReport) -> None:
    fig, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")

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

    validation = report.data_validation
    split = report.split
    axis.text(
        0.02,
        0.80,
        (
            "Проверка входных данных\n"
            f"Количество строк: {validation.row_count}\n"
            f"Период: {validation.start_date} - {validation.end_date}\n"
            f"Частота: {validation.frequency_days} дней\n"
            f"Пропуски: {validation.missing_values}\n"
            f"Дубликаты дат: {validation.duplicated_dates}"
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
            "Валидация: "
            f"{split.validation_size} "
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
            f"Размер скрытого слоя GRU: {report.gru.hyperparams.hidden_size}\n"
            f"Количество слоев GRU: {report.gru.hyperparams.num_layers}\n"
            f"Количество эпох GRU: {report.gru.hyperparams.selected_epochs}\n"
            f"Окно: {report.gru.hyperparams.sequence_length}"
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
    axis.plot(actual_dates, sarimax_df["actual"], label="Actual", color="#111827", linewidth=2.2)
    axis.plot(
        actual_dates,
        sarimax_df["predicted"],
        label="SARIMAX",
        color="#2563eb",
        linewidth=2.0,
    )
    axis.plot(actual_dates, gru_df["predicted"], label="GRU", color="#dc2626", linewidth=2.0)
    axis.set_title("Тестовая выборка: факт и прогноз", fontsize=11)
    axis.set_xlabel("Дата", fontsize=9)
    axis.set_ylabel("Недельные продажи", fontsize=9)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    labels = ["Факт" if label == "Actual" else label for label in labels]
    axis.legend(handles, labels, fontsize=8)
    fig.autofmt_xdate()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_error_page(pdf: PdfPages, report: ComparisonReport) -> None:
    sarimax_df = pd.DataFrame([point.model_dump() for point in report.sarimax.test_forecast])
    gru_df = pd.DataFrame([point.model_dump() for point in report.gru.test_forecast])
    dates = pd.to_datetime(sarimax_df["date"])

    sarimax_abs_error = (sarimax_df["actual"] - sarimax_df["predicted"]).abs()
    gru_abs_error = (gru_df["actual"] - gru_df["predicted"]).abs()

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    axes[0].plot(dates, sarimax_abs_error, label="SARIMAX absolute error", color="#2563eb")
    axes[0].plot(dates, gru_abs_error, label="GRU absolute error", color="#dc2626")
    axes[0].set_title("Абсолютная ошибка на тестовой выборке", fontsize=11)
    axes[0].set_ylabel("Абсолютная ошибка", fontsize=9)
    axes[0].tick_params(axis="both", labelsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].legend(
        ["Абсолютная ошибка SARIMAX", "Абсолютная ошибка GRU"],
        fontsize=8,
    )

    axes[1].plot(report.gru.training_losses, color="#ea580c")
    axes[1].set_title("Кривая обучения GRU на обучающей выборке", fontsize=11)
    axes[1].set_xlabel("Эпоха", fontsize=9)
    axes[1].set_ylabel("Функция потерь MSE", fontsize=9)
    axes[1].tick_params(axis="both", labelsize=8)
    axes[1].grid(alpha=0.3)

    fig.autofmt_xdate()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_comparison_report(store_df: pd.DataFrame, store_id: int) -> ComparisonReport:
    """Build an honest comparison report and save it as a single PDF."""
    min_rows = max(settings.gru_sequence_length + 20, settings.sarimax_seasonal_period + 10)
    validated_df, validation_summary = validate_store_dataframe(store_df, min_rows=min_rows)
    split = split_time_series(validated_df)

    sarimax_result = run_sarimax_pipeline(validated_df, split)
    gru_result = run_gru_pipeline(validated_df, split)

    reports_dir = settings.reports_full_path
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f"store_{store_id}_forecast_comparison.pdf"

    report = ComparisonReport(
        store_id=store_id,
        data_validation=validation_summary,
        split=split.summary,
        sarimax=sarimax_result,
        gru=gru_result,
        pdf_path=str(pdf_path.resolve()),
    )

    with PdfPages(pdf_path) as pdf:
        _plot_overview_page(pdf, report)
        _plot_test_page(pdf, report)
        _plot_error_page(pdf, report)

    return report
