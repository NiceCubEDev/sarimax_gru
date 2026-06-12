"""
Точка входа командной строки для генерации offline-отчета.
"""

import argparse
import json
import sys

from app.config import settings
from app.pipeline.data_loader import data_loader
from app.pipeline.reporting import generate_comparison_report
from app.schemas.forecast import ComparisonReport, ModelMetrics
from app.utils.progress import ProgressSpinner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запустить модели прогнозирования и сформировать PDF-отчет.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initial_methods = subparsers.add_parser(
        "initial-methods",
        help="Проверить данные, обучить SARIMAX и GRU, сравнить модели и создать PDF.",
    )
    initial_methods.add_argument(
        "--store-id",
        type=int,
        default=1,
        help="Идентификатор магазина из датасета Walmart.",
    )
    initial_methods.add_argument(
        "--horizon",
        type=int,
        default=settings.forecast_horizon,
        help="Количество будущих недель для прогноза после последней фактической даты.",
    )
    return parser


def _format_metrics(metrics: ModelMetrics) -> str:
    return (
        f"MAE={metrics.mae:.2f}, MSE={metrics.mse:.2f}, RMSE={metrics.rmse:.2f}, "
        f"MAPE={metrics.mape:.2f}%, sMAPE={metrics.smape:.2f}%, WAPE={metrics.wape:.2f}%"
    )


def _print_console_summary(report: ComparisonReport) -> None:
    stationarity = report.sarimax.stationarity
    print("\nДифференцирование SARIMAX", file=sys.stderr)
    print(f"ADF statistic: {stationarity.adf_statistic:.4f}", file=sys.stderr)
    print(f"p-value: {stationarity.p_value:.4f}", file=sys.stderr)
    print(
        f"Стационарность: {'да' if stationarity.is_stationary else 'нет'}",
        file=sys.stderr,
    )
    print(
        f"Выбранный порядок дифференцирования: d={stationarity.differencing_order}",
        file=sys.stderr,
    )

    print("\nМетрики на обучении", file=sys.stderr)
    print(f"SARIMAX: {_format_metrics(report.sarimax.training_metrics)}", file=sys.stderr)
    print(f"GRU:     {_format_metrics(report.gru.training_metrics)}", file=sys.stderr)

    print("\nМетрики на тесте", file=sys.stderr)
    print(f"SARIMAX: {_format_metrics(report.sarimax.test_metrics)}", file=sys.stderr)
    print(f"GRU:     {_format_metrics(report.gru.test_metrics)}", file=sys.stderr)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "initial-methods":
        parser.error(f"Unsupported command: {args.command}")

    progress_message = (
        f"Обработка данных магазина {args.store_id}: "
        "проверка данных, обучение SARIMAX/GRU и сборка PDF"
    )
    with ProgressSpinner(progress_message):
        report = generate_comparison_report(
            store_df=data_loader.get_store_data(args.store_id),
            store_id=args.store_id,
            horizon=args.horizon,
        )
    _print_console_summary(report)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
