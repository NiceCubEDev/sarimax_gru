"""
Точка входа командной строки для генерации offline-отчета.
"""

import argparse
import json

from app.config import settings
from app.pipeline.data_loader import data_loader
from app.pipeline.reporting import generate_comparison_report
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
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
