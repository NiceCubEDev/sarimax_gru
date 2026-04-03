"""
Command-line entrypoint for offline report generation.
"""

import argparse
import json

from app.service.data_loader import data_loader
from app.service.reporting import generate_comparison_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run initial forecasting methods and generate a PDF report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initial_methods = subparsers.add_parser(
        "initial-methods",
        help="Validate data, train SARIMAX and GRU, compare them on test, generate PDF.",
    )
    initial_methods.add_argument(
        "--store-id",
        type=int,
        default=1,
        help="Store identifier from the Walmart dataset.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "initial-methods":
        parser.error(f"Unsupported command: {args.command}")

    report = generate_comparison_report(
        store_df=data_loader.get_store_data(args.store_id),
        store_id=args.store_id,
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
