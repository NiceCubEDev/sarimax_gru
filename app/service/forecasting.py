"""
Shared preprocessing utilities for honest time-series evaluation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import settings
from app.schemas.forecast import DataSplitSummary, DataValidationSummary


REQUIRED_COLUMNS = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Holiday_Flag",
]


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_idx: int
    validation_end_idx: int
    summary: DataSplitSummary


def validate_store_dataframe(
    store_df: pd.DataFrame,
    min_rows: int,
) -> tuple[pd.DataFrame, DataValidationSummary]:
    """Validate and normalize a store-level time series."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in store_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if not isinstance(store_df.index, pd.DatetimeIndex):
        raise ValueError("Date index must be a DatetimeIndex")

    df = store_df.sort_index().copy()
    duplicated_dates = int(df.index.duplicated().sum())
    if duplicated_dates:
        raise ValueError("Input data contains duplicated dates")

    missing_values = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    if missing_values:
        raise ValueError("Input data contains missing values in required columns")

    numeric_values = df[REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Input data contains non-finite values")

    if len(df) < min_rows:
        raise ValueError(
            f"Not enough observations: got {len(df)}, need at least {min_rows}",
        )

    diffs = df.index.to_series().diff().dropna().dt.days
    if diffs.empty:
        frequency_days = 7
    else:
        frequency_days = int(diffs.mode().iloc[0])
        if not (diffs == frequency_days).all():
            raise ValueError("Input series must have a regular weekly frequency")

    summary = DataValidationSummary(
        row_count=len(df),
        start_date=str(df.index.min().date()),
        end_date=str(df.index.max().date()),
        frequency_days=frequency_days,
        missing_values=missing_values,
        duplicated_dates=duplicated_dates,
    )
    return df, summary


def split_time_series(df: pd.DataFrame) -> TimeSeriesSplit:
    """Perform a chronological train/validation/test split."""
    ratios = settings.train_ratio + settings.validation_ratio + settings.test_ratio
    if abs(ratios - 1.0) > 1e-9:
        raise ValueError("Train, validation and test ratios must sum to 1.0")

    total_size = len(df)
    train_size = int(total_size * settings.train_ratio)
    validation_size = int(total_size * settings.validation_ratio)
    test_size = total_size - train_size - validation_size

    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("Each split must contain at least one observation")

    train = df.iloc[:train_size].copy()
    validation = df.iloc[train_size : train_size + validation_size].copy()
    test = df.iloc[train_size + validation_size :].copy()

    summary = DataSplitSummary(
        train_size=len(train),
        validation_size=len(validation),
        test_size=len(test),
        train_start=str(train.index.min().date()),
        train_end=str(train.index.max().date()),
        validation_start=str(validation.index.min().date()),
        validation_end=str(validation.index.max().date()),
        test_start=str(test.index.min().date()),
        test_end=str(test.index.max().date()),
    )

    return TimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
        train_end_idx=train_size,
        validation_end_idx=train_size + validation_size,
        summary=summary,
    )
