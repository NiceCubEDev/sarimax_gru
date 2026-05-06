"""
CSV loading for the offline Walmart forecasting pipeline.
"""

import pandas as pd

from app.config import settings


class DataLoader:
    """Singleton CSV loader with in-memory caching."""

    _instance = None
    _df: pd.DataFrame | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> pd.DataFrame:
        """Load the configured CSV once and return the cached DataFrame."""
        if self._df is None:
            self._df = pd.read_csv(
                settings.csv_full_path,
                parse_dates=["Date"],
                dayfirst=True,
            )
            self._df = self._df.sort_values(["Store", "Date"]).reset_index(drop=True)
        return self._df

    def get_store_data(self, store_id: int) -> pd.DataFrame:
        """Return a date-indexed time series for one store."""
        df = self.load()
        store_df = df[df["Store"] == store_id].copy()
        if store_df.empty:
            raise ValueError(f"Store {store_id} was not found")
        store_df = store_df.set_index("Date").sort_index()
        return store_df


data_loader = DataLoader()
