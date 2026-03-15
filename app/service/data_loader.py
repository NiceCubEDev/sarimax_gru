"""
ClusterApp Backend — загрузка и кэширование данных Walmart.
"""

import pandas as pd

from app.config import settings


class DataLoader:
    """Синглтон для загрузки и кэширования CSV-данных."""

    _instance = None
    _df: pd.DataFrame | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> pd.DataFrame:
        """Загрузить данные из CSV (с кэшированием)."""
        if self._df is None:
            self._df = pd.read_csv(
                settings.csv_full_path,
                parse_dates=["Date"],
                dayfirst=True,
            )
            # Сортировка по магазину и дате
            self._df = self._df.sort_values(["Store", "Date"]).reset_index(drop=True)
        return self._df

    def get_store_data(self, store_id: int) -> pd.DataFrame:
        """Получить данные конкретного магазина."""
        df = self.load()
        store_df = df[df["Store"] == store_id].copy()
        if store_df.empty:
            raise ValueError(f"Магазин {store_id} не найден")
        store_df = store_df.set_index("Date").sort_index()
        return store_df

    def get_all_stores(self) -> list[int]:
        """Список всех ID магазинов."""
        df = self.load()
        return sorted(df["Store"].unique().tolist())

    def get_store_aggregates(self) -> pd.DataFrame:
        """Агрегированные метрики по магазинам."""
        df = self.load()
        agg = df.groupby("Store").agg(
            mean_sales=("Weekly_Sales", "mean"),
            std_sales=("Weekly_Sales", "std"),
            mean_temperature=("Temperature", "mean"),
            mean_cpi=("CPI", "mean"),
            mean_unemployment=("Unemployment", "mean"),
            mean_fuel_price=("Fuel_Price", "mean"),
        ).reset_index()
        agg = agg.rename(columns={"Store": "store_id"})
        return agg


data_loader = DataLoader()
