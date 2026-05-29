"""
Configuration for the offline forecasting pipeline.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline settings loaded from environment variables or .env."""

    # --- Data ---
    csv_path: str = "data/Walmart.csv"
    reports_dir: str = "build/reports"
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    random_seed: int = 42
    forecast_horizon: int = 12

    # --- SARIMAX defaults ---
    sarimax_seasonal_period: int = 52

    # --- GRU defaults ---
    gru_hidden_size: int = 64
    gru_num_layers: int = 2
    gru_epochs: int = 100
    gru_learning_rate: float = 0.001
    gru_sequence_length: int = 12

    @property
    def csv_full_path(self) -> Path:
        return Path(self.csv_path)

    @property
    def reports_full_path(self) -> Path:
        return Path(self.reports_dir)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
