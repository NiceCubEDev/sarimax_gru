"""
ClusterApp Backend — конфигурация приложения.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения через переменные окружения."""

    # --- Application ---
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-to-random-secret-key"

    # --- Data ---
    csv_path: str = "data/Walmart.csv"
    reports_dir: str = "build/reports"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # --- SARIMAX defaults ---
    sarimax_seasonal_period: int = 52  # недельная сезонность

    # --- GRU defaults ---
    gru_hidden_size: int = 64
    gru_num_layers: int = 2
    gru_epochs: int = 100
    gru_learning_rate: float = 0.001
    gru_sequence_length: int = 12  # окно (кол-во недель)

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
