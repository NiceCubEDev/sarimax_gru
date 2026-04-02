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

    # --- SARIMAX defaults ---
    sarimax_seasonal_period: int = 52  # недельная сезонность
    sarimax_train_ratio: float = 0.8
    sarimax_default_horizon: int = 12

    # --- GRU defaults ---
    gru_hidden_size: int = 64
    gru_num_layers: int = 2
    gru_epochs: int = 100
    gru_learning_rate: float = 0.001
    gru_sequence_length: int = 12  # окно (кол-во недель)
    gru_train_ratio: float = 0.8

    # --- Static files (graphs) ---
    plots_dir: str = "static/plots"

    @property
    def csv_full_path(self) -> Path:
        return Path(self.csv_path)

    @property
    def plots_full_path(self) -> Path:
        path = Path(self.plots_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
