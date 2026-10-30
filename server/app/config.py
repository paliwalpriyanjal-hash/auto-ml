# server/app/config.py
"""
Central configuration, security settings, and resource limits for AutoVeyra platform.
Provides environment-variable overrides with safe, laptop-friendly defaults.
"""

import os
from typing import List


class Settings:
    # Service Information
    APP_NAME: str = "AutoVeyra Tabular AutoML Platform"
    APP_VERSION: str = "2.0-PRO"
    ENVIRONMENT: str = os.getenv("AUTOVEYRA_ENV", "production").lower()

    # Authentication & Authorization Settings
    AUTH_ENABLED: bool = os.getenv("AUTOVEYRA_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    JWT_SECRET: str = os.getenv("AUTOVEYRA_JWT_SECRET", "autoveyra-default-jwt-secret-key-32chars!!")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = int(os.getenv("AUTOVEYRA_JWT_EXPIRATION_HOURS", 24))

    # Ingestion & Upload Resource Ceilings
    # 25 MB default maximum file size
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("AUTOVEYRA_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
    # 100,000 default max tabular rows
    MAX_DATASET_ROWS: int = int(os.getenv("AUTOVEYRA_MAX_DATASET_ROWS", 100_000))
    # 500 default max feature columns
    MAX_DATASET_COLS: int = int(os.getenv("AUTOVEYRA_MAX_DATASET_COLS", 500))

    # Inference & Serving Ceilings
    # Max rows allowed in a single predict batch payload
    MAX_PREDICTION_ROWS: int = int(os.getenv("AUTOVEYRA_MAX_PREDICTION_ROWS", 5_000))
    # In-memory LRU model serving cache size
    MODEL_CACHE_SIZE: int = int(os.getenv("AUTOVEYRA_MODEL_CACHE_SIZE", 8))

    # Compute & Optimization Ceilings
    MAX_TUNING_TRIALS: int = int(os.getenv("AUTOVEYRA_MAX_TUNING_TRIALS", 100))
    MAX_CV_SPLITS: int = int(os.getenv("AUTOVEYRA_MAX_CV_SPLITS", 10))
    MAX_EXPLAIN_SAMPLES: int = int(os.getenv("AUTOVEYRA_MAX_EXPLAIN_SAMPLES", 500))

    # Monitoring & Telemetry Ceilings
    MAX_MONITORING_RECORDS: int = int(os.getenv("AUTOVEYRA_MAX_MONITORING_RECORDS", 50_000))

    # CORS Allowed Origins
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        default_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://auto-ml-studio-phi.vercel.app",
            "https://autoveyra-web.onrender.com",
        ]
        raw = os.getenv("AUTOVEYRA_ALLOWED_ORIGINS")
        if raw:
            custom_origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
            return list(dict.fromkeys(default_origins + custom_origins))
        return default_origins


settings = Settings()
