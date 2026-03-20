"""Explain prediction history and saved runs."""

from __future__ import annotations

from app.schemas import PredictionResult
from app.services.storage import DuckDBStorage


class ExplainService:
    """Retrieve persisted prediction runs for inspection."""

    def __init__(self, storage: DuckDBStorage) -> None:
        self.storage = storage

    def get_last_prediction(self) -> PredictionResult | None:
        return self.storage.get_latest_prediction_run()
