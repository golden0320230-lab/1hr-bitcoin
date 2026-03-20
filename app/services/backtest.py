"""Historical backtesting against naive baselines."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd  # type: ignore[import-untyped]

from app.schemas import BacktestMetric, BacktestResult
from app.services.storage import DuckDBStorage
from app.services.training import ModelTrainer


class BacktestService:
    """Evaluate the trained model against historical baseline strategies."""

    def __init__(self, storage: DuckDBStorage | None = None) -> None:
        self.storage = storage
        self._trainer = ModelTrainer()

    def run(self, dataset: pd.DataFrame) -> BacktestResult:
        if len(dataset) < 10:
            raise ValueError("At least 10 dataset rows are required to run a backtest.")

        ordered = dataset.sort_values("prediction_timestamp").reset_index(drop=True)
        test_size = max(int(len(ordered) * 0.2), 2)
        if test_size >= len(ordered):
            raise ValueError("Backtest dataset is too small for a holdout window.")

        split_index = len(ordered) - test_size
        train_dataset = ordered.iloc[:split_index]
        test_dataset = ordered.iloc[split_index:]

        with TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "backtest_model.pkl"
            training_result = self._trainer.train(train_dataset, artifact_path=artifact_path)
            artifact = self._trainer.load_artifact(artifact_path)
            full_probabilities = self._trainer.predict_dataset_probabilities(artifact, test_dataset)
            no_news_probabilities = self._trainer.predict_dataset_probabilities(
                artifact,
                self._neutralize_news_features(test_dataset),
            )

        labels = test_dataset["label"].astype(int)
        full_metrics = self._metrics(labels, full_probabilities)
        baselines = [
            self._baseline_metric(
                "kalshi_follow",
                labels,
                test_dataset["market_implied_probability"],
            ),
            self._baseline_metric(
                "spot_vs_strike",
                labels,
                self._spot_vs_strike_probabilities(test_dataset),
            ),
            self._baseline_metric(
                "momentum",
                labels,
                self._momentum_probabilities(test_dataset),
            ),
            self._baseline_metric("no_news", labels, no_news_probabilities),
        ]

        result = BacktestResult(
            generated_at=datetime.now(UTC),
            window_start=ModelTrainer._as_datetime(test_dataset["prediction_timestamp"].iloc[0]),
            window_end=ModelTrainer._as_datetime(test_dataset["prediction_timestamp"].iloc[-1]),
            model_name=training_result.model_name,
            num_samples=int(len(test_dataset)),
            accuracy=full_metrics["accuracy"],
            log_loss=full_metrics["log_loss"],
            brier_score=full_metrics["brier_score"],
            baselines=baselines,
            notes=[
                f"Training rows: {len(train_dataset)}",
                f"Evaluation rows: {len(test_dataset)}",
            ],
        )

        if self.storage is not None:
            self.storage.insert_backtest_result(result)

        return result

    @staticmethod
    def _neutralize_news_features(dataset: pd.DataFrame) -> pd.DataFrame:
        neutralized = dataset.copy()
        neutralized["news_weighted_impact"] = 0.0
        neutralized["news_weighted_bullish"] = 0.0
        neutralized["news_weighted_bearish"] = 0.0
        neutralized["high_confidence_article_count"] = 0
        neutralized["breaking_news"] = False
        return neutralized

    @staticmethod
    def _spot_vs_strike_probabilities(dataset: pd.DataFrame) -> list[float]:
        probabilities: list[float] = []
        for _, row in dataset.iterrows():
            probabilities.append(
                0.999 if float(row["spot_price"]) >= float(row["strike_price"]) else 0.001
            )
        return probabilities

    @staticmethod
    def _momentum_probabilities(dataset: pd.DataFrame) -> list[float]:
        probabilities: list[float] = []
        for _, row in dataset.iterrows():
            momentum = float(row["return_15m"])
            probabilities.append(min(max(0.5 + (momentum * 20), 0.001), 0.999))
        return probabilities

    @staticmethod
    def _baseline_metric(
        name: str,
        labels: pd.Series,
        probabilities: pd.Series | list[float],
    ) -> BacktestMetric:
        metrics = BacktestService._metrics(labels, probabilities)
        return BacktestMetric(
            name=name,
            accuracy=metrics["accuracy"],
            log_loss=metrics["log_loss"],
            brier_score=metrics["brier_score"],
        )

    @staticmethod
    def _metrics(
        labels: pd.Series,
        probabilities: pd.Series | list[float],
    ) -> dict[str, float]:
        series = [min(max(float(probability), 1e-6), 1 - 1e-6) for probability in probabilities]
        return ModelTrainer._metrics(labels, series)
