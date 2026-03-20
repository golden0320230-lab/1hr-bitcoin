"""Training dataset generation aligned to live Kalshi BTC market semantics."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from app.schemas import BTCCandle, FeatureVector, KalshiMarket, MarketSnapshot
from app.services.features import FeatureBuilder
from app.services.storage import DuckDBStorage

DEFAULT_DATASET_PATH = Path("data/training_dataset.csv")
DEFAULT_MIN_TRAINING_ROWS = 10
DATASET_TIMESTAMP_COLUMNS = (
    "prediction_timestamp",
    "expiry_time",
    "settlement_timestamp",
    "generated_at",
)
FEATURE_COLUMNS = (
    "spot_price",
    "strike_price",
    "distance_to_strike",
    "distance_to_strike_pct",
    "kalshi_yes_price",
    "kalshi_no_price",
    "market_implied_probability",
    "spread",
    "orderbook_imbalance",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "realized_vol_15m",
    "realized_vol_60m",
    "ma_deviation",
    "momentum_slope",
    "rsi",
    "minutes_to_expiry",
    "news_weighted_impact",
    "news_weighted_bullish",
    "news_weighted_bearish",
    "high_confidence_article_count",
    "breaking_news",
)
DATASET_COLUMNS = (
    "prediction_timestamp",
    "expiry_time",
    "settlement_timestamp",
    "market_ticker",
    "threshold",
    "settlement_price",
    "label",
    "feature_schema_version",
    *FEATURE_COLUMNS,
)


@dataclass(slots=True)
class TrainingDatasetArtifact:
    """Saved dataset metadata returned by the builder."""

    path: Path
    row_count: int
    feature_schema_version: str


@dataclass(slots=True)
class LoadedModelArtifact:
    """Loaded model artifact used for offline inference."""

    path: Path
    model_name: str
    model: Any
    feature_columns: tuple[str, ...]
    feature_schema_version: str
    created_at: datetime
    metrics: dict[str, dict[str, float]]
    training_window_start: datetime
    training_window_end: datetime


@dataclass(slots=True)
class TrainingRunResult:
    """Summary of a completed training run."""

    artifact_path: Path
    dataset_rows: int
    model_name: str
    feature_schema_version: str
    metrics: dict[str, dict[str, float]]
    training_window_start: datetime
    training_window_end: datetime


class TrainingDatasetBuilder:
    """Build historical training examples for above-or-below-at-expiry prediction."""

    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder | None = None,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        strike_increment: float = 0.01,
        history_minutes: int = 60,
    ) -> None:
        if strike_increment <= 0:
            raise ValueError("strike_increment must be positive.")
        if history_minutes <= 0:
            raise ValueError("history_minutes must be positive.")

        self._feature_builder = feature_builder or FeatureBuilder()
        self.dataset_path = Path(dataset_path)
        self.strike_increment = strike_increment
        self.history_minutes = history_minutes

    def build_dataset(
        self,
        candles: list[BTCCandle],
        *,
        horizon_minutes: int = 15,
        step_candles: int = 1,
    ) -> pd.DataFrame:
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive.")
        if step_candles <= 0:
            raise ValueError("step_candles must be positive.")
        if not candles:
            raise ValueError("At least one candle is required.")

        ordered = sorted(candles, key=lambda candle: candle.timestamp)
        rows: list[dict[str, Any]] = []

        for index in range(0, len(ordered), step_candles):
            anchor_candle = ordered[index]
            history = self._history_window(ordered[: index + 1], anchor_candle.timestamp)
            if history is None:
                continue

            expiry_candle = self._future_candle(
                ordered[index + 1 :],
                anchor_candle.timestamp,
                horizon_minutes=horizon_minutes,
            )
            if expiry_candle is None:
                continue

            threshold = self._normalize_threshold(anchor_candle.close)
            market = self._synthetic_market(
                anchor_time=anchor_candle.timestamp,
                expiry_time=expiry_candle.timestamp,
                threshold=threshold,
            )
            snapshot = self._synthetic_snapshot(
                market=market,
                spot_price=anchor_candle.close,
                captured_at=anchor_candle.timestamp,
            )
            features = self._feature_builder.build_feature_vector(
                market=market,
                snapshot=snapshot,
                spot_price=anchor_candle.close,
                candles=history,
                news_scores=[],
                generated_at=anchor_candle.timestamp,
            )
            rows.append(
                self._dataset_row(
                    features=features,
                    threshold=threshold,
                    expiry_candle=expiry_candle,
                )
            )

        if not rows:
            return pd.DataFrame(columns=list(DATASET_COLUMNS))

        dataset = pd.DataFrame(rows)
        return dataset.loc[:, list(DATASET_COLUMNS)]

    def save_dataset(
        self,
        dataset: pd.DataFrame,
        *,
        output_path: str | Path | None = None,
    ) -> TrainingDatasetArtifact:
        destination = Path(output_path) if output_path is not None else self.dataset_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(destination, index=False)
        schema_version = (
            "unknown"
            if dataset.empty
            else str(dataset["feature_schema_version"].iloc[0])
        )
        return TrainingDatasetArtifact(
            path=destination,
            row_count=int(len(dataset)),
            feature_schema_version=schema_version,
        )

    @staticmethod
    def load_dataset(path: str | Path) -> pd.DataFrame:
        dataset = pd.read_csv(path)
        for column in DATASET_TIMESTAMP_COLUMNS:
            if column in dataset.columns:
                dataset[column] = pd.to_datetime(dataset[column], utc=True)
        return dataset

    def _history_window(
        self,
        candles: list[BTCCandle],
        anchor_time: datetime,
    ) -> list[BTCCandle] | None:
        cutoff = anchor_time - timedelta(minutes=self.history_minutes)
        window = [candle for candle in candles if candle.timestamp >= cutoff]
        if not window or window[0].timestamp > cutoff:
            return None
        return window

    @staticmethod
    def _future_candle(
        candles: list[BTCCandle],
        anchor_time: datetime,
        *,
        horizon_minutes: int,
    ) -> BTCCandle | None:
        target = anchor_time + timedelta(minutes=horizon_minutes)
        for candle in candles:
            if candle.timestamp >= target:
                return candle
        return None

    def _normalize_threshold(self, price: float) -> float:
        return round(round(price / self.strike_increment) * self.strike_increment, 2)

    @staticmethod
    def _dataset_row(
        *,
        features: FeatureVector,
        threshold: float,
        expiry_candle: BTCCandle,
    ) -> dict[str, Any]:
        label = int(expiry_candle.close >= threshold)
        return {
            "prediction_timestamp": features.generated_at.isoformat(),
            "expiry_time": (features.generated_at + timedelta(minutes=features.minutes_to_expiry))
            .astimezone(UTC)
            .isoformat(),
            "settlement_timestamp": expiry_candle.timestamp.isoformat(),
            "market_ticker": features.market_ticker,
            "threshold": threshold,
            "settlement_price": expiry_candle.close,
            "label": label,
            "feature_schema_version": features.schema_version,
            **{
                column: getattr(features, column)
                for column in FEATURE_COLUMNS
            },
        }

    def _synthetic_market(
        self,
        *,
        anchor_time: datetime,
        expiry_time: datetime,
        threshold: float,
    ) -> KalshiMarket:
        anchor_utc = anchor_time.astimezone(UTC)
        expiry_utc = expiry_time.astimezone(UTC)
        horizon_minutes = int((expiry_utc - anchor_utc).total_seconds() // 60)
        ticker = f"HIST-{anchor_utc.strftime('%Y%m%d%H%M')}-{int(threshold)}"
        if horizon_minutes == 15:
            title = "BTC price up in next 15 mins?"
        else:
            title = f"Bitcoin above ${threshold:,.2f} at {expiry_utc.isoformat()}?"
        return KalshiMarket(
            ticker=ticker,
            title=title,
            direction="ABOVE",
            threshold=threshold,
            expires_at=expiry_utc,
            status="closed",
        )

    @staticmethod
    def _synthetic_snapshot(
        *,
        market: KalshiMarket,
        spot_price: float,
        captured_at: datetime,
    ) -> MarketSnapshot:
        distance_pct = (spot_price - market.threshold) / market.threshold
        probability_shift = max(min(distance_pct * 20, 0.2), -0.2)
        yes_mid = min(max(0.5 + probability_shift, 0.05), 0.95)
        spread = 0.02
        yes_bid = max(0.0, yes_mid - spread / 2)
        yes_ask = min(1.0, yes_mid + spread / 2)
        no_mid = 1 - yes_mid
        no_bid = max(0.0, no_mid - spread / 2)
        no_ask = min(1.0, no_mid + spread / 2)
        return MarketSnapshot(
            ticker=market.ticker,
            captured_at=captured_at,
            yes_price=round(yes_mid, 4),
            no_price=round(no_mid, 4),
            yes_bid=round(yes_bid, 4),
            yes_ask=round(yes_ask, 4),
            no_bid=round(no_bid, 4),
            no_ask=round(no_ask, 4),
        )


class ModelTrainer:
    """Train, save, and load local sklearn model artifacts."""

    def __init__(self, storage: DuckDBStorage | None = None) -> None:
        self.storage = storage

    def train(
        self,
        dataset: pd.DataFrame,
        *,
        artifact_path: str | Path,
        min_rows: int = DEFAULT_MIN_TRAINING_ROWS,
    ) -> TrainingRunResult:
        if len(dataset) < min_rows:
            raise ValueError(f"At least {min_rows} dataset rows are required to train a model.")

        ordered = dataset.sort_values("prediction_timestamp").reset_index(drop=True)
        labels = ordered["label"].astype(int)
        if labels.nunique() < 2:
            raise ValueError("Training dataset must include both ABOVE and BELOW labels.")

        validation_size = max(int(len(ordered) * 0.2), 1)
        split_index = len(ordered) - validation_size
        if split_index <= 0:
            raise ValueError("Training dataset is too small to create a validation split.")

        train_dataset = ordered.iloc[:split_index]
        validation_dataset = ordered.iloc[split_index:]
        if train_dataset["label"].nunique() < 2:
            train_dataset = ordered
            validation_dataset = ordered

        x_train = self._feature_frame(train_dataset)
        y_train = train_dataset["label"].astype(int)
        x_validation = self._feature_frame(validation_dataset)
        y_validation = validation_dataset["label"].astype(int)

        models = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1_000, random_state=42),
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }
        metrics: dict[str, dict[str, float]] = {}
        fitted_models: dict[str, Any] = {}

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            metrics[model_name] = self._metrics(y_validation, probabilities)
            fitted_models[model_name] = model

        selected_model_name = min(
            metrics,
            key=lambda name: (
                metrics[name]["log_loss"],
                metrics[name]["brier_score"],
                -metrics[name]["accuracy"],
                0 if name == "logistic_regression" else 1,
            ),
        )
        created_at = datetime.now(UTC)
        feature_schema_version = str(ordered["feature_schema_version"].iloc[0])
        training_window_start = self._as_datetime(ordered["prediction_timestamp"].iloc[0])
        training_window_end = self._as_datetime(ordered["prediction_timestamp"].iloc[-1])
        destination = Path(artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        artifact_payload = {
            "model_name": selected_model_name,
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_schema_version": feature_schema_version,
            "created_at": created_at.isoformat(),
            "training_window_start": training_window_start.isoformat(),
            "training_window_end": training_window_end.isoformat(),
            "metrics": metrics,
            "model": fitted_models[selected_model_name],
        }
        with destination.open("wb") as artifact_file:
            pickle.dump(artifact_payload, artifact_file)

        if self.storage is not None:
            self.storage.insert_model_metadata(
                metadata_id=f"{selected_model_name}:{created_at.isoformat()}",
                model_name=selected_model_name,
                feature_schema_version=feature_schema_version,
                created_at=created_at,
                artifact_path=str(destination),
                training_window_start=training_window_start,
                training_window_end=training_window_end,
                metrics=metrics,
            )

        return TrainingRunResult(
            artifact_path=destination,
            dataset_rows=int(len(ordered)),
            model_name=selected_model_name,
            feature_schema_version=feature_schema_version,
            metrics=metrics,
            training_window_start=training_window_start,
            training_window_end=training_window_end,
        )

    @staticmethod
    def load_artifact(path: str | Path) -> LoadedModelArtifact:
        source = Path(path)
        with source.open("rb") as artifact_file:
            payload = pickle.load(artifact_file)

        metrics_payload = payload["metrics"]
        if not isinstance(metrics_payload, dict):
            raise ValueError("Model artifact metrics payload was malformed.")

        return LoadedModelArtifact(
            path=source,
            model_name=str(payload["model_name"]),
            model=payload["model"],
            feature_columns=tuple(payload["feature_columns"]),
            feature_schema_version=str(payload["feature_schema_version"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])).astimezone(UTC),
            metrics={
                str(name): {
                    "accuracy": float(values["accuracy"]),
                    "log_loss": float(values["log_loss"]),
                    "brier_score": float(values["brier_score"]),
                }
                for name, values in metrics_payload.items()
            },
            training_window_start=datetime.fromisoformat(
                str(payload["training_window_start"])
            ).astimezone(UTC),
            training_window_end=datetime.fromisoformat(
                str(payload["training_window_end"])
            ).astimezone(UTC),
        )

    def predict_dataset_probabilities(
        self,
        artifact: LoadedModelArtifact,
        dataset: pd.DataFrame,
    ) -> list[float]:
        feature_frame = self._feature_frame(dataset, feature_columns=artifact.feature_columns)
        probabilities = artifact.model.predict_proba(feature_frame)[:, 1]
        return [float(probability) for probability in probabilities]

    def predict_feature_probability(
        self,
        artifact: LoadedModelArtifact,
        features: FeatureVector,
    ) -> float:
        row = pd.DataFrame(
            [
                {
                    column: getattr(features, column)
                    for column in artifact.feature_columns
                }
            ]
        )
        return self.predict_dataset_probabilities(artifact, row)[0]

    @staticmethod
    def _metrics(labels: pd.Series, probabilities: Any) -> dict[str, float]:
        clipped = [min(max(float(probability), 1e-6), 1 - 1e-6) for probability in probabilities]
        predictions = [1 if probability >= 0.5 else 0 for probability in clipped]
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
            "brier_score": float(brier_score_loss(labels, clipped)),
        }

    @staticmethod
    def _feature_frame(
        dataset: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    ) -> pd.DataFrame:
        feature_frame = dataset.loc[:, list(feature_columns)].copy()
        feature_frame = feature_frame.fillna(0.0)
        for column in feature_columns:
            feature_frame[column] = feature_frame[column].astype(float)
        return feature_frame

    @staticmethod
    def _as_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value))
        return dt.astimezone(UTC) if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
