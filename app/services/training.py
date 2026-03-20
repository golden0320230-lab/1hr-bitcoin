"""Training dataset generation aligned to hourly market semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from app.schemas import BTCCandle, FeatureVector, KalshiMarket, MarketSnapshot
from app.services.features import FeatureBuilder

DEFAULT_DATASET_PATH = Path("data/training_dataset.csv")
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


class TrainingDatasetBuilder:
    """Build historical training examples for above-or-below-at-expiry prediction."""

    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder | None = None,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        strike_increment: float = 100.0,
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
        horizon_minutes: int = 60,
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
        ticker = f"HIST-{anchor_utc.strftime('%Y%m%d%H%M')}-{int(threshold)}"
        title = f"Bitcoin above ${threshold:,.0f} at {expiry_utc.isoformat()}?"
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
