"""Baseline predictor and blend logic."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from math import exp

from app.schemas import (
    ConfidenceBucket,
    FeatureVector,
    KalshiMarket,
    MarketDirection,
    MarketSnapshot,
    PredictionResult,
)


class Predictor:
    """Blend market prior, price features, and news pressure into a research prediction."""

    def __init__(
        self,
        *,
        market_weight: float = 0.45,
        price_weight: float = 0.40,
        news_weight: float = 0.15,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        total_weight = market_weight + price_weight + news_weight
        if total_weight <= 0:
            raise ValueError("Predictor weights must sum to a positive value.")

        self.market_weight = market_weight / total_weight
        self.price_weight = price_weight / total_weight
        self.news_weight = news_weight / total_weight
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def predict(
        self,
        *,
        market: KalshiMarket,
        snapshot: MarketSnapshot,
        features: FeatureVector,
        price_model_probability: float | None = None,
    ) -> PredictionResult:
        market_prior = features.market_implied_probability
        price_probability = (
            price_model_probability
            if price_model_probability is not None
            else self._price_model_probability(features)
        )
        news_probability = 0.5 + (features.news_weighted_impact / 2)

        probability = (
            self.market_weight * market_prior
            + self.price_weight * price_probability
            + self.news_weight * news_probability
        )
        probability = max(0.0, min(probability, 1.0))

        label: MarketDirection = "ABOVE" if probability >= 0.5 else "BELOW"
        confidence = self._confidence_bucket(probability)
        warnings = self._warnings(features)

        return PredictionResult(
            generated_at=self._now_provider(),
            market_ticker=market.ticker,
            label=label,
            probability=round(probability, 4),
            confidence=confidence,
            drivers=self._drivers(features, market_prior, price_probability),
            warnings=warnings,
            degraded=bool(warnings),
            feature_vector=features,
            market=market,
            market_snapshot=snapshot,
        )

    @staticmethod
    def _price_model_probability(features: FeatureVector) -> float:
        score = 0.0
        score += features.distance_to_strike_pct * 40
        score += features.return_5m * 8
        score += features.return_15m * 10
        score += features.return_30m * 6
        score += features.return_60m * 4
        score += features.momentum_slope * 30
        score += (features.rsi - 50) / 12
        score += features.ma_deviation * 20
        score -= features.realized_vol_15m * 1.5
        score -= features.realized_vol_60m
        score += features.news_weighted_impact * 0.5

        return 1 / (1 + exp(-score))

    @staticmethod
    def _confidence_bucket(probability: float) -> ConfidenceBucket:
        edge = abs(probability - 0.5)
        if edge >= 0.2:
            return "high"
        if edge >= 0.1:
            return "medium"
        return "low"

    @staticmethod
    def _warnings(features: FeatureVector) -> list[str]:
        warnings: list[str] = []
        if features.high_confidence_article_count == 0 and features.news_weighted_impact == 0:
            warnings.append("News signal unavailable or neutral; using neutral news contribution.")
        if features.spread >= 0.1:
            warnings.append("Market spread is wide; implied prior may be noisy.")
        return warnings

    @staticmethod
    def _drivers(
        features: FeatureVector,
        market_prior: float,
        price_probability: float,
    ) -> list[str]:
        candidates: list[tuple[float, str]] = []

        if market_prior >= 0.55:
            candidates.append((abs(market_prior - 0.5), "Kalshi market prior leans ABOVE."))
        elif market_prior <= 0.45:
            candidates.append((abs(market_prior - 0.5), "Kalshi market prior leans BELOW."))

        if features.distance_to_strike > 0:
            candidates.append(
                (
                    abs(features.distance_to_strike_pct),
                    "BTC spot is above the strike heading into expiry.",
                )
            )
        else:
            candidates.append(
                (
                    abs(features.distance_to_strike_pct),
                    "BTC spot remains below the strike heading into expiry.",
                )
            )

        if price_probability >= 0.55:
            candidates.append(
                (abs(price_probability - 0.5), "Short-term price action is supportive.")
            )
        elif price_probability <= 0.45:
            candidates.append(
                (abs(price_probability - 0.5), "Short-term price action is weakening.")
            )

        if features.news_weighted_impact >= 0.05:
            candidates.append(
                (features.news_weighted_impact, "Recent news flow is mildly bullish.")
            )
        elif features.news_weighted_impact <= -0.05:
            candidates.append(
                (abs(features.news_weighted_impact), "Recent news flow is mildly bearish.")
            )

        if features.realized_vol_15m >= 0.02:
            candidates.append((features.realized_vol_15m, "Short-horizon volatility is elevated."))

        ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
        unique_messages: list[str] = []
        for _, message in ordered:
            if message not in unique_messages:
                unique_messages.append(message)
            if len(unique_messages) == 3:
                break

        return unique_messages or ["Signal mix is balanced, so confidence remains limited."]
