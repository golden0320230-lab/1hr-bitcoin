"""Feature engineering for market, price, and news signals."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from math import sqrt
from typing import TypedDict

import numpy as np

from app.schemas import (
    ArticleSentimentScore,
    BTCCandle,
    FeatureVector,
    KalshiMarket,
    MarketSnapshot,
)


class NewsAggregate(TypedDict):
    weighted_impact: float
    weighted_bullish: float
    weighted_bearish: float
    high_confidence_article_count: int
    breaking_news: bool


class FeatureBuilder:
    """Build a model-ready feature vector from market, price, and news inputs."""

    def build_feature_vector(
        self,
        *,
        market: KalshiMarket,
        snapshot: MarketSnapshot,
        spot_price: float,
        candles: Sequence[BTCCandle],
        news_scores: Sequence[ArticleSentimentScore] | None = None,
        generated_at: datetime | None = None,
    ) -> FeatureVector:
        if not candles:
            raise ValueError("At least one candle is required to build a feature vector.")

        ordered_candles = sorted(candles, key=lambda candle: candle.timestamp)
        latest_candle = ordered_candles[-1]
        closes = np.array([candle.close for candle in ordered_candles], dtype=float)
        now = generated_at or latest_candle.timestamp.astimezone(UTC)

        yes_price = (
            snapshot.yes_price
            if snapshot.yes_price is not None
            else 1 - (snapshot.no_price or 0.5)
        )
        no_price = snapshot.no_price if snapshot.no_price is not None else 1 - yes_price
        implied_probability = yes_price

        if snapshot.yes_bid is not None and snapshot.yes_ask is not None:
            spread = round(max(snapshot.yes_ask - snapshot.yes_bid, 0.0), 4)
        else:
            spread = round(abs((snapshot.yes_price or 0.5) + (snapshot.no_price or 0.5) - 1.0), 4)

        orderbook_imbalance = None
        if snapshot.yes_bid is not None and snapshot.no_bid is not None:
            denominator = snapshot.yes_bid + snapshot.no_bid
            if denominator > 0:
                orderbook_imbalance = (snapshot.yes_bid - snapshot.no_bid) / denominator

        news_features = self._aggregate_news(news_scores or [])
        mean_close = float(np.mean(closes))

        return FeatureVector(
            generated_at=now,
            market_ticker=market.ticker,
            spot_price=spot_price,
            strike_price=market.threshold,
            distance_to_strike=spot_price - market.threshold,
            distance_to_strike_pct=(spot_price - market.threshold) / market.threshold,
            kalshi_yes_price=yes_price,
            kalshi_no_price=no_price,
            market_implied_probability=implied_probability,
            spread=spread,
            orderbook_imbalance=orderbook_imbalance,
            return_5m=self._window_return(ordered_candles, minutes=5),
            return_15m=self._window_return(ordered_candles, minutes=15),
            return_30m=self._window_return(ordered_candles, minutes=30),
            return_60m=self._window_return(ordered_candles, minutes=60),
            realized_vol_15m=self._realized_volatility(ordered_candles, window=15),
            realized_vol_60m=self._realized_volatility(ordered_candles, window=60),
            ma_deviation=(spot_price - mean_close) / mean_close,
            momentum_slope=self._momentum_slope(closes),
            rsi=self._rsi(closes),
            minutes_to_expiry=max(int((market.expires_at - now).total_seconds() // 60), 0),
            news_weighted_impact=news_features["weighted_impact"],
            news_weighted_bullish=news_features["weighted_bullish"],
            news_weighted_bearish=news_features["weighted_bearish"],
            high_confidence_article_count=news_features["high_confidence_article_count"],
            breaking_news=news_features["breaking_news"],
        )

    @staticmethod
    def _window_return(candles: Sequence[BTCCandle], *, minutes: int) -> float:
        latest = candles[-1].close
        cutoff = candles[-1].timestamp.timestamp() - minutes * 60
        baseline = candles[0].close

        for candle in reversed(candles):
            if candle.timestamp.timestamp() <= cutoff:
                baseline = candle.close
                break

        return (latest - baseline) / baseline

    @staticmethod
    def _realized_volatility(candles: Sequence[BTCCandle], *, window: int) -> float:
        closes = np.array([candle.close for candle in candles[-window:]], dtype=float)
        if len(closes) < 2:
            return 0.0

        returns = np.diff(closes) / closes[:-1]
        if len(returns) == 0:
            return 0.0

        return float(np.std(returns, ddof=0) * sqrt(len(returns)))

    @staticmethod
    def _momentum_slope(closes: np.ndarray) -> float:
        if len(closes) < 2:
            return 0.0

        x_values = np.arange(len(closes), dtype=float)
        slope, _ = np.polyfit(x_values, closes, deg=1)
        return float(slope / closes[-1])

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < 2:
            return 50.0

        deltas = np.diff(closes[-(period + 1) :])
        if len(deltas) == 0:
            return 50.0

        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        average_gain = float(np.mean(gains)) if len(gains) else 0.0
        average_loss = float(np.mean(losses)) if len(losses) else 0.0

        if average_loss == 0:
            return 100.0 if average_gain > 0 else 50.0

        relative_strength = average_gain / average_loss
        return float(100 - (100 / (1 + relative_strength)))

    @staticmethod
    def _aggregate_news(scores: Sequence[ArticleSentimentScore]) -> NewsAggregate:
        if not scores:
            return {
                "weighted_impact": 0.0,
                "weighted_bullish": 0.0,
                "weighted_bearish": 0.0,
                "high_confidence_article_count": 0,
                "breaking_news": False,
            }

        weights = [score.relevance * score.confidence for score in scores]
        total_weight = sum(weights)
        weighted_impact = 0.0
        if total_weight > 0:
            weighted_impact = (
                sum(
                    score.impact_score * weight
                    for score, weight in zip(scores, weights, strict=False)
                )
                / total_weight
            )

        weighted_bullish = sum(
            weight
            for score, weight in zip(scores, weights, strict=False)
            if score.sentiment == "bullish"
        )
        weighted_bearish = sum(
            weight
            for score, weight in zip(scores, weights, strict=False)
            if score.sentiment == "bearish"
        )
        high_confidence_article_count = sum(
            1
            for score in scores
            if score.relevance >= 0.6 and score.confidence >= 0.6
        )

        return {
            "weighted_impact": float(weighted_impact),
            "weighted_bullish": float(weighted_bullish),
            "weighted_bearish": float(weighted_bearish),
            "high_confidence_article_count": high_confidence_article_count,
            "breaking_news": high_confidence_article_count >= 2,
        }
