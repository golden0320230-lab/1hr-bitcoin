"""Tests for the baseline predictor blend logic."""

from __future__ import annotations

from datetime import UTC, datetime

from app.schemas import FeatureVector, KalshiMarket, MarketSnapshot
from app.services.predictor import Predictor


def _market() -> KalshiMarket:
    return KalshiMarket(
        ticker="BTCD-26MAR191600-T84500",
        title="Bitcoin price at Mar 19, 2026 at 4pm EDT?",
        direction="ABOVE",
        threshold=84_500,
        expires_at="2026-03-19T20:00:00Z",
    )


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        ticker="BTCD-26MAR191600-T84500",
        captured_at="2026-03-19T19:30:00Z",
        yes_price=0.51,
        no_price=0.49,
        yes_bid=0.5,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.5,
    )


def _feature_vector(
    *,
    distance_to_strike: float,
    distance_to_strike_pct: float,
    implied_probability: float,
    news_impact: float,
    high_confidence_article_count: int,
) -> FeatureVector:
    return FeatureVector(
        generated_at="2026-03-19T19:30:00Z",
        market_ticker="BTCD-26MAR191600-T84500",
        spot_price=84_320 + distance_to_strike,
        strike_price=84_500,
        distance_to_strike=distance_to_strike,
        distance_to_strike_pct=distance_to_strike_pct,
        kalshi_yes_price=implied_probability,
        kalshi_no_price=1 - implied_probability,
        market_implied_probability=implied_probability,
        spread=0.02,
        return_5m=0.004,
        return_15m=0.006,
        return_30m=0.008,
        return_60m=0.012,
        realized_vol_15m=0.01,
        realized_vol_60m=0.015,
        ma_deviation=0.003,
        momentum_slope=0.002,
        rsi=58,
        minutes_to_expiry=30,
        news_weighted_impact=news_impact,
        news_weighted_bullish=max(news_impact, 0),
        news_weighted_bearish=abs(min(news_impact, 0)),
        high_confidence_article_count=high_confidence_article_count,
        breaking_news=high_confidence_article_count >= 2,
    )


def test_predictor_outputs_above_for_supportive_signal_mix() -> None:
    predictor = Predictor(now_provider=lambda: datetime(2026, 3, 19, 19, 30, tzinfo=UTC))

    result = predictor.predict(
        market=_market(),
        snapshot=_snapshot(),
        features=_feature_vector(
            distance_to_strike=120,
            distance_to_strike_pct=0.0014,
            implied_probability=0.57,
            news_impact=0.2,
            high_confidence_article_count=2,
        ),
    )

    assert result.label == "ABOVE"
    assert result.probability > 0.5
    assert result.confidence in {"medium", "high"}
    assert result.drivers


def test_predictor_outputs_below_for_negative_signal_mix() -> None:
    predictor = Predictor(now_provider=lambda: datetime(2026, 3, 19, 19, 30, tzinfo=UTC))

    bearish_features = _feature_vector(
        distance_to_strike=-350,
        distance_to_strike_pct=-0.0041,
        implied_probability=0.41,
        news_impact=-0.2,
        high_confidence_article_count=2,
    ).model_copy(
        update={
            "return_5m": -0.005,
            "return_15m": -0.008,
            "return_30m": -0.012,
            "return_60m": -0.015,
            "ma_deviation": -0.004,
            "momentum_slope": -0.003,
            "rsi": 41,
        }
    )

    result = predictor.predict(
        market=_market(),
        snapshot=_snapshot(),
        features=bearish_features,
    )

    assert result.label == "BELOW"
    assert result.probability < 0.5


def test_predictor_works_without_news_scores() -> None:
    predictor = Predictor(now_provider=lambda: datetime(2026, 3, 19, 19, 30, tzinfo=UTC))

    result = predictor.predict(
        market=_market(),
        snapshot=_snapshot(),
        features=_feature_vector(
            distance_to_strike=50,
            distance_to_strike_pct=0.0006,
            implied_probability=0.53,
            news_impact=0.0,
            high_confidence_article_count=0,
        ),
    )

    assert result.warnings
    assert result.degraded is True
    assert result.probability >= 0.0


def test_predictor_accepts_price_model_override() -> None:
    predictor = Predictor(now_provider=lambda: datetime(2026, 3, 19, 19, 30, tzinfo=UTC))

    result = predictor.predict(
        market=_market(),
        snapshot=_snapshot(),
        features=_feature_vector(
            distance_to_strike=-50,
            distance_to_strike_pct=-0.0006,
            implied_probability=0.48,
            news_impact=0.0,
            high_confidence_article_count=0,
        ),
        price_model_probability=0.85,
    )

    assert result.label == "ABOVE"
    assert result.probability > 0.5
