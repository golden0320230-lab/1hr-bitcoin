"""Tests for feature engineering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.schemas import ArticleSentimentScore, BTCCandle, KalshiMarket, MarketSnapshot
from app.services.features import FeatureBuilder


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
        yes_price=0.49,
        no_price=0.51,
        yes_bid=0.48,
        yes_ask=0.5,
        no_bid=0.5,
        no_ask=0.52,
    )


def _candles() -> list[BTCCandle]:
    base_time = datetime(2026, 3, 19, 18, 31, tzinfo=UTC)
    candles: list[BTCCandle] = []
    price = 84_100.0
    for offset in range(60):
        open_price = price
        close_price = price + (8 if offset % 2 == 0 else -3)
        high_price = max(open_price, close_price) + 4
        low_price = min(open_price, close_price) - 4
        candles.append(
            BTCCandle(
                source="coinbase",
                timeframe="1m",
                timestamp=base_time + timedelta(minutes=offset),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=10 + offset,
            )
        )
        price = close_price
    return candles


def test_build_feature_vector_with_news_scores() -> None:
    builder = FeatureBuilder()
    news_scores = [
        ArticleSentimentScore(
            article_url="https://example.com/etf",
            model_name="kimi-v1",
            scored_at="2026-03-19T19:10:00Z",
            sentiment="bullish",
            relevance=0.8,
            impact_horizon_minutes=45,
            impact_score=0.4,
            confidence=0.7,
            reason="ETF flow headline is constructive.",
        ),
        ArticleSentimentScore(
            article_url="https://example.com/hack",
            model_name="kimi-v1",
            scored_at="2026-03-19T19:12:00Z",
            sentiment="bearish",
            relevance=0.7,
            impact_horizon_minutes=60,
            impact_score=-0.3,
            confidence=0.8,
            reason="Security incident is risk-off.",
        ),
    ]

    features = builder.build_feature_vector(
        market=_market(),
        snapshot=_snapshot(),
        spot_price=84_320,
        candles=_candles(),
        news_scores=news_scores,
        generated_at=datetime(2026, 3, 19, 19, 30, tzinfo=UTC),
    )

    assert features.market_ticker == "BTCD-26MAR191600-T84500"
    assert features.distance_to_strike == -180
    assert features.minutes_to_expiry == 30
    assert features.spread == 0.02
    assert features.high_confidence_article_count == 2
    assert features.breaking_news is True
    assert features.news_weighted_bullish > 0
    assert features.news_weighted_bearish > 0


def test_build_feature_vector_uses_neutral_news_features_when_scores_missing() -> None:
    builder = FeatureBuilder()

    features = builder.build_feature_vector(
        market=_market(),
        snapshot=_snapshot(),
        spot_price=84_320,
        candles=_candles(),
        news_scores=[],
        generated_at=datetime(2026, 3, 19, 19, 30, tzinfo=UTC),
    )

    assert features.news_weighted_impact == 0.0
    assert features.news_weighted_bullish == 0.0
    assert features.news_weighted_bearish == 0.0
    assert features.high_confidence_article_count == 0
    assert features.breaking_news is False
