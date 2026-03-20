"""Validation tests for shared domain schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas import (
    ArticleSentimentScore,
    BacktestResult,
    BTCCandle,
    FeatureVector,
    KalshiMarket,
    MarketSnapshot,
    NewsArticle,
    PredictionResult,
)


def test_kalshi_market_normalizes_expiry_to_utc() -> None:
    market = KalshiMarket(
        ticker="BTC-1HR-72500",
        title="Will Bitcoin close above $72,500 at 4pm ET?",
        direction="ABOVE",
        threshold=72_500,
        expires_at="2026-03-19T16:00:00-04:00",
    )

    assert market.expires_at.tzinfo == UTC
    assert market.expires_at.isoformat() == "2026-03-19T20:00:00+00:00"


def test_market_snapshot_requires_some_price_data() -> None:
    with pytest.raises(ValidationError):
        MarketSnapshot(
            ticker="BTC-1HR-72500",
            captured_at="2026-03-19T20:00:00Z",
        )


def test_btc_candle_rejects_invalid_ohlc_values() -> None:
    with pytest.raises(ValidationError):
        BTCCandle(
            source="coinbase",
            timeframe="1m",
            timestamp="2026-03-19T20:00:00Z",
            open=72_000,
            high=71_900,
            low=71_800,
            close=71_950,
        )


def test_news_article_accepts_naive_datetime_and_normalizes_to_utc() -> None:
    article = NewsArticle(
        title="Bitcoin ETF inflows rise",
        url="https://example.com/bitcoin-etf",
        source="Example News",
        published_at=datetime(2026, 3, 19, 20, 0, 0),
        summary="Short summary",
    )

    assert article.published_at.tzinfo == UTC
    assert article.published_at.isoformat() == "2026-03-19T20:00:00+00:00"


def test_article_sentiment_rejects_out_of_range_payload() -> None:
    with pytest.raises(ValidationError):
        ArticleSentimentScore(
            article_url="https://example.com/bitcoin-etf",
            model_name="kimi-claw-v1",
            scored_at="2026-03-19T20:05:00Z",
            market_call="UP",
            sentiment="bullish",
            relevance=1.2,
            impact_horizon_minutes=45,
            impact_score=0.4,
            confidence=0.8,
            reason="ETF flow headline is constructive for short-term BTC sentiment.",
        )


def test_feature_vector_rejects_negative_minutes_to_expiry() -> None:
    with pytest.raises(ValidationError):
        FeatureVector(
            generated_at="2026-03-19T20:10:00Z",
            market_ticker="BTC-1HR-72500",
            spot_price=72_320,
            strike_price=72_500,
            distance_to_strike=-180,
            distance_to_strike_pct=-0.0025,
            kalshi_yes_price=0.46,
            kalshi_no_price=0.54,
            market_implied_probability=0.46,
            spread=0.03,
            return_5m=-0.001,
            return_15m=0.002,
            return_30m=0.003,
            return_60m=0.006,
            realized_vol_15m=0.012,
            realized_vol_60m=0.025,
            ma_deviation=-0.001,
            momentum_slope=-0.2,
            rsi=48,
            minutes_to_expiry=-5,
        )


def test_prediction_result_accepts_nested_models() -> None:
    market = KalshiMarket(
        ticker="BTC-1HR-72500",
        title="Will Bitcoin close above $72,500 at 4pm ET?",
        direction="ABOVE",
        threshold=72_500,
        expires_at="2026-03-19T20:00:00Z",
    )
    snapshot = MarketSnapshot(
        ticker="BTC-1HR-72500",
        captured_at="2026-03-19T19:45:00Z",
        yes_price=0.48,
        no_price=0.52,
    )
    features = FeatureVector(
        generated_at="2026-03-19T19:45:00Z",
        market_ticker="BTC-1HR-72500",
        spot_price=72_320,
        strike_price=72_500,
        distance_to_strike=-180,
        distance_to_strike_pct=-0.0025,
        kalshi_yes_price=0.48,
        kalshi_no_price=0.52,
        market_implied_probability=0.48,
        spread=0.03,
        return_5m=-0.001,
        return_15m=0.002,
        return_30m=0.003,
        return_60m=0.006,
        realized_vol_15m=0.012,
        realized_vol_60m=0.025,
        ma_deviation=-0.001,
        momentum_slope=-0.2,
        rsi=48,
        minutes_to_expiry=15,
    )

    result = PredictionResult(
        generated_at="2026-03-19T19:45:00Z",
        market_ticker="BTC-1HR-72500",
        label="BELOW",
        probability=0.58,
        confidence="medium",
        drivers=["BTC remains below the strike into expiry."],
        feature_vector=features,
        market=market,
        market_snapshot=snapshot,
    )

    assert result.feature_vector is not None
    assert result.market is not None
    assert result.market_snapshot is not None


def test_backtest_result_requires_window_end_after_start() -> None:
    with pytest.raises(ValidationError):
        BacktestResult(
            generated_at="2026-03-19T20:15:00Z",
            window_start="2026-03-19T20:00:00Z",
            window_end="2026-03-19T19:00:00Z",
            model_name="logistic-regression",
            num_samples=32,
            accuracy=0.56,
            log_loss=0.67,
        )
