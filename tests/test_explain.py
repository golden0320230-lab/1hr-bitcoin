"""Tests for prediction history retrieval and explain support."""

from __future__ import annotations

from pathlib import Path

from app.schemas import FeatureVector, KalshiMarket, MarketSnapshot, PredictionResult
from app.services.explain import ExplainService
from app.services.storage import DuckDBStorage


def _prediction_result() -> PredictionResult:
    market = KalshiMarket(
        ticker="BTCD-26MAR191600-T84500",
        title="Bitcoin price at Mar 19, 2026 at 4pm EDT?",
        direction="ABOVE",
        threshold=84_500,
        expires_at="2026-03-19T20:00:00Z",
    )
    snapshot = MarketSnapshot(
        ticker=market.ticker,
        captured_at="2026-03-19T19:30:00Z",
        yes_price=0.51,
        no_price=0.49,
        yes_bid=0.5,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.5,
    )
    features = FeatureVector(
        generated_at="2026-03-19T19:30:00Z",
        market_ticker=market.ticker,
        spot_price=84_700,
        strike_price=84_500,
        distance_to_strike=200,
        distance_to_strike_pct=200 / 84_500,
        kalshi_yes_price=0.51,
        kalshi_no_price=0.49,
        market_implied_probability=0.51,
        spread=0.02,
        return_5m=0.003,
        return_15m=0.004,
        return_30m=0.006,
        return_60m=0.01,
        realized_vol_15m=0.01,
        realized_vol_60m=0.015,
        ma_deviation=0.002,
        momentum_slope=0.001,
        rsi=56,
        minutes_to_expiry=30,
    )
    return PredictionResult(
        generated_at="2026-03-19T19:30:00Z",
        market_ticker=market.ticker,
        label="ABOVE",
        probability=0.58,
        confidence="medium",
        drivers=["BTC spot is above the strike heading into expiry."],
        warnings=["News signal unavailable or neutral; using neutral news contribution."],
        degraded=True,
        feature_vector=features,
        market=market,
        market_snapshot=snapshot,
    )


def test_explain_service_returns_latest_prediction_run(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    prediction = _prediction_result()

    try:
        storage.insert_prediction_run(prediction)
        service = ExplainService(storage)
        latest = service.get_last_prediction()

        assert latest is not None
        assert latest.market_ticker == prediction.market_ticker
        assert latest.label == prediction.label
        assert latest.drivers == prediction.drivers
    finally:
        storage.close()
