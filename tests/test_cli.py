"""Smoke tests for the Typer CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import sin

from typer.testing import CliRunner

import app.cli as cli
from app.cli import app
from app.config import Settings
from app.schemas import (
    BTCCandle,
    FeatureVector,
    KalshiMarket,
    MarketSnapshot,
    NewsArticle,
    PredictionResult,
)

runner = CliRunner()


def _training_candles() -> list[BTCCandle]:
    start = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
    candles: list[BTCCandle] = []
    closes = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        118,
        117,
        116,
        115,
        114,
        113,
        112,
        111,
        110,
        109,
        108,
        107,
        106,
        105,
        104,
        103,
        102,
        101,
        100,
        99,
    ]
    for index, close in enumerate(closes):
        candles.append(
            BTCCandle(
                source="coinbase",
                timeframe="5m",
                timestamp=start + timedelta(minutes=index * 5),
                open=close - 0.5,
                high=close + 0.5,
                low=close - 1.0,
                close=close,
                volume=10 + index,
            )
        )
    return candles


def _backtest_candles() -> list[BTCCandle]:
    start = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    candles: list[BTCCandle] = []
    for index in range(180):
        close = 84_500 + int(220 * sin(index / 4)) + ((index % 8) - 4) * 12
        candles.append(
            BTCCandle(
                source="coinbase",
                timeframe="5m",
                timestamp=start + timedelta(minutes=index * 5),
                open=close - 15,
                high=close + 20,
                low=close - 20,
                close=close,
                volume=25 + index,
            )
        )
    return candles


def _predict_candles() -> list[BTCCandle]:
    start = datetime(2026, 3, 19, 18, 0, tzinfo=UTC)
    candles: list[BTCCandle] = []
    for index in range(90):
        close = 84_420 + (index * 3)
        candles.append(
            BTCCandle(
                source="coinbase",
                timeframe="1m",
                timestamp=start + timedelta(minutes=index),
                open=close - 5,
                high=close + 8,
                low=close - 8,
                close=close,
                volume=15 + index,
            )
        )
    return candles


def test_root_help_shows_bootstrap_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Research-only CLI for the Kalshi BTC 15-minute predictor." in result.output
    assert "market" in result.output
    assert "predict" in result.output


def test_market_help_works() -> None:
    result = runner.invoke(app, ["market", "--help"])

    assert result.exit_code == 0
    assert "Show the live Kalshi BTC 15-minute market." in result.output


def test_market_json_includes_warnings_field(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH="./data/test_cli_market.duckdb",
            MODEL_PATH="./models/test_cli_market.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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
    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, snapshot),
    )

    result = runner.invoke(app, ["market", "--json"])

    assert result.exit_code == 0
    assert "\"warnings\": []" in result.output


def test_monitor_json_reports_hit_when_side_reaches_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_monitor_hit.duckdb",
            MODEL_PATH=tmp_path / "test_cli_monitor_hit.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    market = KalshiMarket(
        ticker="KXBTC15M-26MAR191545-45",
        title="BTC price up in next 15 mins?",
        direction="ABOVE",
        threshold=84_500,
        expires_at="2026-03-19T19:45:00Z",
    )
    snapshots = iter(
        [
            MarketSnapshot(
                ticker=market.ticker,
                captured_at="2026-03-19T19:31:00Z",
                yes_price=0.54,
                no_price=0.46,
                yes_bid=0.53,
                yes_ask=0.55,
                no_bid=0.45,
                no_ask=0.47,
            ),
            MarketSnapshot(
                ticker=market.ticker,
                captured_at="2026-03-19T19:32:00Z",
                yes_price=0.62,
                no_price=0.38,
                yes_bid=0.61,
                yes_ask=0.63,
                no_bid=0.37,
                no_ask=0.39,
            ),
        ]
    )

    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, next(snapshots)),
    )

    result = runner.invoke(
        app,
        ["monitor", "up", "0.60", "--poll-seconds", "15", "--max-checks", "2", "--json"],
    )

    assert result.exit_code == 0
    assert "\"status\": \"hit\"" in result.output
    assert "\"target_side\": \"up\"" in result.output
    assert "\"observed_price\": 0.62" in result.output


def test_monitor_json_reports_not_hit_after_max_checks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_monitor_not_hit.duckdb",
            MODEL_PATH=tmp_path / "test_cli_monitor_not_hit.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    market = KalshiMarket(
        ticker="KXBTC15M-26MAR191545-45",
        title="BTC price up in next 15 mins?",
        direction="ABOVE",
        threshold=84_500,
        expires_at="2026-03-19T19:45:00Z",
    )
    snapshots = iter(
        [
            MarketSnapshot(
                ticker=market.ticker,
                captured_at="2026-03-19T19:31:00Z",
                yes_price=0.54,
                no_price=0.46,
                yes_bid=0.53,
                yes_ask=0.55,
                no_bid=0.45,
                no_ask=0.47,
            ),
            MarketSnapshot(
                ticker=market.ticker,
                captured_at="2026-03-19T19:32:00Z",
                yes_price=0.56,
                no_price=0.44,
                yes_bid=0.55,
                yes_ask=0.57,
                no_bid=0.43,
                no_ask=0.45,
            ),
        ]
    )

    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, next(snapshots)),
    )

    result = runner.invoke(
        app,
        ["monitor", "up", "0.70", "--poll-seconds", "15", "--max-checks", "2", "--json"],
    )

    assert result.exit_code == 0
    assert "\"status\": \"not_hit\"" in result.output
    assert "Target price was not reached before monitoring stopped." in result.output


def test_monitor_json_handles_missing_live_market(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_monitor_no_market.duckdb",
            MODEL_PATH=tmp_path / "test_cli_monitor_no_market.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    responses = iter([None, None])
    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: next(responses),
    )

    result = runner.invoke(
        app,
        ["monitor", "down", "65", "--poll-seconds", "15", "--max-checks", "2", "--json"],
    )

    assert result.exit_code == 0
    assert "\"status\": \"not_hit\"" in result.output
    assert "\"target_price\": 0.65" in result.output
    assert "No live BTC 15-minute Kalshi market found during monitoring." in result.output


def test_news_command_outputs_articles(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH="./data/test_cli_news.duckdb",
            MODEL_PATH="./models/test_cli_news.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )
    monkeypatch.setattr(
        cli.NewsClient,
        "fetch_recent_articles",
        lambda self, limit, lookback_hours: [
            NewsArticle(
                title="Bitcoin ETF inflows rise",
                url="https://example.com/etf",
                source="Example Wire",
                published_at="2026-03-19T18:30:00Z",
                summary="Fresh inflow headline",
            )
        ],
    )

    result = runner.invoke(app, ["news", "--limit", "1"])

    assert result.exit_code == 0
    assert "Bitcoin ETF inflows rise" in result.output


def test_news_command_surfaces_partial_source_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH="./data/test_cli_news_warning.duckdb",
            MODEL_PATH="./models/test_cli_news_warning.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

    def _fetch_recent_articles(self, limit, lookback_hours):
        self.last_warnings = [
            "GDELT rate-limited the request; continuing without that source."
        ]
        return [
            NewsArticle(
                title="Bitcoin ETF inflows rise",
                url="https://example.com/etf",
                source="Example Wire",
                published_at="2026-03-19T18:30:00Z",
                summary="Fresh inflow headline",
            )
        ]

    monkeypatch.setattr(cli.NewsClient, "fetch_recent_articles", _fetch_recent_articles)

    result = runner.invoke(app, ["news", "--limit", "1", "--json"])

    assert result.exit_code == 0
    assert "GDELT rate-limited the request; continuing without that source." in result.output


def test_predict_json_outputs_structured_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH="./data/test_cli_predict.duckdb",
            MODEL_PATH="./models/test_cli_predict.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

    market = KalshiMarket(
        ticker="BTCD-26MAR191600-T84500",
        title="Bitcoin price at Mar 19, 2026 at 4pm EDT?",
        direction="ABOVE",
        threshold=84_500,
        expires_at="2026-03-19T20:00:00Z",
    )
    snapshot = MarketSnapshot(
        ticker="BTCD-26MAR191600-T84500",
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
    prediction = PredictionResult(
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

    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, snapshot),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: 84_700.0)
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _predict_candles(),
    )
    monkeypatch.setattr(
        cli.NewsClient,
        "fetch_recent_articles",
        lambda self, limit, lookback_hours: [],
    )
    monkeypatch.setattr(
        cli.FeatureBuilder,
        "build_feature_vector",
        lambda self, **kwargs: features,
    )
    monkeypatch.setattr(
        cli.Predictor,
        "predict",
        lambda self, market, snapshot, features, price_model_probability=None: prediction,
    )

    result = runner.invoke(app, ["predict", "--json"])

    assert result.exit_code == 0
    assert "\"prediction\"" in result.output
    assert "\"label\": \"ABOVE\"" in result.output
    assert "\"warnings\"" in result.output


def test_predict_json_includes_partial_news_source_warning(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_predict_news_warning.duckdb",
            MODEL_PATH=tmp_path / "missing.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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

    def _fetch_recent_articles(self, limit, lookback_hours):
        self.last_warnings = [
            "GDELT rate-limited the request; continuing without that source."
        ]
        return []

    monkeypatch.setattr(cli.KalshiClient, "get_live_btc_market", lambda self: (market, snapshot))
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: 84_700.0)
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _predict_candles(),
    )
    monkeypatch.setattr(cli.NewsClient, "fetch_recent_articles", _fetch_recent_articles)

    result = runner.invoke(app, ["predict", "--json"])

    assert result.exit_code == 0
    assert "GDELT rate-limited the request; continuing without that source." in result.output
    assert "\"prediction\"" in result.output


def test_explain_last_outputs_saved_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH="./data/test_cli_explain.duckdb",
            MODEL_PATH="./models/test_cli_explain.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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
    prediction = PredictionResult(
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

    monkeypatch.setattr(cli.ExplainService, "get_last_prediction", lambda self: prediction)

    result = runner.invoke(app, ["explain", "--last"])

    assert result.exit_code == 0
    assert "Prediction: ABOVE" in result.output
    assert "Feature values:" in result.output


def test_explain_json_outputs_structured_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH="./data/test_cli_explain_json.duckdb",
            MODEL_PATH="./models/test_cli_explain_json.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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
    prediction = PredictionResult(
        generated_at="2026-03-19T19:30:00Z",
        market_ticker=market.ticker,
        label="ABOVE",
        probability=0.58,
        confidence="medium",
        drivers=["BTC spot is above the strike heading into expiry."],
        warnings=["Model artifact missing; using heuristic price model."],
        degraded=True,
        feature_vector=features,
        market=market,
        market_snapshot=snapshot,
    )

    monkeypatch.setattr(cli.ExplainService, "get_last_prediction", lambda self: prediction)

    result = runner.invoke(app, ["explain", "--last", "--json"])

    assert result.exit_code == 0
    assert "\"prediction\"" in result.output
    assert "\"warnings\"" in result.output


def test_train_command_builds_model_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_train.duckdb",
            MODEL_PATH=tmp_path / "baseline.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles_range",
        lambda self, start_at, end_at, timeframe: _training_candles(),
    )

    result = runner.invoke(app, ["train", "--days", "30"])

    assert result.exit_code == 0
    assert "Selected model:" in result.output
    assert (tmp_path / "baseline.pkl").exists()


def test_backtest_command_outputs_baseline_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_backtest.duckdb",
            MODEL_PATH=tmp_path / "baseline.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles_range",
        lambda self, start_at, end_at, timeframe: _backtest_candles(),
    )

    result = runner.invoke(app, ["backtest", "--days", "30"])

    assert result.exit_code == 0
    assert "Backtest model:" in result.output
    assert "Baselines:" in result.output


def test_predict_json_warns_when_model_artifact_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_predict_missing_model.duckdb",
            MODEL_PATH=tmp_path / "missing.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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

    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, snapshot),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: 84_700.0)
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _predict_candles(),
    )

    result = runner.invoke(app, ["predict", "--json", "--no-news"])

    assert result.exit_code == 0
    assert "Model artifact missing; using heuristic price model." in result.output


def test_predict_uses_cached_candles_when_coinbase_fetch_fails(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "test_cli_predict_cached.duckdb"
    storage = cli.DuckDBStorage(db_path)
    try:
        storage.insert_candles(_predict_candles())
    finally:
        storage.close()

    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=db_path,
            MODEL_PATH=tmp_path / "missing.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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

    def _raise_coinbase_error(*args, **kwargs):
        raise cli.CoinbaseServiceError("boom")

    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, snapshot),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", _raise_coinbase_error)
    monkeypatch.setattr(cli.CoinbaseClient, "get_candles", _raise_coinbase_error)

    result = runner.invoke(app, ["predict", "--json", "--no-news"])

    assert result.exit_code == 0
    assert "Using cached BTC candles for feature generation." in result.output
    assert "\"prediction\"" in result.output


def test_predict_human_output_stays_research_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            APP_ENV="test",
            DB_PATH=tmp_path / "test_cli_predict_safe.duckdb",
            MODEL_PATH=tmp_path / "missing.pkl",
            KIMICLAW_BASE_URL="https://replace-me.example.com",
            KIMICLAW_API_KEY="replace-me",
            KIMICLAW_MODEL="replace-me",
        ),
    )

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

    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: (market, snapshot),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: 84_700.0)
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _predict_candles(),
    )

    result = runner.invoke(app, ["predict", "--no-news"])

    assert result.exit_code == 0
    assert "Research only. Not financial advice. No trade execution performed." in result.output
    assert "buy" not in result.output.lower()
    assert "sell" not in result.output.lower()
