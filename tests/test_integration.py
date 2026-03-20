"""Offline integration coverage using representative provider fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from typer.testing import CliRunner

import app.cli as cli
from app.cli import app
from app.config import Settings
from app.services.coinbase import CoinbaseClient
from app.services.kalshi import KalshiClient
from app.services.kimiclaw import KimiClawClient
from app.services.news import GDELT_DOC_API_URL, NewsClient

runner = CliRunner()
FIXTURES_DIR = Path(__file__).parent / "fixtures"
ORIGINAL_KALSHI_GET_LIVE = KalshiClient.get_live_btc_market
ORIGINAL_COINBASE_GET_SPOT = CoinbaseClient.get_spot_price
ORIGINAL_COINBASE_GET_CANDLES = CoinbaseClient.get_candles
ORIGINAL_NEWS_FETCH = NewsClient.fetch_recent_articles
ORIGINAL_KIMICLAW_SCORE = KimiClawClient.score_articles


def _fixture_json(*parts: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR.joinpath(*parts)).read_text(encoding="utf-8"))


def _fixture_text(*parts: str) -> str:
    return FIXTURES_DIR.joinpath(*parts).read_text(encoding="utf-8")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        DB_PATH=tmp_path / "integration.duckdb",
        MODEL_PATH=tmp_path / "missing.pkl",
        KIMICLAW_BASE_URL="https://kimi.example.com",
        KIMICLAW_API_KEY="test-key",
        KIMICLAW_MODEL="kimi-btc-v1",
    )


def _kalshi_client() -> KalshiClient:
    payload = _fixture_json("kalshi", "live_markets.json")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return KalshiClient(
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="https://api.elections.kalshi.com/trade-api/v2",
        ),
        now_provider=lambda: datetime(2026, 3, 19, 19, 37, tzinfo=UTC),
    )


def _coinbase_client() -> CoinbaseClient:
    spot_payload = _fixture_json("coinbase", "spot.json")
    candle_payload = _fixture_json("coinbase", "candles.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/spot"):
            return httpx.Response(200, json=spot_payload)
        if request.url.path.endswith("/candles"):
            return httpx.Response(200, json=candle_payload)
        return httpx.Response(404, json={"error": "not found"})

    return CoinbaseClient(
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="https://api.coinbase.com",
        ),
        now_provider=lambda: datetime(2026, 3, 19, 19, 30, tzinfo=UTC),
    )


def _news_client() -> NewsClient:
    rss_xml = _fixture_text("news", "google_news_rss.xml")
    gdelt_payload = _fixture_json("news", "gdelt_articles.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://news.google.com/rss/search"):
            return httpx.Response(200, text=rss_xml)
        if str(request.url).startswith(GDELT_DOC_API_URL):
            return httpx.Response(200, json=gdelt_payload)
        return httpx.Response(404, json={"error": "not found"})

    return NewsClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )


def _kimiclaw_client(valid: bool = True) -> KimiClawClient:
    payload = _fixture_json(
        "kimiclaw",
        "valid_response.json" if valid else "invalid_response.json",
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return KimiClawClient(
        base_url="https://kimi.example.com",
        api_key="test-key",
        model_name="kimi-btc-v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )


def _fixture_market():
    client = _kalshi_client()
    try:
        discovered = ORIGINAL_KALSHI_GET_LIVE(client)
        assert discovered is not None
        return discovered
    finally:
        client.close()


def _fixture_spot() -> float:
    client = _coinbase_client()
    try:
        return ORIGINAL_COINBASE_GET_SPOT(client)
    finally:
        client.close()


def _fixture_candles():
    client = _coinbase_client()
    try:
        return ORIGINAL_COINBASE_GET_CANDLES(
            client,
            lookback_minutes=15,
            timeframe="1m",
            store=False,
        )
    finally:
        client.close()


def _fixture_articles():
    client = _news_client()
    try:
        return ORIGINAL_NEWS_FETCH(client, limit=10, lookback_hours=24)
    finally:
        client.close()


def _fixture_scores(valid: bool = True):
    articles = _fixture_articles()
    client = _kimiclaw_client(valid=valid)
    try:
        return ORIGINAL_KIMICLAW_SCORE(client, articles)
    finally:
        client.close()


def test_market_command_runs_offline_with_kalshi_fixture(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: _fixture_market(),
    )

    result = runner.invoke(app, ["market", "--json"])

    assert result.exit_code == 0
    assert "\"ticker\": \"KXBTC15M-26MAR191545-45\"" in result.output


def test_news_command_runs_offline_with_news_and_kimiclaw_fixtures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli.NewsClient,
        "fetch_recent_articles",
        lambda self, limit, lookback_hours: _fixture_articles()[:limit],
    )
    monkeypatch.setattr(
        cli.KimiClawClient,
        "score_articles",
        lambda self, articles: _fixture_scores(valid=True),
    )

    result = runner.invoke(app, ["news", "--limit", "2", "--json"])

    assert result.exit_code == 0
    assert "\"articles\"" in result.output
    assert "\"scores\"" in result.output
    assert "\"sentiment\": \"bullish\"" in result.output


def test_predict_command_runs_offline_with_provider_fixtures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: _fixture_market(),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: _fixture_spot())
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _fixture_candles(),
    )
    monkeypatch.setattr(
        cli.NewsClient,
        "fetch_recent_articles",
        lambda self, limit, lookback_hours: _fixture_articles()[:limit],
    )
    monkeypatch.setattr(
        cli.KimiClawClient,
        "score_articles",
        lambda self, articles: _fixture_scores(valid=True),
    )

    result = runner.invoke(app, ["predict", "--json"])

    assert result.exit_code == 0
    assert "\"prediction\"" in result.output
    assert "\"articles\"" in result.output


def test_predict_with_missing_news_still_returns_prediction(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: _fixture_market(),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: _fixture_spot())
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _fixture_candles(),
    )
    monkeypatch.setattr(
        cli.NewsClient,
        "fetch_recent_articles",
        lambda self, limit, lookback_hours: [],
    )

    result = runner.invoke(app, ["predict", "--json"])

    assert result.exit_code == 0
    assert "No recent BTC-related articles found; using neutral news contribution." in result.output
    assert "\"prediction\"" in result.output


def test_predict_with_invalid_kimiclaw_output_uses_neutral_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli.KalshiClient,
        "get_live_btc_market",
        lambda self: _fixture_market(),
    )
    monkeypatch.setattr(cli.CoinbaseClient, "get_spot_price", lambda self: _fixture_spot())
    monkeypatch.setattr(
        cli.CoinbaseClient,
        "get_candles",
        lambda self, lookback_minutes, timeframe: _fixture_candles(),
    )
    monkeypatch.setattr(
        cli.NewsClient,
        "fetch_recent_articles",
        lambda self, limit, lookback_hours: _fixture_articles()[:limit],
    )
    monkeypatch.setattr(
        cli.KimiClawClient,
        "score_articles",
        lambda self, articles: _fixture_scores(valid=False),
    )

    result = runner.invoke(app, ["predict", "--json"])

    assert result.exit_code == 0
    assert (
        "KimiClaw unavailable or invalid output; using neutral news contribution."
        in result.output
    )
    assert "\"prediction\"" in result.output
