"""Tests for Kalshi live market discovery."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.services.kalshi import KalshiClient
from app.services.storage import DuckDBStorage


def _build_http_client(
    *,
    btcd_markets: list[dict[str, Any]],
    fallback_markets: list[dict[str, Any]] | None = None,
) -> httpx.Client:
    fallback_items = fallback_markets if fallback_markets is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/markets"):
            return httpx.Response(404, json={"error": "not found"})

        if request.url.params.get("series_ticker") == "BTCD":
            return httpx.Response(200, json={"cursor": "", "markets": btcd_markets})

        return httpx.Response(200, json={"cursor": "", "markets": fallback_items})

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.elections.kalshi.com/trade-api/v2",
    )


def _hourly_btc_payload(
    *,
    ticker: str,
    threshold: float | None,
    yes_bid: str,
    yes_ask: str,
    no_bid: str,
    no_ask: str,
    close_time: str = "2026-03-19T20:00:00Z",
    strike_type: str = "greater",
    title: str = "Bitcoin price at Mar 19, 2026 at 4pm EDT?",
    yes_sub_title: str = "$84,500 or above",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": "BTCD-26MAR191600",
        "title": title,
        "status": "open",
        "market_type": "binary",
        "strike_type": strike_type,
        "yes_sub_title": yes_sub_title,
        "no_sub_title": yes_sub_title,
        "open_time": "2026-03-19T19:00:00Z",
        "close_time": close_time,
        "expected_expiration_time": "2026-03-19T20:05:00Z",
        "expiration_time": "2026-03-26T20:00:00Z",
        "updated_time": "2026-03-19T19:20:00Z",
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "no_bid_dollars": no_bid,
        "no_ask_dollars": no_ask,
        "last_price_dollars": yes_bid,
        "liquidity_dollars": "1500.0000",
        "volume_fp": "12.00",
        "open_interest_fp": "45.00",
        "rules_primary": (
            "If the CF Benchmarks settlement exceeds the strike, the market resolves to Yes."
        ),
    }
    if threshold is not None:
        payload["floor_strike"] = threshold
    return payload


def test_get_live_btc_hourly_market_returns_best_live_candidate_and_stores_it(
    tmp_path: Path,
) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    http_client = _build_http_client(
        btcd_markets=[
            _hourly_btc_payload(
                ticker="BTCD-26MAR191600-T84500",
                threshold=84_500,
                yes_bid="0.47",
                yes_ask="0.51",
                no_bid="0.49",
                no_ask="0.53",
            ),
            _hourly_btc_payload(
                ticker="BTCD-26MAR191600-T86000",
                threshold=86_000,
                yes_bid="0.08",
                yes_ask="0.12",
                no_bid="0.88",
                no_ask="0.92",
            ),
        ],
    )
    client = KalshiClient(
        http_client=http_client,
        storage=storage,
        now_provider=lambda: datetime(2026, 3, 19, 19, 15, tzinfo=UTC),
    )

    try:
        discovered = client.get_live_btc_hourly_market()
        assert discovered is not None

        market, snapshot = discovered
        assert market.ticker == "BTCD-26MAR191600-T84500"
        assert market.direction == "ABOVE"
        assert market.threshold == 84_500
        assert market.expires_at.isoformat() == "2026-03-19T20:00:00+00:00"
        assert snapshot.yes_price == 0.49
        assert snapshot.no_price == 0.51
        assert storage.count_rows("markets") == 1
        assert storage.count_rows("market_snapshots") == 1
    finally:
        client.close()
        storage.close()


def test_get_live_btc_hourly_market_uses_subtitle_when_structured_strike_missing() -> None:
    http_client = _build_http_client(
        btcd_markets=[
            _hourly_btc_payload(
                ticker="BTCD-26MAR191600-T84500",
                threshold=None,
                yes_bid="0.44",
                yes_ask="0.48",
                no_bid="0.52",
                no_ask="0.56",
                yes_sub_title="$84,500 or above",
            )
        ]
    )
    client = KalshiClient(
        http_client=http_client,
        now_provider=lambda: datetime(2026, 3, 19, 19, 15, tzinfo=UTC),
    )

    try:
        discovered = client.get_live_btc_hourly_market()
        assert discovered is not None
        market, _ = discovered
        assert market.threshold == 84_500
    finally:
        client.close()


def test_get_live_btc_hourly_market_returns_none_when_no_live_hourly_btc_market_exists() -> None:
    fallback_market = {
        "ticker": "KXBTC15M-26MAR191500-00",
        "event_ticker": "KXBTC15M-26MAR191500",
        "title": "BTC price up in next 15 mins?",
        "status": "initialized",
        "open_time": "2026-03-19T19:00:00Z",
        "close_time": "2026-03-19T19:15:00Z",
        "updated_time": "2026-03-19T19:00:00Z",
        "yes_bid_dollars": "0.51",
        "yes_ask_dollars": "0.53",
        "no_bid_dollars": "0.47",
        "no_ask_dollars": "0.49",
    }
    http_client = _build_http_client(btcd_markets=[], fallback_markets=[fallback_market])
    client = KalshiClient(
        http_client=http_client,
        now_provider=lambda: datetime(2026, 3, 19, 19, 15, tzinfo=UTC),
    )

    try:
        assert client.get_live_btc_hourly_market() is None
    finally:
        client.close()
