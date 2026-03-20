"""Tests for Kalshi live market discovery."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.services.kalshi import KalshiClient
from app.services.storage import DuckDBStorage


def _build_http_client(*, series_markets: list[dict[str, Any]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/markets"):
            return httpx.Response(404, json={"error": "not found"})

        if request.url.params.get("series_ticker") == "KXBTC15M":
            return httpx.Response(200, json={"cursor": "", "markets": series_markets})

        return httpx.Response(200, json={"cursor": "", "markets": []})

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.elections.kalshi.com/trade-api/v2",
    )


def _fifteen_minute_btc_payload(
    *,
    ticker: str,
    threshold: float | None,
    yes_bid: str,
    yes_ask: str,
    no_bid: str,
    no_ask: str,
    status: str = "active",
    open_time: str = "2026-03-19T19:30:00Z",
    close_time: str = "2026-03-19T19:45:00Z",
    yes_sub_title: str = "Target Price: $84,500.00",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": "KXBTC15M-26MAR191545",
        "title": "BTC price up in next 15 mins?",
        "status": status,
        "market_type": "binary",
        "strike_type": "greater_or_equal",
        "yes_sub_title": yes_sub_title,
        "no_sub_title": "Target price: TBD",
        "open_time": open_time,
        "close_time": close_time,
        "expected_expiration_time": "2026-03-19T19:50:00Z",
        "expiration_time": "2026-03-26T19:45:00Z",
        "updated_time": "2026-03-19T19:35:00Z",
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "no_bid_dollars": no_bid,
        "no_ask_dollars": no_ask,
        "last_price_dollars": yes_bid,
        "liquidity_dollars": "1500.0000",
        "volume_fp": "12.00",
        "open_interest_fp": "45.00",
        "rules_primary": (
            "If the simple average of the sixty seconds of CF Benchmarks' BRTI before "
            "expiration is at least the target price, the market resolves to Yes."
        ),
    }
    if threshold is not None:
        payload["floor_strike"] = threshold
    return payload


def test_get_live_btc_market_returns_best_live_candidate_and_stores_it(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    http_client = _build_http_client(
        series_markets=[
            _fifteen_minute_btc_payload(
                ticker="KXBTC15M-26MAR191545-45",
                threshold=84_500,
                yes_bid="0.51",
                yes_ask="0.53",
                no_bid="0.47",
                no_ask="0.49",
            ),
            _fifteen_minute_btc_payload(
                ticker="KXBTC15M-26MAR191545-50",
                threshold=84_600,
                yes_bid="0.88",
                yes_ask="0.92",
                no_bid="0.08",
                no_ask="0.12",
                yes_sub_title="Target Price: $84,600.00",
            ),
        ],
    )
    client = KalshiClient(
        http_client=http_client,
        storage=storage,
        now_provider=lambda: datetime(2026, 3, 19, 19, 36, tzinfo=UTC),
    )

    try:
        discovered = client.get_live_btc_market()
        assert discovered is not None

        market, snapshot = discovered
        assert market.ticker == "KXBTC15M-26MAR191545-45"
        assert market.direction == "ABOVE"
        assert market.threshold == 84_500
        assert market.expires_at.isoformat() == "2026-03-19T19:45:00+00:00"
        assert snapshot.yes_price == 0.52
        assert snapshot.no_price == 0.48
        assert storage.count_rows("markets") == 1
        assert storage.count_rows("market_snapshots") == 1
    finally:
        client.close()
        storage.close()


def test_get_live_btc_market_uses_subtitle_when_structured_strike_missing() -> None:
    http_client = _build_http_client(
        series_markets=[
            _fifteen_minute_btc_payload(
                ticker="KXBTC15M-26MAR191545-45",
                threshold=None,
                yes_bid="0.44",
                yes_ask="0.48",
                no_bid="0.52",
                no_ask="0.56",
                yes_sub_title="Target Price: $84,500.00",
            )
        ]
    )
    client = KalshiClient(
        http_client=http_client,
        now_provider=lambda: datetime(2026, 3, 19, 19, 36, tzinfo=UTC),
    )

    try:
        discovered = client.get_live_btc_market()
        assert discovered is not None
        market, _ = discovered
        assert market.threshold == 84_500
    finally:
        client.close()


def test_get_live_btc_market_returns_none_when_no_live_market_exists() -> None:
    http_client = _build_http_client(
        series_markets=[
            _fifteen_minute_btc_payload(
                ticker="KXBTC15M-26MAR191600-00",
                threshold=84_500,
                yes_bid="0.51",
                yes_ask="0.53",
                no_bid="0.47",
                no_ask="0.49",
                status="initialized",
                open_time="2026-03-19T19:45:00Z",
                close_time="2026-03-19T20:00:00Z",
            )
        ]
    )
    client = KalshiClient(
        http_client=http_client,
        now_provider=lambda: datetime(2026, 3, 19, 19, 36, tzinfo=UTC),
    )

    try:
        assert client.get_live_btc_market() is None
    finally:
        client.close()
