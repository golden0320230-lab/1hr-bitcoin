"""Tests for Coinbase market data ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.coinbase import CoinbaseClient, CoinbaseServiceError
from app.services.storage import DuckDBStorage


def _build_http_client(
    *,
    spot_amount: str = "70610.25",
    candles: list[dict[str, Any]] | None = None,
    candles_status_code: int = 200,
) -> httpx.Client:
    candle_items = candles if candles is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/spot"):
            return httpx.Response(200, json={"data": {"amount": spot_amount, "currency": "USD"}})

        if request.url.path.endswith("/candles"):
            return httpx.Response(candles_status_code, json={"candles": candle_items})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.coinbase.com",
    )


def test_get_spot_price_parses_public_spot_endpoint() -> None:
    client = CoinbaseClient(http_client=_build_http_client(spot_amount="70555.12"))

    try:
        assert client.get_spot_price() == 70555.12
    finally:
        client.close()


def test_get_candles_normalizes_sorts_and_stores_results(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    client = CoinbaseClient(
        http_client=_build_http_client(
            candles=[
                {
                    "start": "1773977340",
                    "low": "70585.38",
                    "high": "70666.78",
                    "open": "70585.38",
                    "close": "70625.39",
                    "volume": "9.58269575",
                },
                {
                    "start": "1773977280",
                    "low": "70500.67",
                    "high": "70585.38",
                    "open": "70513.58",
                    "close": "70585.38",
                    "volume": "4.78648424",
                },
            ]
        ),
        storage=storage,
        now_provider=lambda: datetime(2026, 3, 19, 19, 55, tzinfo=UTC),
    )

    try:
        candles = client.get_candles(
            lookback_minutes=5,
            end_at=datetime(2026, 3, 19, 19, 55, tzinfo=UTC),
        )

        assert [candle.timestamp.isoformat() for candle in candles] == [
            "2026-03-20T03:28:00+00:00",
            "2026-03-20T03:29:00+00:00",
        ]
        assert storage.count_rows("btc_candles") == 2

        duplicate_run = client.get_candles(
            lookback_minutes=5,
            end_at=datetime(2026, 3, 19, 19, 55, tzinfo=UTC),
        )
        assert len(duplicate_run) == 2
        assert storage.count_rows("btc_candles") == 2
    finally:
        client.close()
        storage.close()


def test_get_candles_raises_clear_error_on_failed_response() -> None:
    client = CoinbaseClient(http_client=_build_http_client(candles_status_code=500))

    try:
        with pytest.raises(CoinbaseServiceError):
            client.get_candles()
    finally:
        client.close()
