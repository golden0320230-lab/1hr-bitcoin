"""Coinbase public market data client."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.schemas import BTCCandle, CandleTimeframe
from app.services.storage import DuckDBStorage

_GRANULARITY_MAP: dict[CandleTimeframe, str] = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "1h": "ONE_HOUR",
}


class CoinbaseServiceError(RuntimeError):
    """Raised when Coinbase public market data cannot be fetched or parsed."""


class CoinbaseClient:
    """Thin client for Coinbase public spot and candle data."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.coinbase.com",
        http_client: httpx.Client | None = None,
        storage: DuckDBStorage | None = None,
        timeout_seconds: float = 20.0,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(base_url=base_url, timeout=timeout_seconds)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CoinbaseClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_spot_price(self, product_id: str = "BTC-USD") -> float:
        payload = self._request_json(f"/v2/prices/{product_id}/spot")
        data = payload.get("data")
        if not isinstance(data, dict) or "amount" not in data:
            raise CoinbaseServiceError(
                "Coinbase spot price response did not include a spot amount."
            )

        return float(data["amount"])

    def get_candles(
        self,
        *,
        product_id: str = "BTC-USD",
        timeframe: CandleTimeframe = "1m",
        lookback_minutes: int = 60,
        end_at: datetime | None = None,
        store: bool = True,
    ) -> list[BTCCandle]:
        if lookback_minutes <= 0:
            raise ValueError("lookback_minutes must be positive.")

        end_time = end_at.astimezone(UTC) if end_at is not None else self._now_provider()
        start_time = end_time - timedelta(minutes=lookback_minutes)

        payload = self._request_json(
            f"/api/v3/brokerage/market/products/{product_id}/candles",
            params={
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "granularity": _GRANULARITY_MAP[timeframe],
            },
            headers={"cache-control": "no-cache"},
        )
        raw_candles = payload.get("candles")
        if not isinstance(raw_candles, list):
            raise CoinbaseServiceError("Coinbase candles response did not include a candles list.")

        candles = sorted(
            (
                self._normalize_candle(item, product_id=product_id, timeframe=timeframe)
                for item in raw_candles
            ),
            key=lambda candle: candle.timestamp,
        )

        if store and self.storage is not None:
            self.storage.insert_candles(candles)

        return candles

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, CoinbaseServiceError)),
    )
    def _request_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.get(path, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CoinbaseServiceError(f"Coinbase request failed for {path}.") from exc

        try:
            payload = cast(dict[str, Any], response.json())
        except ValueError as exc:
            raise CoinbaseServiceError(f"Coinbase response for {path} was not valid JSON.") from exc

        return payload

    @staticmethod
    def _normalize_candle(
        payload: dict[str, Any],
        *,
        product_id: str,
        timeframe: CandleTimeframe,
    ) -> BTCCandle:
        return BTCCandle(
            source="coinbase",
            product_id=product_id,
            timeframe=timeframe,
            timestamp=datetime.fromtimestamp(int(payload["start"]), tz=UTC),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload["volume"]),
            raw_payload=payload,
        )
