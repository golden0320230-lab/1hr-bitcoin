"""Kalshi market discovery helpers for BTC hourly markets."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx

from app.schemas import KalshiMarket, MarketDirection, MarketSnapshot, MarketStatus
from app.services.storage import DuckDBStorage

_THRESHOLD_PATTERN = re.compile(
    r"\$?(?P<threshold>[\d,]+(?:\.\d+)?)\s+or\s+(?P<direction>above|below)",
    re.IGNORECASE,
)


class KalshiClient:
    """Client for discovering live BTC hourly markets from Kalshi's public API."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
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

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_live_btc_hourly_market(self) -> tuple[KalshiMarket, MarketSnapshot] | None:
        payloads = self._fetch_candidate_payloads()
        live_payloads = [
            payload for payload in payloads if self._is_live_btc_hourly_market(payload)
        ]

        if not live_payloads:
            return None

        chosen_payload = min(live_payloads, key=self._candidate_sort_key)
        market = self.parse_market(chosen_payload)
        snapshot = self.parse_snapshot(chosen_payload)

        if self.storage is not None:
            self.storage.upsert_market(market)
            self.storage.insert_market_snapshot(snapshot)

        return market, snapshot

    def parse_market(self, payload: dict[str, Any]) -> KalshiMarket:
        threshold = self._extract_threshold(payload)
        direction = self._extract_direction(payload)
        expires_at = self._extract_expiry(payload)

        return KalshiMarket(
            ticker=str(payload["ticker"]),
            title=str(payload["title"]),
            event_ticker=self._as_text(payload.get("event_ticker")) or None,
            direction=direction,
            threshold=threshold,
            expires_at=expires_at,
            status=self._normalize_status(payload.get("status")),
            rules=self._as_text(payload.get("rules_primary")) or None,
            raw_payload=payload,
        )

    def parse_snapshot(self, payload: dict[str, Any]) -> MarketSnapshot:
        yes_bid = self._as_probability(payload.get("yes_bid_dollars"))
        yes_ask = self._as_probability(payload.get("yes_ask_dollars"))
        no_bid = self._as_probability(payload.get("no_bid_dollars"))
        no_ask = self._as_probability(payload.get("no_ask_dollars"))
        last_price = self._as_probability(payload.get("last_price_dollars"))

        yes_price = self._derive_price(yes_bid, yes_ask, last_price)
        no_price = self._derive_price(
            no_bid,
            no_ask,
            None if last_price is None else round(1 - last_price, 4),
        )

        if yes_price is None and no_price is not None:
            yes_price = round(1 - no_price, 4)
        if no_price is None and yes_price is not None:
            no_price = round(1 - yes_price, 4)

        volume = self._as_float(payload.get("volume_fp"))
        if volume is None:
            volume = self._as_float(payload.get("volume_24h_fp"))

        open_interest_raw = self._as_float(payload.get("open_interest_fp"))
        open_interest = int(open_interest_raw) if open_interest_raw is not None else None

        return MarketSnapshot(
            ticker=str(payload["ticker"]),
            captured_at=self._extract_updated_time(payload),
            yes_price=yes_price,
            no_price=no_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            volume=volume,
            open_interest=open_interest,
            raw_payload=payload,
        )

    def _fetch_candidate_payloads(self) -> list[dict[str, Any]]:
        series_payloads = self._fetch_markets(series_ticker="BTCD", limit=200, max_pages=3)
        if series_payloads:
            return series_payloads

        return self._fetch_markets(limit=1_000, max_pages=5)

    def _fetch_markets(
        self,
        *,
        series_ticker: str | None = None,
        limit: int,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        cursor: str | None = None

        for _ in range(max_pages):
            params: dict[str, Any] = {"limit": limit}
            if series_ticker is not None:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor

            response = self._client.get("/markets", params=params)
            response.raise_for_status()
            data = cast(dict[str, Any], response.json())
            page_items = cast(list[dict[str, Any]], data.get("markets", []))
            payloads.extend(page_items)

            cursor_value = data.get("cursor")
            cursor = str(cursor_value) if cursor_value else None
            if cursor is None:
                break

        return payloads

    def _is_live_btc_hourly_market(self, payload: dict[str, Any]) -> bool:
        ticker = self._as_text(payload.get("ticker")).upper()
        event_ticker = self._as_text(payload.get("event_ticker")).upper()
        title = self._as_text(payload.get("title"))
        status = self._as_text(payload.get("status")).lower()

        if status in {"settled", "closed", "finalized"}:
            return False

        identifier_text = " ".join([ticker, event_ticker, title]).upper()
        if "BITCOIN" not in identifier_text and "BTC" not in identifier_text:
            return False

        open_time = self._parse_datetime(payload.get("open_time"))
        close_time = self._parse_datetime(payload.get("close_time"))
        if open_time is None or close_time is None:
            return False

        duration = close_time - open_time
        if duration < timedelta(minutes=45) or duration > timedelta(minutes=75):
            return False

        if close_time <= self._now_provider():
            return False

        try:
            self._extract_threshold(payload)
            self._extract_direction(payload)
        except ValueError:
            return False

        return True

    def _candidate_sort_key(self, payload: dict[str, Any]) -> tuple[datetime, float, float]:
        close_time = self._extract_expiry(payload)
        midpoint = self._derive_price(
            self._as_probability(payload.get("yes_bid_dollars")),
            self._as_probability(payload.get("yes_ask_dollars")),
            self._as_probability(payload.get("last_price_dollars")),
        )
        distance_from_fair = 1.0 if midpoint is None else abs(midpoint - 0.5)
        liquidity = self._as_float(payload.get("liquidity_dollars")) or 0.0

        return close_time, distance_from_fair, -liquidity

    def _extract_threshold(self, payload: dict[str, Any]) -> float:
        for key in ("floor_strike", "cap_strike"):
            numeric_value = self._as_float(payload.get(key))
            if numeric_value is not None and numeric_value > 0:
                return numeric_value

        for text_key in ("yes_sub_title", "no_sub_title", "title"):
            text = self._as_text(payload.get(text_key))
            match = _THRESHOLD_PATTERN.search(text)
            if match is not None:
                return float(match.group("threshold").replace(",", ""))

        raise ValueError("Unable to extract market threshold from payload.")

    def _extract_direction(self, payload: dict[str, Any]) -> MarketDirection:
        strike_type = self._as_text(payload.get("strike_type")).lower()
        if strike_type == "greater":
            return "ABOVE"
        if strike_type == "less":
            return "BELOW"

        for text_key in ("yes_sub_title", "no_sub_title", "title"):
            text = self._as_text(payload.get(text_key))
            match = _THRESHOLD_PATTERN.search(text)
            if match is not None:
                return "ABOVE" if match.group("direction").lower() == "above" else "BELOW"

        raise ValueError("Unable to determine market direction from payload.")

    def _extract_expiry(self, payload: dict[str, Any]) -> datetime:
        close_time = self._parse_datetime(payload.get("close_time"))
        if close_time is not None:
            return close_time

        expected_expiration_time = self._parse_datetime(payload.get("expected_expiration_time"))
        if expected_expiration_time is not None:
            return expected_expiration_time

        raise ValueError("Unable to determine market close time from payload.")

    def _extract_updated_time(self, payload: dict[str, Any]) -> datetime:
        for key in ("updated_time", "created_time", "open_time"):
            parsed = self._parse_datetime(payload.get(key))
            if parsed is not None:
                return parsed

        return self._now_provider()

    @staticmethod
    def _normalize_status(value: Any) -> MarketStatus:
        status = str(value).lower() if value is not None else "unknown"
        if status in {"open", "closed", "settled"}:
            return cast(MarketStatus, status)
        return "unknown"

    @staticmethod
    def _derive_price(
        bid: float | None,
        ask: float | None,
        last: float | None,
    ) -> float | None:
        if bid is not None and ask is not None:
            return round((bid + ask) / 2, 4)
        if last is not None:
            return last
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        return None

    @staticmethod
    def _as_probability(value: Any) -> float | None:
        numeric_value = KalshiClient._as_float(value)
        if numeric_value is None:
            return None
        return round(numeric_value, 4)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value in {None, ""}:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value)
