"""Tests for hosted KimiClaw news scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.schemas import NewsArticle
from app.services.kimiclaw import KimiClawClient, KimiClawServiceError
from app.services.storage import DuckDBStorage


def _article() -> NewsArticle:
    return NewsArticle(
        title="Bitcoin ETF inflows rise again",
        url="https://example.com/bitcoin-etf",
        source="Example Wire",
        published_at="2026-03-19T18:30:00Z",
        summary="Fresh ETF inflows may support near-term BTC sentiment.",
    )


def _build_http_client(
    *,
    response_json: dict[str, Any] | None = None,
    status_code: int = 200,
) -> httpx.Client:
    payload = response_json if response_json is not None else {}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_score_article_parses_valid_openai_style_json_and_stores_result(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    client = KimiClawClient(
        base_url="https://kimi.example.com",
        api_key="test-key",
        model_name="kimi-btc-v1",
        http_client=_build_http_client(
            response_json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"sentiment":"bullish","relevance":0.82,'
                                '"impact_horizon_minutes":45,"impact_score":0.35,'
                                '"confidence":0.74,"reason":"ETF inflow headline may support BTC."}'
                            )
                        }
                    }
                ]
            }
        ),
        storage=storage,
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        score = client.score_article(_article())

        assert score.sentiment == "bullish"
        assert score.relevance == 0.82
        assert score.impact_score == 0.35
        assert storage.count_rows("news_scores") == 1
    finally:
        client.close()
        storage.close()


def test_score_article_falls_back_to_neutral_on_invalid_json() -> None:
    client = KimiClawClient(
        base_url="https://kimi.example.com",
        api_key="test-key",
        model_name="kimi-btc-v1",
        http_client=_build_http_client(
            response_json={"choices": [{"message": {"content": "not json"}}]}
        ),
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        score = client.score_article(_article())

        assert score.sentiment == "neutral"
        assert score.impact_score == 0.0
        assert "fallback" in score.reason.lower()
    finally:
        client.close()


def test_score_article_can_raise_without_fallback() -> None:
    client = KimiClawClient(
        base_url="https://kimi.example.com",
        api_key="test-key",
        model_name="kimi-btc-v1",
        http_client=_build_http_client(status_code=500),
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        with pytest.raises(KimiClawServiceError):
            client.score_article(_article(), allow_fallback=False)
    finally:
        client.close()


def test_build_prompt_sanitizes_html_and_truncates_untrusted_article_content() -> None:
    article = NewsArticle(
        title="<script>alert('x')</script>Bitcoin headline " + ("A" * 400),
        url="https://example.com/unsafe",
        source="<b>Example Wire</b>",
        published_at="2026-03-19T18:30:00Z",
        summary="<p>Fresh update</p><script>malicious()</script>" + ("B" * 4_000),
    )
    client = KimiClawClient(
        base_url="https://kimi.example.com",
        api_key="test-key",
        model_name="kimi-btc-v1",
        http_client=_build_http_client(),
        max_article_chars=120,
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        prompt = client.build_prompt(article)

        assert "<script>" not in prompt
        assert "malicious()" not in prompt
        assert "Title: Bitcoin headline" in prompt
        assert "Source: Example Wire" in prompt
        assert len(prompt) < 1_000
    finally:
        client.close()
