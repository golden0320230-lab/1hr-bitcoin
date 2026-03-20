"""Hosted KimiClaw client for structured BTC news sentiment scoring."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import orjson

from app.schemas import (
    ArticleSentimentScore,
    NewsArticle,
    normalize_reviewer_market_call,
)
from app.services.storage import DuckDBStorage
from app.utils.retries import retry_operation
from app.utils.text import sanitize_text, truncate_text

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "news_impact_prompt.txt"


class KimiClawServiceError(RuntimeError):
    """Raised when the hosted scoring endpoint cannot be used successfully."""


class KimiClawClient:
    """HTTP client for a hosted KimiClaw endpoint that returns strict JSON."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        http_client: httpx.Client | None = None,
        storage: DuckDBStorage | None = None,
        timeout_seconds: float = 20.0,
        max_article_chars: int = 2_000,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.storage = storage
        self.max_article_chars = max_article_chars
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> KimiClawClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def score_article(
        self,
        article: NewsArticle,
        *,
        allow_fallback: bool = True,
    ) -> ArticleSentimentScore:
        try:
            response_payload = self._request_score(article)
            parsed_json = self._extract_json_payload(response_payload)
            impact_score = parsed_json["impact_score"]
            score = ArticleSentimentScore(
                article_url=article.url,
                model_name=self.model_name,
                scored_at=self._now_provider(),
                market_call=normalize_reviewer_market_call(
                    parsed_json.get("market_call"),
                    impact_score=impact_score,
                ),
                sentiment=parsed_json["sentiment"],
                relevance=parsed_json["relevance"],
                impact_horizon_minutes=parsed_json["impact_horizon_minutes"],
                impact_score=impact_score,
                confidence=parsed_json["confidence"],
                reason=parsed_json["reason"],
                raw_response=response_payload,
            )
        except (KimiClawServiceError, KeyError, TypeError, ValueError) as exc:
            if not allow_fallback:
                raise
            score = self._neutral_fallback(article, reason=f"KimiClaw fallback: {exc}")

        if self.storage is not None:
            self.storage.insert_news_scores([score])

        return score

    def score_articles(
        self,
        articles: Iterable[NewsArticle],
        *,
        allow_fallback: bool = True,
    ) -> list[ArticleSentimentScore]:
        return [
            self.score_article(article, allow_fallback=allow_fallback)
            for article in articles
        ]

    def build_prompt(self, article: NewsArticle) -> str:
        title = truncate_text(sanitize_text(article.title), max_chars=240)
        source = truncate_text(sanitize_text(article.source), max_chars=120)
        summary = sanitize_text(article.summary or "")
        article_payload = orjson.dumps(
            {
                "title": title,
                "source": source,
                "published_at_utc": article.published_at.isoformat(),
                "url": str(article.url),
                "summary": truncate_text(summary, max_chars=self.max_article_chars),
            },
            option=orjson.OPT_INDENT_2,
        ).decode("utf-8")
        payload = "\n".join(
            [
                self._prompt_template.strip(),
                "",
                "Article payload JSON:",
                article_payload,
            ]
        )
        return payload.strip()

    @retry_operation(httpx.HTTPError, KimiClawServiceError)
    def _request_score(self, article: NewsArticle) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Return JSON only.",
                },
                {
                    "role": "user",
                    "content": self.build_prompt(article),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise KimiClawServiceError("KimiClaw endpoint request failed.") from exc

        try:
            return cast(dict[str, Any], response.json())
        except ValueError as exc:
            raise KimiClawServiceError("KimiClaw response was not valid JSON.") from exc

    @staticmethod
    def _extract_json_payload(response_payload: dict[str, Any]) -> dict[str, Any]:
        if "sentiment" in response_payload:
            return response_payload

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise KimiClawServiceError("KimiClaw response did not include choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise KimiClawServiceError("KimiClaw response choice was malformed.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise KimiClawServiceError("KimiClaw response message was malformed.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise KimiClawServiceError("KimiClaw response message content was empty.")

        try:
            parsed = cast(dict[str, Any], orjson.loads(content))
        except orjson.JSONDecodeError as exc:
            raise KimiClawServiceError("KimiClaw content was not valid JSON.") from exc

        return parsed

    def _neutral_fallback(self, article: NewsArticle, *, reason: str) -> ArticleSentimentScore:
        return ArticleSentimentScore(
            article_url=article.url,
            model_name=self.model_name,
            scored_at=self._now_provider(),
            market_call="NEUTRAL",
            sentiment="neutral",
            relevance=0.0,
            impact_horizon_minutes=60,
            impact_score=0.0,
            confidence=0.0,
            reason=truncate_text(reason, max_chars=240),
            raw_response={"fallback": True},
        )
