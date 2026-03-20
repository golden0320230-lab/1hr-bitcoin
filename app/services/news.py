"""News ingestion pipeline for BTC-related articles."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, cast
from urllib.parse import quote_plus

import feedparser  # type: ignore[import-untyped]
import httpx
from pydantic import HttpUrl, TypeAdapter

from app.schemas import NewsArticle
from app.services.storage import DuckDBStorage
from app.utils.retries import retry_operation
from app.utils.text import fingerprint_article, sanitize_text

GOOGLE_NEWS_BASE_URL = "https://news.google.com/rss/search"
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_NEWS_TERMS = ("bitcoin", "btc", "bitcoin etf", "crypto market")
_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


class NewsServiceError(RuntimeError):
    """Raised when news sources cannot be parsed or fetched."""


class NewsClient:
    """Fetch and normalize BTC-related news from free public sources."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        storage: DuckDBStorage | None = None,
        timeout_seconds: float = 20.0,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self.last_warnings: list[str] = []
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> NewsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def build_google_news_rss_url(
        self,
        *,
        terms: Iterable[str] = DEFAULT_NEWS_TERMS,
        lookback_hours: int = 24,
    ) -> str:
        query = " OR ".join(f'"{term}"' for term in terms)
        query = f"({query}) when:{lookback_hours}h"
        return (
            f"{GOOGLE_NEWS_BASE_URL}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )

    def fetch_recent_articles(
        self,
        *,
        limit: int = 10,
        lookback_hours: int = 24,
        include_gdelt: bool = True,
    ) -> list[NewsArticle]:
        self.last_warnings = []
        articles: list[NewsArticle] = []
        source_failures = 0

        try:
            articles.extend(self.fetch_google_news_rss(limit=limit, lookback_hours=lookback_hours))
        except NewsServiceError as exc:
            source_failures += 1
            self.last_warnings.append(self._source_warning("Google News RSS", exc))

        if include_gdelt:
            gdelt_limit = max(limit, 10)
            try:
                articles.extend(
                    self.fetch_gdelt_articles(limit=gdelt_limit, lookback_hours=lookback_hours)
                )
            except NewsServiceError as exc:
                source_failures += 1
                self.last_warnings.append(self._source_warning("GDELT", exc))

        attempted_sources = 2 if include_gdelt else 1
        if source_failures >= attempted_sources:
            self.last_warnings.append("All news sources failed; returning no recent articles.")

        deduped = self._deduplicate_articles(articles, lookback_hours=lookback_hours)
        limited = deduped[:limit]

        if self.storage is not None and limited:
            self.storage.insert_articles(limited)

        return limited

    def fetch_google_news_rss(
        self,
        *,
        limit: int = 10,
        lookback_hours: int = 24,
        terms: Iterable[str] = DEFAULT_NEWS_TERMS,
    ) -> list[NewsArticle]:
        response_text = self._request_text(
            self.build_google_news_rss_url(terms=terms, lookback_hours=lookback_hours)
        )

        parse_feed = cast(Any, feedparser.parse)
        parsed = parse_feed(response_text)
        entries = cast(list[dict[str, Any]], parsed.entries)

        articles: list[NewsArticle] = []
        for entry in entries[: max(limit * 3, limit)]:
            url = self._coerce_url(entry.get("link"))
            if url is None:
                continue

            published_at = self._parse_feed_datetime(entry)
            if published_at is None:
                continue

            summary = sanitize_text(str(entry.get("summary", ""))) or None
            title = sanitize_text(str(entry.get("title", "")))
            source = self._extract_feed_source(entry)
            if not title or not source:
                continue

            articles.append(
                NewsArticle(
                    title=title,
                    url=url,
                    source=source,
                    published_at=published_at,
                    summary=summary,
                    content_fingerprint=fingerprint_article(title, summary),
                    raw_payload=self._serialize_feed_entry(entry),
                )
            )

        return self._deduplicate_articles(articles, lookback_hours=lookback_hours)[:limit]

    def fetch_gdelt_articles(
        self,
        *,
        limit: int = 10,
        lookback_hours: int = 24,
        query: str = '"bitcoin" OR "btc"',
    ) -> list[NewsArticle]:
        params = {
            "query": query,
            "mode": "artlist",
            "sort": "datedesc",
            "maxrecords": str(limit),
            "format": "json",
        }
        payload = self._request_json(GDELT_DOC_API_URL, params=params)
        items = payload.get("articles")
        if not isinstance(items, list):
            return []

        articles: list[NewsArticle] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            url = self._coerce_url(item.get("url"))
            if url is None:
                continue

            published_at = self._parse_gdelt_datetime(item.get("seendate"))
            if published_at is None:
                continue

            title = sanitize_text(str(item.get("title", "")))
            summary = sanitize_text(str(item.get("seendescription", ""))) or None
            source = sanitize_text(str(item.get("domain", item.get("sourcecountry", "GDELT"))))
            if not title or not source:
                continue

            articles.append(
                NewsArticle(
                    title=title,
                    url=url,
                    source=source,
                    published_at=published_at,
                    summary=summary,
                    content_fingerprint=fingerprint_article(title, summary),
                    raw_payload=item,
                )
            )

        return self._deduplicate_articles(articles, lookback_hours=lookback_hours)[:limit]

    def _deduplicate_articles(
        self,
        articles: Iterable[NewsArticle],
        *,
        lookback_hours: int,
    ) -> list[NewsArticle]:
        cutoff = self._now_provider() - timedelta(hours=lookback_hours)
        deduped_by_key: dict[str, NewsArticle] = {}

        for article in articles:
            if article.published_at < cutoff:
                continue

            dedup_key = str(article.url)
            if article.content_fingerprint is not None:
                dedup_key = f"{dedup_key}|{article.content_fingerprint}"

            existing = deduped_by_key.get(dedup_key)
            if existing is None or article.published_at > existing.published_at:
                deduped_by_key[dedup_key] = article

        fingerprint_seen: set[str] = set()
        ordered = sorted(
            deduped_by_key.values(),
            key=lambda article: article.published_at,
            reverse=True,
        )

        filtered: list[NewsArticle] = []
        for article in ordered:
            fingerprint = article.content_fingerprint
            if fingerprint is not None and fingerprint in fingerprint_seen:
                continue
            if fingerprint is not None:
                fingerprint_seen.add(fingerprint)
            filtered.append(article)

        return filtered

    @retry_operation(httpx.HTTPError, NewsServiceError)
    def _request_text(self, url: str) -> str:
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NewsServiceError(self._request_error_message(url, exc)) from exc
        return response.text

    @retry_operation(httpx.HTTPError, NewsServiceError)
    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NewsServiceError(self._request_error_message(url, exc)) from exc

        try:
            return cast(dict[str, Any], response.json())
        except ValueError as exc:
            raise NewsServiceError(f"News response for {url} was not valid JSON.") from exc

    @staticmethod
    def _request_error_message(url: str, exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            return f"News request failed for {url} with status {exc.response.status_code}."
        return f"News request failed for {url}."

    @staticmethod
    def _source_warning(source_name: str, exc: NewsServiceError) -> str:
        message = str(exc)
        if "429" in message:
            return f"{source_name} rate-limited the request; continuing without that source."
        if "not valid JSON" in message:
            return (
                f"{source_name} returned an invalid response; continuing without that source."
            )
        return f"{source_name} fetch failed; continuing without that source."

    @staticmethod
    def _extract_feed_source(entry: dict[str, Any]) -> str:
        source = entry.get("source")
        if isinstance(source, dict):
            return sanitize_text(str(source.get("title", "")))
        return sanitize_text(str(source or "Google News"))

    @staticmethod
    def _coerce_url(value: Any) -> HttpUrl | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text.startswith(("http://", "https://")):
            return None
        return _HTTP_URL_ADAPTER.validate_python(text)

    @staticmethod
    def _parse_feed_datetime(entry: dict[str, Any]) -> datetime | None:
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed is not None:
            published = datetime(
                parsed.tm_year,
                parsed.tm_mon,
                parsed.tm_mday,
                parsed.tm_hour,
                parsed.tm_min,
                parsed.tm_sec,
                tzinfo=UTC,
            )
            return published

        raw_value = entry.get("published") or entry.get("updated")
        if raw_value is None:
            return None

        try:
            dt = parsedate_to_datetime(str(raw_value))
        except (TypeError, ValueError):
            return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _parse_gdelt_datetime(value: Any) -> datetime | None:
        if value is None:
            return None

        try:
            return datetime.strptime(str(value), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            return None

    @staticmethod
    def _serialize_feed_entry(entry: dict[str, Any]) -> dict[str, Any]:
        source = entry.get("source")
        source_title = source.get("title") if isinstance(source, dict) else source

        return {
            "title": entry.get("title"),
            "link": entry.get("link"),
            "published": entry.get("published"),
            "updated": entry.get("updated"),
            "summary": entry.get("summary"),
            "source": source_title,
        }
