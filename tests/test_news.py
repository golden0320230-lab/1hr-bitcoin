"""Tests for BTC news ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.services.news import GDELT_DOC_API_URL, NewsClient
from app.services.storage import DuckDBStorage
from app.utils.text import fingerprint_article, sanitize_text, truncate_text


def _build_http_client(*, rss_xml: str, gdelt_payload: dict[str, Any]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://news.google.com/rss/search"):
            return httpx.Response(200, text=rss_xml)

        if str(request.url).startswith(GDELT_DOC_API_URL):
            return httpx.Response(200, json=gdelt_payload)

        return httpx.Response(404, json={"error": "not found"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_text_utils_strip_html_and_truncate() -> None:
    cleaned = sanitize_text("<p> Bitcoin <strong>ETF</strong> inflows rise. </p>")

    assert cleaned == "Bitcoin ETF inflows rise."
    assert truncate_text(cleaned, max_chars=12) == "Bitcoin ETF…"
    assert fingerprint_article("Title", "<b>Summary</b>") == fingerprint_article("Title", "Summary")


def test_sanitize_text_removes_script_content() -> None:
    cleaned = sanitize_text(
        "<div>Headline<script>alert('x')</script><style>.x{}</style>Summary</div>"
    )

    assert cleaned == "Headline Summary"


def test_news_client_filters_stale_and_duplicate_articles_before_storage(tmp_path: Path) -> None:
    rss_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Bitcoin ETF inflows rise</title>
          <link>https://example.com/etf-inflows</link>
          <pubDate>Thu, 19 Mar 2026 18:30:00 GMT</pubDate>
          <description><![CDATA[<p>Fresh inflow headline</p>]]></description>
          <source>Example Wire</source>
        </item>
        <item>
          <title>Bitcoin ETF inflows rise</title>
          <link>https://example.com/etf-inflows-copy</link>
          <pubDate>Thu, 19 Mar 2026 18:35:00 GMT</pubDate>
          <description><![CDATA[<p>Fresh inflow headline</p>]]></description>
          <source>Example Wire</source>
        </item>
        <item>
          <title>Old Bitcoin article</title>
          <link>https://example.com/old-story</link>
          <pubDate>Wed, 18 Mar 2026 10:00:00 GMT</pubDate>
          <description><![CDATA[<p>Too old</p>]]></description>
          <source>Example Wire</source>
        </item>
      </channel>
    </rss>
    """
    gdelt_payload = {
        "articles": [
            {
                "title": "BTC miners expand operations",
                "url": "https://gdelt.example.com/miners-expand",
                "domain": "gdelt.example.com",
                "seendate": "20260319T183000Z",
                "seendescription": "Expansion plans may support sentiment.",
            }
        ]
    }
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    client = NewsClient(
        http_client=_build_http_client(rss_xml=rss_xml, gdelt_payload=gdelt_payload),
        storage=storage,
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        articles = client.fetch_recent_articles(limit=10, lookback_hours=12)

        assert len(articles) == 2
        assert [article.title for article in articles] == [
            "Bitcoin ETF inflows rise",
            "BTC miners expand operations",
        ]
        assert storage.count_rows("news_articles") == 2
    finally:
        client.close()
        storage.close()


def test_fetch_google_news_rss_and_gdelt_normalize_sources() -> None:
    rss_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Bitcoin rebounds after selloff</title>
          <link>https://example.com/rebound</link>
          <pubDate>Thu, 19 Mar 2026 18:30:00 GMT</pubDate>
          <description><![CDATA[<p>Bounce after dip</p>]]></description>
          <source>Reuters</source>
        </item>
      </channel>
    </rss>
    """
    gdelt_payload = {
        "articles": [
            {
                "title": "BTC options market heats up",
                "url": "https://gdelt.example.com/options",
                "domain": "coindesk.com",
                "seendate": "20260319T183500Z",
                "seendescription": "Options activity increased sharply.",
            }
        ]
    }
    client = NewsClient(
        http_client=_build_http_client(rss_xml=rss_xml, gdelt_payload=gdelt_payload),
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        google_articles = client.fetch_google_news_rss(limit=5, lookback_hours=6)
        gdelt_articles = client.fetch_gdelt_articles(limit=5, lookback_hours=6)

        assert google_articles[0].source == "Reuters"
        assert google_articles[0].summary == "Bounce after dip"
        assert gdelt_articles[0].source == "coindesk.com"
        assert gdelt_articles[0].summary == "Options activity increased sharply."
    finally:
        client.close()


def test_fetch_recent_articles_tolerates_gdelt_rate_limit() -> None:
    rss_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Bitcoin rebounds after selloff</title>
          <link>https://example.com/rebound</link>
          <pubDate>Thu, 19 Mar 2026 18:30:00 GMT</pubDate>
          <description><![CDATA[<p>Bounce after dip</p>]]></description>
          <source>Reuters</source>
        </item>
      </channel>
    </rss>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://news.google.com/rss/search"):
            return httpx.Response(200, text=rss_xml)
        if str(request.url).startswith(GDELT_DOC_API_URL):
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(404, json={"error": "not found"})

    client = NewsClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    try:
        articles = client.fetch_recent_articles(limit=5, lookback_hours=6)

        assert len(articles) == 1
        assert articles[0].source == "Reuters"
        assert client.last_warnings == [
            "GDELT rate-limited the request; continuing without that source."
        ]
    finally:
        client.close()
