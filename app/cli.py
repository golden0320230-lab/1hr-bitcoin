"""Typer CLI entrypoint for the Kalshi BTC tool."""

from __future__ import annotations

import logging as stdlib_logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

import orjson
import typer
from rich.console import Console
from rich.table import Table

from app import __version__
from app.config import Settings, get_settings
from app.constants import APP_NAME, CLI_NAME, RESEARCH_DISCLAIMER
from app.logging import configure_logging, get_logger
from app.schemas import ArticleSentimentScore, BTCCandle, FeatureVector, NewsArticle
from app.services.backtest import BacktestService
from app.services.coinbase import CoinbaseClient, CoinbaseServiceError
from app.services.explain import ExplainService
from app.services.features import FeatureBuilder
from app.services.kalshi import KalshiClient, KalshiServiceError
from app.services.kimiclaw import KimiClawClient, KimiClawServiceError
from app.services.news import NewsClient, NewsServiceError
from app.services.news_reviewers import CLINewsReviewerClient, CLIReviewerServiceError
from app.services.predictor import Predictor
from app.services.storage import DuckDBStorage
from app.services.training import ModelTrainer, TrainingDatasetBuilder
from app.utils.text import sanitize_text, truncate_text

app = typer.Typer(
    name=CLI_NAME,
    help="Research-only CLI for the Kalshi BTC 15-minute predictor.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


@dataclass(slots=True)
class Runtime:
    """Per-command runtime state stored on the Typer context."""

    settings: Settings
    logger: stdlib_logging.Logger
    storage: DuckDBStorage


def _build_runtime(log_level_override: str | None = None) -> Runtime:
    settings = get_settings()
    if log_level_override is not None:
        settings = settings.with_overrides(log_level=log_level_override)
    settings.ensure_runtime_directories()
    configure_logging(settings.log_level)
    return Runtime(
        settings=settings,
        logger=get_logger(),
        storage=DuckDBStorage(settings.db_path),
    )


def _get_runtime(ctx: typer.Context) -> Runtime:
    runtime = ctx.obj
    if isinstance(runtime, Runtime):
        return runtime
    raise RuntimeError("CLI context was not initialized.")


def _json_echo(payload: Mapping[str, object]) -> None:
    typer.echo(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))


def _is_kimiclaw_configured(settings: Settings) -> bool:
    return all(
        [
            settings.kimiclaw_base_url != "https://replace-me.example.com",
            settings.kimiclaw_api_key.get_secret_value() != "replace-me",
            settings.kimiclaw_model != "replace-me",
        ]
    )


def _render_market(market_payload: Mapping[str, object]) -> None:
    threshold = cast(float, market_payload["threshold"])
    yes_price = cast(float, market_payload["yes_price"])
    no_price = cast(float, market_payload["no_price"])
    table = Table(show_header=False, box=None)
    table.add_row("Ticker", str(market_payload["ticker"]))
    table.add_row("Question", str(market_payload["title"]))
    table.add_row("Threshold", f"${threshold:,.2f}")
    table.add_row("Direction", str(market_payload["direction"]))
    table.add_row("Expiry", str(market_payload["expires_at"]))
    table.add_row("Yes", f"{yes_price:.2%}")
    table.add_row("No", f"{no_price:.2%}")
    table.add_row("Implied prior", f"{yes_price:.2%}")
    console.print(table)


def _market_payload(
    market_ticker: str,
    title: str,
    threshold: float,
    direction: str,
    expires_at: str,
    yes_price: float,
    no_price: float,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        {
            "ticker": market_ticker,
            "title": title,
            "threshold": threshold,
            "direction": direction,
            "expires_at": expires_at,
            "yes_price": yes_price,
            "no_price": no_price,
        },
    )


def _render_warnings(warnings: list[str]) -> None:
    if not warnings:
        return

    console.print("Warnings:")
    for warning in warnings:
        console.print(f"- {warning}")


def _normalize_monitor_side(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"up", "down"}:
        raise typer.BadParameter("Side must be either 'up' or 'down'.", param_hint="side")
    return normalized


def _normalize_monitor_price(value: float) -> float:
    if 0 <= value <= 1:
        return value
    if 1 < value <= 100:
        return round(value / 100, 4)
    raise typer.BadParameter(
        "Target price must be between 0 and 1, or between 1 and 100 cents.",
        param_hint="target_price",
    )


def _monitor_side_price(side: str, yes_price: float, no_price: float) -> float:
    return yes_price if side == "up" else no_price


def _normalize_reviewer(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"kimiclaw", "claude", "codex"}:
        raise typer.BadParameter(
            "Reviewer must be one of: kimiclaw, claude, codex.",
            param_hint="--reviewer",
        )
    return normalized


def _reviewer_label(reviewer: str) -> str:
    return {
        "kimiclaw": "KimiClaw",
        "claude": "Claude Code",
        "codex": "Codex",
    }[reviewer]


def _reviewer_fallback_used(scores: list[ArticleSentimentScore]) -> bool:
    return any(score.raw_response == {"fallback": True} for score in scores)


def _kimiclaw_fallback_used(scores: list[ArticleSentimentScore]) -> bool:
    return any(score.raw_response == {"fallback": True} for score in scores)


def _score_articles_with_warnings(
    runtime: Runtime,
    articles: list[NewsArticle],
    *,
    reviewer: str = "kimiclaw",
    reviewer_model: str | None = None,
) -> tuple[list[ArticleSentimentScore], list[str]]:
    if not articles:
        return [], ["No recent BTC-related articles found; using neutral news contribution."]

    normalized_reviewer = _normalize_reviewer(reviewer)
    warnings: list[str] = []
    capped_articles = articles
    if normalized_reviewer in {"claude", "codex"} and len(articles) > 5:
        capped_articles = articles[:5]
        warnings.append(
            f"{_reviewer_label(normalized_reviewer)} reviewer capped at 5 articles; "
            "using newest articles only."
        )

    if normalized_reviewer == "kimiclaw":
        if not _is_kimiclaw_configured(runtime.settings):
            return [], ["KimiClaw is not configured; using neutral news contribution."]

        with KimiClawClient(
            base_url=runtime.settings.kimiclaw_base_url,
            api_key=runtime.settings.kimiclaw_api_key.get_secret_value(),
            model_name=reviewer_model or runtime.settings.kimiclaw_model,
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as client:
            try:
                scores = client.score_articles(capped_articles)
            except KimiClawServiceError:
                return [], warnings + ["KimiClaw unavailable; using neutral news contribution."]

        if _kimiclaw_fallback_used(scores):
            warnings.append(
                "KimiClaw unavailable or invalid output; using neutral news contribution."
            )
        return scores, warnings

    provider: Literal["codex", "claude"] = (
        "codex" if normalized_reviewer == "codex" else "claude"
    )
    with CLINewsReviewerClient(
        provider=provider,
        model_name=reviewer_model,
        storage=runtime.storage,
        timeout_seconds=max(runtime.settings.http_timeout_seconds, 60.0),
    ) as client:
        try:
            scores = client.score_articles(capped_articles)
        except CLIReviewerServiceError:
            return [], warnings + [
                f"{_reviewer_label(normalized_reviewer)} reviewer unavailable; "
                "using neutral news contribution."
            ]

    if _reviewer_fallback_used(scores):
        warnings.append(
            f"{_reviewer_label(normalized_reviewer)} reviewer unavailable or invalid output; "
            "using neutral news contribution."
        )
    return scores, warnings


def _render_review_table(
    articles: list[NewsArticle],
    scores: list[ArticleSentimentScore],
) -> None:
    article_by_url = {str(article.url): article for article in articles}
    table = Table(title="BTC News Review")
    table.add_column("Published")
    table.add_column("Source")
    table.add_column("Reviewer")
    table.add_column("Call")
    table.add_column("Sentiment")
    table.add_column("Impact")
    table.add_column("Conf.")
    table.add_column("Reason")

    for score in scores:
        article = article_by_url.get(str(score.article_url))
        published = article.published_at.strftime("%Y-%m-%d %H:%M") if article is not None else "-"
        source = article.source if article is not None else "-"
        table.add_row(
            published,
            source,
            score.model_name,
            score.market_call,
            score.sentiment,
            f"{score.impact_score:+.2f}",
            f"{score.confidence:.2f}",
            truncate_text(sanitize_text(score.reason), max_chars=120),
        )

    console.print(table)


def _build_review_summary(scores: list[ArticleSentimentScore]) -> dict[str, object] | None:
    if not scores:
        return None

    up_weight = 0.0
    down_weight = 0.0
    neutral_weight = 0.0
    up_count = 0
    down_count = 0
    neutral_count = 0

    for score in scores:
        weight = score.relevance * score.confidence
        if score.market_call == "UP":
            up_weight += weight
            up_count += 1
        elif score.market_call == "DOWN":
            down_weight += weight
            down_count += 1
        else:
            neutral_weight += weight
            neutral_count += 1

    net_score = up_weight - down_weight
    if net_score > 0.05:
        market_call = "UP"
    elif net_score < -0.05:
        market_call = "DOWN"
    else:
        market_call = "NEUTRAL"

    return {
        "market_call": market_call,
        "net_score": round(net_score, 4),
        "up_count": up_count,
        "down_count": down_count,
        "neutral_count": neutral_count,
        "up_weight": round(up_weight, 4),
        "down_weight": round(down_weight, 4),
        "neutral_weight": round(neutral_weight, 4),
    }


def _load_cached_candles(storage: DuckDBStorage) -> list[BTCCandle]:
    for timeframe in ("1m", "5m", "15m"):
        candles = storage.list_recent_candles(source="coinbase", timeframe=timeframe, limit=90)
        if candles:
            return candles
    return []


def _load_price_model_probability(
    runtime: Runtime,
    features: FeatureVector,
) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    model_path = runtime.settings.model_path
    if not model_path.exists():
        warnings.append("Model artifact missing; using heuristic price model.")
        return None, warnings

    try:
        artifact = ModelTrainer.load_artifact(model_path)
        probability = ModelTrainer().predict_feature_probability(artifact, features)
    except (OSError, ValueError, KeyError, TypeError):
        warnings.append("Model artifact could not be loaded; using heuristic price model.")
        return None, warnings

    return probability, warnings


@app.callback()
def root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed CLI version and exit.",
        is_eager=True,
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Override the configured log level for the current command.",
    ),
) -> None:
    """Initialize common runtime state for all commands."""

    if version:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit()

    ctx.obj = _build_runtime(log_level)


@app.command()
def market(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Show the live Kalshi BTC 15-minute market."""

    runtime = _get_runtime(ctx)
    try:
        with KalshiClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as client:
            discovered = client.get_live_btc_market()
    except KalshiServiceError:
        payload = {
            "market": None,
            "warnings": ["Kalshi market discovery failed."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("Kalshi market discovery failed.")
            console.print(RESEARCH_DISCLAIMER)
        return

    if discovered is None:
        no_market_payload: dict[str, object] = {
            "market": None,
            "warnings": ["No live BTC 15-minute Kalshi market found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(no_market_payload)
        else:
            console.print("No live BTC 15-minute Kalshi market found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    market_model, snapshot = discovered
    market_payload = _market_payload(
        market_ticker=market_model.ticker,
        title=market_model.title,
        threshold=market_model.threshold,
        direction=market_model.direction,
        expires_at=market_model.expires_at.isoformat(),
        yes_price=snapshot.yes_price or 0.0,
        no_price=snapshot.no_price or 0.0,
    )

    if json_output:
        _json_echo(
            {
                "market": market_payload,
                "warnings": [],
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
    else:
        _render_market(market_payload)
        console.print(RESEARCH_DISCLAIMER)


@app.command()
def monitor(
    ctx: typer.Context,
    side: str = typer.Argument(..., help="Which side price to monitor: up or down."),
    target_price: float = typer.Argument(
        ...,
        help="Stop when the watched side reaches this price. Accepts 0-1 or 1-100 cents.",
    ),
    poll_seconds: int = typer.Option(
        20,
        "--poll-seconds",
        min=15,
        help="Seconds between Kalshi checks. Minimum 15 to avoid over-polling.",
    ),
    max_checks: int = typer.Option(
        180,
        "--max-checks",
        min=1,
        help="Maximum number of Kalshi polls before stopping.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Monitor the live Kalshi BTC 15-minute market until a side reaches a target price."""

    runtime = _get_runtime(ctx)
    normalized_side = _normalize_monitor_side(side)
    normalized_target_price = _normalize_monitor_price(target_price)
    warnings: list[str] = []
    last_market_payload: dict[str, object] | None = None
    last_observed_price: float | None = None

    with KalshiClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as client:
        for check in range(1, max_checks + 1):
            try:
                discovered = client.get_live_btc_market()
            except KalshiServiceError:
                warning = "Kalshi market discovery failed during monitoring."
                if warning not in warnings:
                    warnings.append(warning)
                discovered = None

            if discovered is None:
                if not json_output:
                    console.print(
                        f"[{check}/{max_checks}] No live BTC 15-minute Kalshi market found."
                    )
            else:
                market_model, snapshot = discovered
                yes_price = snapshot.yes_price or 0.0
                no_price = snapshot.no_price or 0.0
                watched_price = _monitor_side_price(normalized_side, yes_price, no_price)
                last_observed_price = watched_price
                last_market_payload = _market_payload(
                    market_ticker=market_model.ticker,
                    title=market_model.title,
                    threshold=market_model.threshold,
                    direction=market_model.direction,
                    expires_at=market_model.expires_at.isoformat(),
                    yes_price=yes_price,
                    no_price=no_price,
                )

                if not json_output:
                    minutes_to_expiry = max(
                        int((market_model.expires_at - snapshot.captured_at).total_seconds() // 60),
                        0,
                    )
                    console.print(
                        f"[{check}/{max_checks}] {market_model.ticker} | "
                        f"up {yes_price:.2%} | down {no_price:.2%} | "
                        f"watching {normalized_side} for {normalized_target_price:.2%} | "
                        f"expires in {minutes_to_expiry}m"
                    )

                if watched_price >= normalized_target_price:
                    payload = {
                        "status": "hit",
                        "target_side": normalized_side,
                        "target_price": normalized_target_price,
                        "observed_price": watched_price,
                        "checks": check,
                        "market": last_market_payload,
                        "warnings": warnings,
                        "disclaimer": RESEARCH_DISCLAIMER,
                    }
                    if json_output:
                        _json_echo(payload)
                    else:
                        console.print(
                            f"Target hit: {normalized_side} reached {watched_price:.2%} "
                            f"(target {normalized_target_price:.2%})."
                        )
                        _render_market(last_market_payload)
                        _render_warnings(warnings)
                        console.print(RESEARCH_DISCLAIMER)
                    return

            if check < max_checks:
                time.sleep(poll_seconds)

    if last_market_payload is None:
        warnings.append("No live BTC 15-minute Kalshi market found during monitoring.")
    else:
        warnings.append("Target price was not reached before monitoring stopped.")

    payload = {
        "status": "not_hit",
        "target_side": normalized_side,
        "target_price": normalized_target_price,
        "observed_price": last_observed_price,
        "checks": max_checks,
        "market": last_market_payload,
        "warnings": warnings,
        "disclaimer": RESEARCH_DISCLAIMER,
    }
    if json_output:
        _json_echo(payload)
    else:
        console.print(
            f"Monitor stopped after {max_checks} checks without reaching "
            f"{normalized_target_price:.2%} on {normalized_side}."
        )
        if last_market_payload is not None:
            _render_market(last_market_payload)
        _render_warnings(warnings)
        console.print(RESEARCH_DISCLAIMER)


@app.command()
def news(
    ctx: typer.Context,
    limit: int | None = typer.Option(None, "--limit", min=1, help="Maximum articles to return."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    score: bool = typer.Option(
        True,
        "--score/--no-score",
        help="Score articles with the selected reviewer.",
    ),
    reviewer: str = typer.Option(
        "kimiclaw",
        "--reviewer",
        help="Reviewer to score articles with: kimiclaw, claude, or codex.",
    ),
    reviewer_model: str | None = typer.Option(
        None,
        "--reviewer-model",
        help="Optional model override for the selected reviewer.",
    ),
) -> None:
    """Fetch and display recent BTC-related news."""

    runtime = _get_runtime(ctx)
    effective_limit = limit or runtime.settings.news_article_limit
    warnings: list[str] = []

    try:
        with NewsClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as client:
            articles = client.fetch_recent_articles(limit=effective_limit, lookback_hours=24)
            warnings.extend(client.last_warnings)
    except NewsServiceError:
        articles = []
        warnings.append("News fetch failed; returning no recent articles.")

    scores: list[ArticleSentimentScore] = []
    if score and articles:
        scores, score_warnings = _score_articles_with_warnings(
            runtime,
            articles,
            reviewer=reviewer,
            reviewer_model=reviewer_model,
        )
        warnings.extend(score_warnings)
    review_summary = _build_review_summary(scores)

    if json_output:
        _json_echo(
            {
                "articles": [article.model_dump(mode="json") for article in articles],
                "scores": [score.model_dump(mode="json") for score in scores],
                "reviewer": _normalize_reviewer(reviewer) if score else None,
                "review_summary": review_summary,
                "warnings": warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    if scores:
        if review_summary is not None:
            console.print(
                "Reviewer market call: "
                f"{review_summary['market_call']} "
                f"(up {review_summary['up_count']}, "
                f"down {review_summary['down_count']}, "
                f"neutral {review_summary['neutral_count']})"
            )
        _render_review_table(articles, scores)
    else:
        table = Table(title="BTC News")
        table.add_column("Published")
        table.add_column("Source")
        table.add_column("Title")

        for article in articles:
            table.add_row(
                article.published_at.strftime("%Y-%m-%d %H:%M"),
                article.source,
                article.title,
            )

        console.print(table)
    _render_warnings(warnings)
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def predict(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    news_limit: int | None = typer.Option(None, "--news-limit", min=1, help="Article limit."),
    no_news: bool = typer.Option(False, "--no-news", help="Disable news ingestion and scoring."),
    reviewer: str = typer.Option(
        "kimiclaw",
        "--reviewer",
        help="News reviewer to use when scoring: kimiclaw, claude, or codex.",
    ),
    reviewer_model: str | None = typer.Option(
        None,
        "--reviewer-model",
        help="Optional model override for the selected reviewer.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show extra pipeline details."),
    save_run: bool = typer.Option(False, "--save-run", help="Persist the prediction to DuckDB."),
) -> None:
    """Run the research-only ABOVE or BELOW prediction flow."""

    runtime = _get_runtime(ctx)
    warnings: list[str] = []

    try:
        with KalshiClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as kalshi_client:
            discovered = kalshi_client.get_live_btc_market()
    except KalshiServiceError:
        payload = {
            "prediction": None,
            "warnings": ["Kalshi market discovery failed."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("Kalshi market discovery failed.")
            console.print(RESEARCH_DISCLAIMER)
        return

    if discovered is None:
        payload = {
            "prediction": None,
            "warnings": ["No live BTC 15-minute Kalshi market found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("No live BTC 15-minute Kalshi market found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    market_model, snapshot = discovered

    spot_price: float | None = None
    candles: list[BTCCandle] = []
    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            spot_price = coinbase_client.get_spot_price()
        except CoinbaseServiceError:
            warnings.append(
                "BTC spot fetch failed; using latest available candle close if possible."
            )

        try:
            candles = coinbase_client.get_candles(lookback_minutes=90, timeframe="1m")
        except CoinbaseServiceError:
            warnings.append("BTC candle fetch failed; using cached BTC candles if available.")

    if not candles:
        candles = _load_cached_candles(runtime.storage)
        if candles:
            warnings.append("Using cached BTC candles for feature generation.")

    if spot_price is None and candles:
        spot_price = candles[-1].close
        warnings.append("Using latest cached candle close as BTC spot price.")

    if spot_price is None or not candles:
        error_warnings = warnings + ["BTC fetch failed and no cached data was available."]
        payload = {
            "prediction": None,
            "warnings": error_warnings,
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            for warning in error_warnings:
                console.print(warning)
            console.print(RESEARCH_DISCLAIMER)
        return

    articles = []
    scores: list[ArticleSentimentScore] = []
    if not no_news:
        try:
            with NewsClient(
                storage=runtime.storage,
                timeout_seconds=runtime.settings.http_timeout_seconds,
            ) as news_client:
                articles = news_client.fetch_recent_articles(
                    limit=news_limit or runtime.settings.news_article_limit,
                    lookback_hours=24,
                )
                warnings.extend(news_client.last_warnings)
        except NewsServiceError:
            warnings.append("News fetch failed; using neutral news contribution.")
            articles = []

        if articles:
            scores, score_warnings = _score_articles_with_warnings(
                runtime,
                articles,
                reviewer=reviewer,
                reviewer_model=reviewer_model,
            )
            warnings.extend(score_warnings)
        elif not any("fetch failed" in warning.lower() for warning in warnings) and not any(
            "rate-limited" in warning.lower() for warning in warnings
        ):
            warnings.append(
                "No recent BTC-related articles found; using neutral news contribution."
            )
    else:
        warnings.append("News scoring disabled; using neutral news contribution.")
    review_summary = _build_review_summary(scores)

    features = FeatureBuilder().build_feature_vector(
        market=market_model,
        snapshot=snapshot,
        spot_price=spot_price,
        candles=candles,
        news_scores=scores,
        generated_at=snapshot.captured_at,
    )
    price_model_probability, model_warnings = _load_price_model_probability(runtime, features)
    warnings.extend(model_warnings)
    result = Predictor().predict(
        market=market_model,
        snapshot=snapshot,
        features=features,
        price_model_probability=price_model_probability,
    )

    if warnings:
        result = result.model_copy(
            update={
                "warnings": result.warnings
                + [warning for warning in warnings if warning not in result.warnings],
                "degraded": True,
            }
        )

    if save_run:
        runtime.storage.insert_prediction_run(result)

    if json_output:
        _json_echo(
            {
                "prediction": result.model_dump(mode="json"),
                "articles": [article.model_dump(mode="json") for article in articles],
                "news_reviewer": _normalize_reviewer(reviewer) if not no_news else None,
                "news_review_summary": review_summary,
                "warnings": result.warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    console.print(f"Live market: {market_model.title}")
    console.print(f"Kalshi ticker: {market_model.ticker}")
    console.print(f"BTC spot: ${spot_price:,.2f}")
    console.print(f"Distance to strike: {features.distance_to_strike:+.2f}")
    console.print(f"Time to expiry: {features.minutes_to_expiry}m")
    if review_summary is not None:
        console.print(f"News reviewer call: {review_summary['market_call']}")
    console.print()
    console.print(f"Prediction: {result.label}")
    console.print(f"Probability: {result.probability:.2%}")
    console.print(f"Confidence: {result.confidence}")
    console.print("Top drivers:")
    for driver in result.drivers:
        console.print(f"- {driver}")

    _render_warnings(result.warnings)
    if verbose:
        console.print(f"Articles considered: {len(articles)}")

    console.print(RESEARCH_DISCLAIMER)


@app.command()
def train(
    ctx: typer.Context,
    days: int = typer.Option(90, "--days", min=7, help="Historical window for training."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Train or refresh the local baseline model."""

    runtime = _get_runtime(ctx)
    dataset_path = runtime.settings.db_path.parent / "training_dataset.csv"
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)

    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            candles = coinbase_client.get_candles_range(
                start_at=start_time,
                end_at=end_time,
                timeframe="5m",
            )
        except CoinbaseServiceError as exc:
            console.print(f"Unable to fetch training candles: {exc}")
            console.print(RESEARCH_DISCLAIMER)
            raise typer.Exit(code=1) from exc

    dataset_builder = TrainingDatasetBuilder(dataset_path=dataset_path, strike_increment=0.01)
    dataset = dataset_builder.build_dataset(candles, horizon_minutes=15, step_candles=1)
    saved_dataset = dataset_builder.save_dataset(dataset)

    try:
        result = ModelTrainer(runtime.storage).train(
            dataset,
            artifact_path=runtime.settings.model_path,
        )
    except ValueError as exc:
        console.print(f"Unable to train model: {exc}")
        console.print(RESEARCH_DISCLAIMER)
        raise typer.Exit(code=1) from exc

    if json_output:
        _json_echo(
            {
                "training": {
                    "model_name": result.model_name,
                    "artifact_path": str(result.artifact_path),
                    "dataset_path": str(saved_dataset.path),
                    "dataset_rows": result.dataset_rows,
                    "feature_schema_version": result.feature_schema_version,
                    "training_window_start": result.training_window_start.isoformat(),
                    "training_window_end": result.training_window_end.isoformat(),
                    "metrics": result.metrics,
                },
                "warnings": [],
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    console.print(f"Training rows: {result.dataset_rows}")
    console.print(f"Dataset saved: {saved_dataset.path}")
    console.print(f"Selected model: {result.model_name}")
    console.print(f"Artifact saved: {result.artifact_path}")
    console.print(
        "Validation metrics: "
        f"accuracy {result.metrics[result.model_name]['accuracy']:.2%}, "
        f"log loss {result.metrics[result.model_name]['log_loss']:.4f}, "
        f"Brier {result.metrics[result.model_name]['brier_score']:.4f}"
    )
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def backtest(
    ctx: typer.Context,
    days: int = typer.Option(30, "--days", min=7, help="Historical window for evaluation."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Run historical evaluations against archived windows."""

    runtime = _get_runtime(ctx)
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)

    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            candles = coinbase_client.get_candles_range(
                start_at=start_time,
                end_at=end_time,
                timeframe="5m",
            )
        except CoinbaseServiceError as exc:
            console.print(f"Unable to fetch backtest candles: {exc}")
            console.print(RESEARCH_DISCLAIMER)
            raise typer.Exit(code=1) from exc

    dataset = TrainingDatasetBuilder(strike_increment=0.01).build_dataset(
        candles,
        horizon_minutes=15,
    )

    try:
        result = BacktestService(runtime.storage).run(dataset)
    except ValueError as exc:
        console.print(f"Unable to run backtest: {exc}")
        console.print(RESEARCH_DISCLAIMER)
        raise typer.Exit(code=1) from exc

    if json_output:
        _json_echo(
            {
                "backtest": result.model_dump(mode="json"),
                "warnings": [],
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    console.print(f"Backtest model: {result.model_name}")
    console.print(f"Backtest samples: {result.num_samples}")
    console.print(f"Accuracy: {result.accuracy:.2%}")
    console.print(f"Log loss: {result.log_loss:.4f}")
    if result.brier_score is not None:
        console.print(f"Brier score: {result.brier_score:.4f}")
    console.print("Baselines:")
    for baseline in result.baselines:
        console.print(
            f"- {baseline.name}: accuracy {baseline.accuracy:.2%}, "
            f"log loss {baseline.log_loss:.4f}, "
            f"Brier {baseline.brier_score:.4f}"
        )
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def explain(
    ctx: typer.Context,
    last: bool = typer.Option(
        True,
        "--last/--no-last",
        help="Show the latest saved prediction run.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Inspect the latest saved prediction run."""

    runtime = _get_runtime(ctx)
    if not last:
        raise typer.BadParameter("Only --last is currently supported.", param_hint="--last")

    service = ExplainService(runtime.storage)
    prediction = service.get_last_prediction()

    if prediction is None:
        payload = {
            "prediction": None,
            "warnings": ["No saved prediction run found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("No saved prediction run found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    if json_output:
        _json_echo(
            {
                "prediction": prediction.model_dump(mode="json"),
                "warnings": prediction.warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    market = prediction.market
    features = prediction.feature_vector
    if market is not None:
        console.print(f"Live market: {market.title}")
        console.print(f"Kalshi ticker: {market.ticker}")
    console.print(f"Prediction: {prediction.label}")
    console.print(f"Probability: {prediction.probability:.2%}")
    console.print(f"Confidence: {prediction.confidence}")
    console.print("Main drivers:")
    for driver in prediction.drivers:
        console.print(f"- {driver}")
    if features is not None:
        console.print("Feature values:")
        console.print(f"- Spot price: ${features.spot_price:,.2f}")
        console.print(f"- Strike price: ${features.strike_price:,.2f}")
        console.print(f"- Distance to strike: {features.distance_to_strike:+.2f}")
        console.print(f"- Market implied probability: {features.market_implied_probability:.2%}")
        console.print(f"- Return 15m: {features.return_15m:+.2%}")
        console.print(f"- Return 60m: {features.return_60m:+.2%}")
        console.print(f"- News weighted impact: {features.news_weighted_impact:+.2f}")
        console.print(f"- Minutes to expiry: {features.minutes_to_expiry}")
    _render_warnings(prediction.warnings)
    console.print(RESEARCH_DISCLAIMER)


def main() -> None:
    """Run the CLI application."""

    app()
