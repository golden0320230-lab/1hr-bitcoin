"""Typer CLI entrypoint for the Kalshi BTC tool."""

from __future__ import annotations

import logging as stdlib_logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import orjson
import typer
from rich.console import Console
from rich.table import Table

from app import __version__
from app.config import Settings, get_settings
from app.constants import APP_NAME, CLI_NAME, RESEARCH_DISCLAIMER
from app.logging import configure_logging, get_logger
from app.services.coinbase import CoinbaseClient, CoinbaseServiceError
from app.services.explain import ExplainService
from app.services.features import FeatureBuilder
from app.services.kalshi import KalshiClient
from app.services.kimiclaw import KimiClawClient
from app.services.news import NewsClient
from app.services.predictor import Predictor
from app.services.storage import DuckDBStorage

app = typer.Typer(
    name=CLI_NAME,
    help="Research-only CLI for the Kalshi BTC 1-hour predictor.",
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
    console.print(table)
    console.print(RESEARCH_DISCLAIMER)


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
    """Show the live Kalshi BTC hourly market."""

    runtime = _get_runtime(ctx)
    with KalshiClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as client:
        discovered = client.get_live_btc_hourly_market()

    if discovered is None:
        no_market_payload: dict[str, object] = {
            "market": None,
            "warnings": ["No live BTC hourly Kalshi market found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(no_market_payload)
        else:
            console.print("No live BTC hourly Kalshi market found.")
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
        _json_echo({"market": market_payload, "disclaimer": RESEARCH_DISCLAIMER})
    else:
        _render_market(market_payload)


@app.command()
def news(
    ctx: typer.Context,
    limit: int | None = typer.Option(None, "--limit", min=1, help="Maximum articles to return."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    score: bool = typer.Option(True, "--score/--no-score", help="Score articles with KimiClaw."),
) -> None:
    """Fetch and display recent BTC-related news."""

    runtime = _get_runtime(ctx)
    effective_limit = limit or runtime.settings.news_article_limit
    warnings: list[str] = []

    with NewsClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as client:
        articles = client.fetch_recent_articles(limit=effective_limit, lookback_hours=24)

    scores = []
    if score and articles and _is_kimiclaw_configured(runtime.settings):
        with KimiClawClient(
            base_url=runtime.settings.kimiclaw_base_url,
            api_key=runtime.settings.kimiclaw_api_key.get_secret_value(),
            model_name=runtime.settings.kimiclaw_model,
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as client:
            scores = client.score_articles(articles)
    elif score and articles:
        warnings.append("KimiClaw is not configured; returning unscored articles.")

    if json_output:
        _json_echo(
            {
                "articles": [article.model_dump(mode="json") for article in articles],
                "scores": [score.model_dump(mode="json") for score in scores],
                "warnings": warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    table = Table(title="BTC News")
    table.add_column("Published")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("Sentiment")
    score_by_url = {str(score.article_url): score.sentiment for score in scores}

    for article in articles:
        table.add_row(
            article.published_at.strftime("%Y-%m-%d %H:%M"),
            article.source,
            article.title,
            score_by_url.get(str(article.url), "unscored"),
        )

    console.print(table)
    for warning in warnings:
        console.print(warning)
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def predict(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    news_limit: int | None = typer.Option(None, "--news-limit", min=1, help="Article limit."),
    no_news: bool = typer.Option(False, "--no-news", help="Disable news ingestion and scoring."),
    verbose: bool = typer.Option(False, "--verbose", help="Show extra pipeline details."),
    save_run: bool = typer.Option(False, "--save-run", help="Persist the prediction to DuckDB."),
) -> None:
    """Run the research-only ABOVE or BELOW prediction flow."""

    runtime = _get_runtime(ctx)
    warnings: list[str] = []

    with KalshiClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as kalshi_client:
        discovered = kalshi_client.get_live_btc_hourly_market()

    if discovered is None:
        payload = {
            "prediction": None,
            "warnings": ["No live BTC hourly Kalshi market found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("No live BTC hourly Kalshi market found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    market_model, snapshot = discovered

    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            spot_price = coinbase_client.get_spot_price()
            candles = coinbase_client.get_candles(lookback_minutes=90, timeframe="1m")
        except CoinbaseServiceError as exc:
            raise typer.Exit(code=1) from exc

    articles = []
    scores = []
    if not no_news:
        with NewsClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as news_client:
            articles = news_client.fetch_recent_articles(
                limit=news_limit or runtime.settings.news_article_limit,
                lookback_hours=24,
            )
        if articles and _is_kimiclaw_configured(runtime.settings):
            with KimiClawClient(
                base_url=runtime.settings.kimiclaw_base_url,
                api_key=runtime.settings.kimiclaw_api_key.get_secret_value(),
                model_name=runtime.settings.kimiclaw_model,
                storage=runtime.storage,
                timeout_seconds=runtime.settings.http_timeout_seconds,
            ) as kimi_client:
                scores = kimi_client.score_articles(articles)
        elif articles:
            warnings.append("KimiClaw is not configured; using neutral news contribution.")
        else:
            warnings.append(
                "No recent BTC-related articles found; using neutral news contribution."
            )
    else:
        warnings.append("News scoring disabled; using neutral news contribution.")

    features = FeatureBuilder().build_feature_vector(
        market=market_model,
        snapshot=snapshot,
        spot_price=spot_price,
        candles=candles,
        news_scores=scores,
        generated_at=snapshot.captured_at,
    )
    result = Predictor().predict(market=market_model, snapshot=snapshot, features=features)

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
    console.print()
    console.print(f"Prediction: {result.label}")
    console.print(f"Probability: {result.probability:.2%}")
    console.print(f"Confidence: {result.confidence}")
    console.print("Top drivers:")
    for driver in result.drivers:
        console.print(f"- {driver}")

    if verbose and result.warnings:
        console.print("Warnings:")
        for warning in result.warnings:
            console.print(f"- {warning}")
        console.print(f"Articles considered: {len(articles)}")

    console.print(RESEARCH_DISCLAIMER)


@app.command()
def train(ctx: typer.Context) -> None:
    """Train or refresh the local baseline model."""

    runtime = _get_runtime(ctx)
    runtime.logger.info("train command will be implemented in a later issue.")
    console.print("train: Local model training arrives in issue 13.")
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def backtest(ctx: typer.Context) -> None:
    """Run historical evaluations against archived windows."""

    runtime = _get_runtime(ctx)
    runtime.logger.info("backtest command will be implemented in a later issue.")
    console.print("backtest: Historical evaluation arrives in issue 14.")
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def explain(
    ctx: typer.Context,
    last: bool = typer.Option(
        True,
        "--last/--no-last",
        help="Show the latest saved prediction run.",
    ),
) -> None:
    """Inspect the latest saved prediction run."""

    runtime = _get_runtime(ctx)
    if not last:
        raise typer.BadParameter("Only --last is currently supported.", param_hint="--last")

    service = ExplainService(runtime.storage)
    prediction = service.get_last_prediction()

    if prediction is None:
        console.print("No saved prediction run found.")
        console.print(RESEARCH_DISCLAIMER)
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
    if prediction.warnings:
        console.print("Warnings:")
        for warning in prediction.warnings:
            console.print(f"- {warning}")
    console.print(RESEARCH_DISCLAIMER)


def main() -> None:
    """Run the CLI application."""

    app()
