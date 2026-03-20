"""Typer CLI entrypoint for the Kalshi BTC tool."""

from __future__ import annotations

import logging as stdlib_logging
from dataclasses import dataclass

import typer
from rich.console import Console

from app import __version__
from app.config import Settings, get_settings
from app.constants import APP_NAME, CLI_NAME, PLACEHOLDER_MESSAGE, RESEARCH_DISCLAIMER
from app.logging import configure_logging, get_logger

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


def _build_runtime(log_level_override: str | None = None) -> Runtime:
    settings = get_settings()
    if log_level_override is not None:
        settings = settings.with_overrides(log_level=log_level_override)
    settings.ensure_runtime_directories()
    configure_logging(settings.log_level)
    return Runtime(settings=settings, logger=get_logger())


def _get_runtime(ctx: typer.Context) -> Runtime:
    runtime = ctx.obj
    if isinstance(runtime, Runtime):
        return runtime
    raise RuntimeError("CLI context was not initialized.")


def _emit_placeholder(ctx: typer.Context, command_name: str, summary: str) -> None:
    runtime = _get_runtime(ctx)
    runtime.logger.info("Running placeholder command: %s", command_name)
    console.print(f"{command_name}: {summary}")
    console.print(PLACEHOLDER_MESSAGE)
    console.print(RESEARCH_DISCLAIMER)


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
def market(ctx: typer.Context) -> None:
    """Show the live Kalshi BTC hourly market."""

    _emit_placeholder(ctx, "market", "Live market discovery is scheduled for issue 4.")


@app.command()
def predict(ctx: typer.Context) -> None:
    """Run the research-only ABOVE or BELOW prediction flow."""

    _emit_placeholder(ctx, "predict", "The prediction pipeline is scheduled for issue 10.")


@app.command()
def news(ctx: typer.Context) -> None:
    """Fetch and display recent BTC-related news."""

    _emit_placeholder(ctx, "news", "News ingestion is scheduled for issue 6.")


@app.command()
def train(ctx: typer.Context) -> None:
    """Train or refresh the local baseline model."""

    _emit_placeholder(ctx, "train", "Training is scheduled for issue 13.")


@app.command()
def backtest(ctx: typer.Context) -> None:
    """Run historical evaluations against archived windows."""

    _emit_placeholder(ctx, "backtest", "Backtesting is scheduled for issue 14.")


@app.command()
def explain(ctx: typer.Context) -> None:
    """Inspect the latest saved prediction run."""

    _emit_placeholder(ctx, "explain", "Prediction explanations are scheduled for issue 11.")


def main() -> None:
    """Run the CLI application."""

    app()

