"""Smoke tests for the Typer CLI."""

from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


def test_root_help_shows_bootstrap_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Research-only CLI for the Kalshi BTC 1-hour predictor." in result.output
    assert "market" in result.output
    assert "predict" in result.output


def test_market_help_works() -> None:
    result = runner.invoke(app, ["market", "--help"])

    assert result.exit_code == 0
    assert "Show the live Kalshi BTC hourly market." in result.output
