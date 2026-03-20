"""Tests for local CLI-backed news reviewers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from app.schemas import NewsArticle
from app.services import news_reviewers
from app.services.news_reviewers import CLINewsReviewerClient, CLIReviewerServiceError


def _article() -> NewsArticle:
    return NewsArticle(
        title="Bitcoin ETF inflows rise again",
        url="https://example.com/bitcoin-etf",
        source="Example Wire",
        published_at="2026-03-19T18:30:00Z",
        summary="Fresh ETF inflows may support near-term BTC sentiment.",
    )


def test_claude_reviewer_parses_structured_json() -> None:
    commands: list[list[str]] = []

    def runner(
        command: list[str],
        workspace_dir: Path,
        timeout_seconds: float,
        input_text: str | None,
    ) -> CompletedProcess[str]:
        commands.append(command)
        assert workspace_dir.name == "Bitcoin-1hr-checker"
        assert timeout_seconds == 60.0
        assert input_text is None
        return CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                '{"market_call":"UP","sentiment":"bullish","relevance":0.81,'
                '"impact_horizon_minutes":45,"impact_score":0.34,'
                '"confidence":0.72,"reason":"ETF flow may support BTC."}'
            ),
            stderr="",
        )

    client = CLINewsReviewerClient(
        provider="claude",
        model_name="claude-sonnet",
        timeout_seconds=60.0,
        command_runner=runner,
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    score = client.score_article(_article())

    assert score.model_name == "claude-sonnet"
    assert score.market_call == "UP"
    assert score.sentiment == "bullish"
    assert score.impact_score == 0.34
    assert commands[0][0].lower().endswith("claude.cmd")
    assert commands[0][1] == "-p"
    assert "--json-schema" in commands[0]


def test_codex_reviewer_reads_output_file() -> None:
    commands: list[list[str]] = []

    def runner(
        command: list[str],
        workspace_dir: Path,
        timeout_seconds: float,
        input_text: str | None,
    ) -> CompletedProcess[str]:
        commands.append(command)
        assert input_text is not None
        assert "Article payload JSON:" in input_text
        output_index = command.index("-o") + 1
        output_path = Path(command[output_index])
        output_path.write_text(
            (
                '{"market_call":"DOWN","sentiment":"bearish","relevance":0.77,'
                '"impact_horizon_minutes":30,"impact_score":-0.28,'
                '"confidence":0.68,"reason":"Headline adds near-term risk."}'
            ),
            encoding="utf-8",
        )
        return CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    client = CLINewsReviewerClient(
        provider="codex",
        model_name="gpt-5",
        timeout_seconds=60.0,
        command_runner=runner,
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    score = client.score_article(_article())

    assert score.model_name == "gpt-5"
    assert score.market_call == "DOWN"
    assert score.sentiment == "bearish"
    assert score.impact_score == -0.28
    assert commands[0][0].lower().endswith("codex.cmd")
    assert commands[0][1] == "exec"
    assert commands[0][2:6] == ["-c", 'model_reasoning_effort="high"', "-c", "mcp_servers={}"]
    assert commands[0][-1] == "-"
    assert "--output-schema" in commands[0]


def test_cli_reviewer_can_raise_without_fallback() -> None:
    def runner(
        command: list[str],
        workspace_dir: Path,
        timeout_seconds: float,
        input_text: str | None,
    ) -> CompletedProcess[str]:
        return CompletedProcess(args=command, returncode=1, stdout="", stderr="boom")

    client = CLINewsReviewerClient(
        provider="claude",
        timeout_seconds=60.0,
        command_runner=runner,
        now_provider=lambda: datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
    )

    with pytest.raises(CLIReviewerServiceError):
        client.score_article(_article(), allow_fallback=False)


def test_resolve_executable_prefers_windows_cmd_shim(monkeypatch) -> None:
    monkeypatch.setattr(news_reviewers.os, "name", "nt")

    def _which(command: str) -> str | None:
        mapping = {
            "codex.cmd": r"C:\Users\elasm\AppData\Roaming\npm\codex.cmd",
            "codex": r"C:\Users\elasm\AppData\Roaming\npm\codex",
        }
        return mapping.get(command)

    monkeypatch.setattr(news_reviewers.shutil, "which", _which)

    assert (
        CLINewsReviewerClient._resolve_executable("codex")
        == r"C:\Users\elasm\AppData\Roaming\npm\codex.cmd"
    )
