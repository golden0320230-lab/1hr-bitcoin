"""Local CLI-backed news review clients for Codex and Claude."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Literal, cast

import orjson

from app.schemas import (
    ArticleSentimentScore,
    NewsArticle,
    normalize_reviewer_market_call,
)
from app.services.storage import DuckDBStorage
from app.utils.text import sanitize_text, truncate_text

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "news_impact_prompt.txt"
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "news_impact_response_schema.json"
)
ReviewerProvider = Literal["codex", "claude"]
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_ERROR_JSON_PATTERN = re.compile(r"ERROR:\s*(\{.*?\})(?:\s+Warning:|\s*$)", re.DOTALL)


class CLIReviewerServiceError(RuntimeError):
    """Raised when a local reviewer CLI cannot return usable structured output."""


CommandRunner = Callable[[list[str], Path, float, str | None], CompletedProcess[str]]


def _default_command_runner(
    command: list[str],
    workspace_dir: Path,
    timeout_seconds: float,
    input_text: str | None = None,
) -> CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(workspace_dir),
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


class CLINewsReviewerClient:
    """Run a local CLI reviewer and normalize the result into article sentiment scores."""

    def __init__(
        self,
        *,
        provider: ReviewerProvider,
        model_name: str | None = None,
        storage: DuckDBStorage | None = None,
        timeout_seconds: float = 60.0,
        max_article_chars: int = 2_000,
        workspace_dir: str | Path | None = None,
        command_runner: CommandRunner | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.storage = storage
        self.timeout_seconds = timeout_seconds
        self.max_article_chars = max_article_chars
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
        self._command_runner = command_runner or _default_command_runner
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        self._schema_text = SCHEMA_PATH.read_text(encoding="utf-8")

    def close(self) -> None:
        return None

    def __enter__(self) -> CLINewsReviewerClient:
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
            impact_score = response_payload["impact_score"]
            score = ArticleSentimentScore(
                article_url=article.url,
                model_name=self._score_model_name(),
                scored_at=self._now_provider(),
                market_call=normalize_reviewer_market_call(
                    response_payload.get("market_call"),
                    impact_score=impact_score,
                ),
                sentiment=response_payload["sentiment"],
                relevance=response_payload["relevance"],
                impact_horizon_minutes=response_payload["impact_horizon_minutes"],
                impact_score=impact_score,
                confidence=response_payload["confidence"],
                reason=response_payload["reason"],
                raw_response={
                    "provider": self.provider,
                    "response": response_payload,
                },
            )
        except (CLIReviewerServiceError, KeyError, TypeError, ValueError) as exc:
            if not allow_fallback:
                raise
            score = self._neutral_fallback(article, reason=f"{self.provider} fallback: {exc}")

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

    def _request_score(self, article: NewsArticle) -> dict[str, Any]:
        prompt = self.build_prompt(article)
        if self.provider == "codex":
            output_text = self._run_codex(prompt)
        else:
            output_text = self._run_claude(prompt)
        return self._extract_json_payload(output_text)

    @staticmethod
    def _resolve_executable(command_name: str) -> str:
        if os.name == "nt":
            for candidate in (
                f"{command_name}.cmd",
                f"{command_name}.bat",
                f"{command_name}.exe",
                command_name,
            ):
                resolved = shutil.which(candidate)
                if resolved is not None:
                    return resolved
        resolved = shutil.which(command_name)
        if resolved is not None:
            return resolved
        return command_name

    def _run_codex(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        try:
            command = [
                self._resolve_executable("codex"),
                "exec",
                "-c",
                'model_reasoning_effort="high"',
                "-c",
                "mcp_servers={}",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--ephemeral",
                "--output-schema",
                str(SCHEMA_PATH),
                "-o",
                str(output_path),
            ]
            if self.model_name is not None:
                command.extend(["-m", self.model_name])
            command.append("-")

            completed = self._command_runner(
                command,
                self.workspace_dir,
                self.timeout_seconds,
                prompt,
            )
            if completed.returncode != 0:
                stderr = self._summarize_process_failure(completed.stderr or completed.stdout)
                raise CLIReviewerServiceError(f"Codex CLI failed: {stderr or 'unknown error'}")

            output_text = output_path.read_text(encoding="utf-8").strip()
            if not output_text:
                raise CLIReviewerServiceError("Codex CLI returned empty output.")
            return output_text
        finally:
            output_path.unlink(missing_ok=True)

    def _run_claude(self, prompt: str) -> str:
        command = [
            self._resolve_executable("claude"),
            "-p",
            "--no-session-persistence",
            "--tools",
            "",
            "--json-schema",
            self._schema_text,
        ]
        if self.model_name is not None:
            command.extend(["--model", self.model_name])
        command.append(prompt)

        completed = self._command_runner(command, self.workspace_dir, self.timeout_seconds, None)
        if completed.returncode != 0:
            stderr = self._summarize_process_failure(completed.stderr or completed.stdout)
            raise CLIReviewerServiceError(f"Claude CLI failed: {stderr or 'unknown error'}")

        output_text = completed.stdout.strip()
        if not output_text:
            raise CLIReviewerServiceError("Claude CLI returned empty output.")
        return output_text

    @staticmethod
    def _extract_json_payload(output_text: str) -> dict[str, Any]:
        stripped = output_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.removeprefix("json").strip()

        try:
            return cast(dict[str, Any], orjson.loads(stripped))
        except orjson.JSONDecodeError:
            match = _JSON_OBJECT_PATTERN.search(stripped)
            if match is None:
                raise CLIReviewerServiceError(
                    "Reviewer output was not valid JSON."
                ) from None
            try:
                return cast(dict[str, Any], orjson.loads(match.group(0)))
            except orjson.JSONDecodeError as exc:
                raise CLIReviewerServiceError("Reviewer output was not valid JSON.") from exc

    def _score_model_name(self) -> str:
        if self.model_name is not None:
            return self.model_name
        return f"{self.provider}-cli"

    @staticmethod
    def _summarize_process_failure(output_text: str) -> str:
        match = _ERROR_JSON_PATTERN.search(output_text)
        if match is not None:
            try:
                payload = cast(dict[str, Any], orjson.loads(match.group(1)))
            except orjson.JSONDecodeError:
                payload = {}
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message.strip():
                    return truncate_text(sanitize_text(message), max_chars=240)

        lines = [line.strip() for line in output_text.splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("warning:"):
                continue
            if lowered.startswith("error:"):
                return truncate_text(sanitize_text(line[6:]), max_chars=240)
            if "failed" in lowered or "unsupported" in lowered:
                return truncate_text(sanitize_text(line), max_chars=240)

        return truncate_text(sanitize_text(output_text), max_chars=240)

    def _neutral_fallback(self, article: NewsArticle, *, reason: str) -> ArticleSentimentScore:
        return ArticleSentimentScore(
            article_url=article.url,
            model_name=self._score_model_name(),
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
