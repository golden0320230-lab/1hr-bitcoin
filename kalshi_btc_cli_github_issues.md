# Kalshi BTC 1-Hour Predictor CLI — GitHub Issues Master File

This file contains the full issue set for building the project. Each issue is written so it can be copied into GitHub as-is.

---

# Epic

## Epic: Build a local CLI-only BTC predictor for live Kalshi 1-hour markets

### Objective
Build a local-first, research-only CLI application that:
- discovers the live Kalshi BTC 1-hour market
- extracts the threshold and expiry
- fetches live BTC spot and recent historical candles
- ingests recent BTC news
- sends relevant news to hosted KimiClaw for structured sentiment scoring
- combines market prior, price action, and news pressure into an ABOVE/BELOW prediction
- never places trades
- never gives financial advice

### Global completion criteria
- CLI works locally end to end
- prediction uses live market data, BTC data, and news
- graceful degradation exists for failed dependencies
- tests pass
- lint/type checks pass
- README is complete

---

# Issue 1

## Title
Bootstrap repository, tooling, and CLI entrypoint

## Branch
`feat/bootstrap-cli-foundation`

## Goal
Create the initial project structure, dependency management, code quality tooling, and a working CLI entrypoint.

## Why this matters
This establishes the project foundation so all later issues can be built on a typed, testable, maintainable base.

## Scope
- initialize Python project
- configure `uv`
- add Typer CLI entrypoint
- add logging/config modules
- add linting/type-checking/test tooling
- add pre-commit hooks
- add basic package structure

## Required files
- `pyproject.toml`
- `README.md`
- `.gitignore`
- `.env.example`
- `.pre-commit-config.yaml`
- `app/__init__.py`
- `app/cli.py`
- `app/config.py`
- `app/logging.py`
- `app/constants.py`
- `tests/test_cli.py`

## Implementation tasks
1. Create the repository structure from the build spec.
2. Configure dependencies for:
   - typer
   - httpx
   - pydantic
   - duckdb
   - pandas
   - scikit-learn
   - feedparser
   - beautifulsoup4
   - rich
   - tenacity
   - orjson
   - pytest
   - ruff
   - mypy
3. Implement a Typer app with a root command.
4. Add placeholder commands:
   - `market`
   - `predict`
   - `news`
   - `train`
   - `backtest`
   - `explain`
5. Add typed config loading from environment variables.
6. Add a reusable logger setup.
7. Add a simple smoke test for CLI help.
8. Add pre-commit config for ruff and basic checks.

## Acceptance criteria
- `btc-kalshi --help` works
- `btc-kalshi market --help` works
- `pytest` runs successfully
- `ruff check .` passes
- `mypy app` passes
- project structure matches the spec

## Dependencies
None

---

# Issue 2

## Title
Implement typed schemas and shared domain models

## Branch
`feat/schemas-and-domain-models`

## Goal
Create all shared pydantic/domain models required by the application.

## Why this matters
The rest of the system depends on consistent structured data for markets, candles, articles, KimiClaw scores, features, and predictions.

## Scope
- market schema
- market snapshot schema
- candle schema
- article schema
- article sentiment schema
- feature vector schema
- prediction result schema
- backtest result schema

## Required files
- `app/schemas.py`
- update tests as needed

## Implementation tasks
1. Define strongly typed schemas for:
   - Kalshi market metadata
   - Kalshi market pricing snapshot
   - BTC candle rows
   - news articles
   - KimiClaw response JSON
   - prediction run output
   - backtest metrics
2. Add validators where needed for timestamps, ranges, and required fields.
3. Ensure all timestamps normalize to UTC-aware datetime objects internally.
4. Add unit tests for schema validation behavior.

## Acceptance criteria
- schemas exist for all major entities
- invalid payloads raise validation errors
- tests cover valid and invalid cases

## Dependencies
- Issue 1

---

# Issue 3

## Title
Implement local storage layer with DuckDB

## Branch
`feat/duckdb-storage-layer`

## Goal
Create the local storage layer and database initialization logic.

## Why this matters
The app needs persistent local storage for candles, markets, news, scores, model metadata, and prediction history.

## Scope
- initialize DuckDB database
- create tables
- create insert/query helpers
- enforce idempotency where practical

## Required files
- `app/services/storage.py`
- tests for storage layer

## Required tables
- `markets`
- `market_snapshots`
- `btc_candles`
- `news_articles`
- `news_scores`
- `prediction_runs`
- `model_metadata`
- `backtest_results`

## Implementation tasks
1. Create database initialization logic.
2. Create SQL schema for all required tables.
3. Add helper methods for inserts and lookups.
4. Add unique constraints or dedup logic for:
   - candle timestamp + timeframe + source
   - news URL
   - market ticker
5. Store raw JSON payloads where helpful.
6. Add tests for DB initialization and dedup behavior.

## Acceptance criteria
- database initializes automatically
- tables are created on first run
- duplicate URLs are not inserted twice
- duplicate candle rows are not inserted twice
- storage tests pass

## Dependencies
- Issue 1
- Issue 2

---

# Issue 4

## Title
Implement Kalshi service for live BTC hourly market discovery

## Branch
`feat/kalshi-live-market-discovery`

## Goal
Discover the active live Kalshi BTC 1-hour market, normalize the threshold/expiry, and fetch live market pricing.

## Why this matters
The tool must use the latest live Kalshi BTC hourly market, not a hardcoded ticker.

## Scope
- query Kalshi public market data
- find active BTC hourly market
- parse threshold from title or structured payload
- normalize question into ABOVE/BELOW semantics
- fetch latest prices

## Required files
- `app/services/kalshi.py`
- `tests/test_kalshi.py`

## Implementation tasks
1. Implement Kalshi HTTP client using public endpoints.
2. Add functions to retrieve relevant events/markets.
3. Identify the active BTC hourly market.
4. Parse and normalize:
   - ticker
   - title
   - strike / threshold
   - expiry time
   - yes/no prices
5. Add logic for when no BTC hourly market is currently live.
6. Store discovered market and pricing snapshots in DuckDB.
7. Add mocked response fixtures and unit/integration tests.

## Acceptance criteria
- `btc-kalshi market` returns a live BTC hourly market when available
- threshold is parsed correctly
- expiry is normalized correctly
- yes/no prices are available in the returned model
- graceful message is shown when no market is live

## Dependencies
- Issue 1
- Issue 2
- Issue 3

---

# Issue 5

## Title
Implement BTC spot and candle ingestion service

## Branch
`feat/btc-spot-and-candles`

## Goal
Fetch BTC spot and recent historical candles from public sources and cache them locally.

## Why this matters
The prediction engine depends on current BTC price and recent price-action history.

## Scope
- Coinbase public spot fetch
- Coinbase candle fetch
- support 1m, 5m, 15m, 1h windows
- local storage and dedup
- retry/timeout handling

## Required files
- `app/services/coinbase.py`
- `tests/test_coinbase.py`

## Implementation tasks
1. Implement a Coinbase client for:
   - current BTC-USD spot
   - recent candles
2. Normalize candle payloads into the project candle schema.
3. Support configurable lookback windows.
4. Save candles into DuckDB.
5. Add a fallback mode or clear warning path if fetch fails.
6. Add tests for payload normalization and storage.

## Acceptance criteria
- current BTC price can be fetched
- recent 1m candle history can be fetched and stored
- duplicate candles are not reinserted
- timeouts and retries are implemented
- tests pass

## Dependencies
- Issue 1
- Issue 2
- Issue 3

---

# Issue 6

## Title
Implement news ingestion pipeline for BTC-related articles

## Branch
`feat/news-ingestion-pipeline`

## Goal
Fetch recent BTC-related news articles from free US-usable sources and normalize them into local storage.

## Why this matters
News is part of the signal mix and must be ingested before KimiClaw can evaluate directional impact.

## Scope
- Google News RSS ingestion
- GDELT ingestion
- normalize article fields
- de-duplicate by URL/content fingerprint
- configurable lookback

## Required files
- `app/services/news.py`
- `app/utils/text.py`
- `tests/test_news.py`

## Implementation tasks
1. Implement RSS query generation for BTC-related terms.
2. Implement article ingestion from Google News RSS.
3. Implement GDELT ingestion or structured query support.
4. Normalize article fields:
   - title
   - url
   - source
   - published_at
   - summary/snippet
5. Add de-duplication logic by URL and content fingerprint.
6. Filter out stale articles outside lookback.
7. Store normalized articles in DuckDB.
8. Add tests for dedup and freshness filtering.

## Acceptance criteria
- `btc-kalshi news --limit 10` returns recent BTC-related articles
- duplicate articles are filtered out
- stale articles are excluded correctly
- tests pass

## Dependencies
- Issue 1
- Issue 2
- Issue 3

---

# Issue 7

## Title
Implement KimiClaw client and structured article sentiment scoring

## Branch
`feat/kimiclaw-news-scoring`

## Goal
Send relevant articles to KimiClaw and parse strict JSON sentiment/impact results.

## Why this matters
The user explicitly wants KimiClaw to judge whether news is good or bad for short-horizon BTC price impact.

## Scope
- configurable KimiClaw endpoint client
- prompt template
- strict JSON validation
- fallback to neutral if scoring fails

## Required files
- `app/services/kimiclaw.py`
- `app/prompts/news_impact_prompt.txt`
- `tests/test_kimiclaw.py`

## Implementation tasks
1. Implement a KimiClaw HTTP client abstraction.
2. Support config values for:
   - base URL
   - API key
   - model name
3. Create a prompt template that instructs KimiClaw to return JSON only.
4. Define the required output structure:
   - sentiment
   - relevance
   - impact_horizon_minutes
   - impact_score
   - confidence
   - reason
5. Validate the JSON response with pydantic.
6. Add retry logic and timeout handling.
7. Add graceful fallback when:
   - endpoint errors
   - invalid JSON
   - empty output
8. Store scores in DuckDB.

## Acceptance criteria
- valid KimiClaw responses are parsed into typed models
- invalid model output is rejected and logged
- the app falls back to neutral news if KimiClaw is unavailable
- tests cover good and bad response cases

## Dependencies
- Issue 1
- Issue 2
- Issue 3
- Issue 6

---

# Issue 8

## Title
Implement feature engineering for market, price, and news signals

## Branch
`feat/feature-engineering`

## Goal
Transform raw market data, BTC data, and news scores into a feature vector suitable for prediction.

## Why this matters
This issue creates the signal layer the predictor depends on.

## Scope
- Kalshi prior features
- price-action features
- news aggregate features
- feature schema versioning

## Required files
- `app/services/features.py`
- `tests/test_features.py`

## Implementation tasks
1. Implement Kalshi prior features:
   - yes price
   - no price
   - implied probability
   - spread
   - optional orderbook imbalance if available
2. Implement price-action features:
   - spot price
   - distance to strike
   - distance to strike percent
   - 5m/15m/30m/60m returns
   - realized volatility
   - moving average deviation
   - RSI or simple oscillator
   - momentum slope
   - minutes to expiry
3. Implement news aggregate features:
   - weighted impact score
   - weighted bullish count
   - weighted bearish count
   - count of high-confidence relevant articles
   - breaking-news flag
4. Build a typed feature vector object.
5. Add tests for feature generation and edge cases.

## Acceptance criteria
- a complete feature vector can be generated from valid inputs
- missing news produces neutral news-derived features
- tests cover normal and degraded cases

## Dependencies
- Issue 2
- Issue 4
- Issue 5
- Issue 6
- Issue 7

---

# Issue 9

## Title
Implement baseline prediction engine and blending logic

## Branch
`feat/baseline-prediction-engine`

## Goal
Create the first working prediction engine that combines Kalshi prior, price features, and news pressure into an ABOVE/BELOW call.

## Why this matters
This is the first issue that delivers the core product outcome.

## Scope
- baseline deterministic blend
- probability output
- confidence bucket
- explanation generation

## Required files
- `app/services/predictor.py`
- `tests/test_predictor.py`

## Implementation tasks
1. Implement a first-pass blend strategy that combines:
   - market prior
   - price signal probability
   - news score
2. Make blend weights configurable.
3. Output:
   - label: ABOVE/BELOW
   - probability
   - confidence bucket
   - top driver explanations
4. Ensure degraded paths still return results when news is missing.
5. Add tests for blend behavior and output schema.

## Acceptance criteria
- predictor produces ABOVE/BELOW with probability
- confidence bucket is included
- explanation contains top drivers
- predictor works without news scoring
- tests pass

## Dependencies
- Issue 2
- Issue 8

---

# Issue 10

## Title
Wire the `market`, `news`, and `predict` CLI commands end to end

## Branch
`feat/cli-end-to-end-commands`

## Goal
Connect all service layers into usable CLI commands.

## Why this matters
This makes the application actually usable from the terminal.

## Scope
- `market` command
- `news` command
- `predict` command
- JSON output mode
- verbose mode
- save-run mode

## Required files
- update `app/cli.py`
- add or update CLI tests

## Implementation tasks
1. Implement `market` command using Kalshi service.
2. Implement `news` command using news + optional KimiClaw scoring.
3. Implement `predict` command with the full pipeline:
   - discover market
   - fetch BTC spot/candles
   - ingest news
   - score articles
   - generate features
   - run predictor
4. Add `--json` output mode.
5. Add `--no-news`, `--verbose`, and `--save-run` flags.
6. Add Rich-formatted terminal output.
7. Ensure disclaimer is always shown in human-readable mode.

## Acceptance criteria
- `btc-kalshi market` works
- `btc-kalshi news --limit 10` works
- `btc-kalshi predict` works end to end
- `btc-kalshi predict --json` returns valid structured JSON
- CLI tests pass

## Dependencies
- Issue 4
- Issue 5
- Issue 6
- Issue 7
- Issue 8
- Issue 9

---

# Issue 11

## Title
Persist prediction runs and build explain command

## Branch
`feat/prediction-history-and-explain`

## Goal
Save prediction run outputs locally and allow the user to inspect the most recent run in detail.

## Why this matters
This improves debuggability, transparency, and future evaluation.

## Scope
- save run metadata
- save feature vector snapshot
- save prediction output
- implement `explain --last`

## Required files
- `app/services/explain.py`
- update `app/cli.py`
- related tests

## Implementation tasks
1. Persist each saved prediction run to DuckDB.
2. Store:
   - market context
   - feature vector
   - prediction result
   - warning/degraded flags
3. Implement `explain --last`.
4. Format output to show:
   - prediction
   - probability
   - main drivers
   - feature values
   - warnings
5. Add tests for persistence and explain retrieval.

## Acceptance criteria
- prediction runs can be stored and retrieved
- `btc-kalshi explain --last` works
- explanations include major feature contributors
- tests pass

## Dependencies
- Issue 3
- Issue 9
- Issue 10

---

# Issue 12

## Title
Build training dataset generator aligned to market question semantics

## Branch
`feat/training-dataset-builder`

## Goal
Create a training dataset generator that builds historical rows aligned to the actual market question: finish above or below a threshold at expiry.

## Why this matters
Training on generic next-candle direction would not match the Kalshi market question and would produce the wrong model target.

## Scope
- define label construction logic
- build historical examples
- align inputs to threshold and expiry
- save dataset locally

## Required files
- `app/services/training.py`
- tests for dataset generation

## Implementation tasks
1. Implement logic to generate historical prediction rows.
2. Ensure each row includes:
   - prediction timestamp
   - threshold
   - expiry time
   - features available at prediction time
   - label indicating above/below at expiry
3. Save datasets in a reusable local format.
4. Version the feature schema.
5. Add tests verifying label correctness.

## Acceptance criteria
- historical dataset rows align to above/below-at-expiry semantics
- label generation is tested
- training dataset can be reused by model training

## Dependencies
- Issue 5
- Issue 8

---

# Issue 13

## Title
Implement local model training and model artifact management

## Branch
`feat/model-training-and-artifacts`

## Goal
Train baseline local models and persist a selected model artifact for inference.

## Why this matters
This upgrades the predictor from hand-tuned logic toward a repeatable ML-backed baseline.

## Scope
- logistic regression baseline
- gradient boosting baseline
- model comparison
- artifact persistence
- metadata persistence

## Required files
- update `app/services/training.py`
- maybe add `models/` artifact handling
- tests for training workflow

## Implementation tasks
1. Train at least two baseline models:
   - logistic regression
   - gradient boosting classifier
2. Compare validation performance.
3. Select the default model based on metrics and stability.
4. Save the model artifact locally.
5. Save metadata including:
   - feature schema version
   - training window
   - model type
   - metrics
6. Add CLI support for `train`.
7. Add tests for training workflow and missing-model behavior.

## Acceptance criteria
- `btc-kalshi train --days 90` creates a local model artifact
- model metadata is saved
- inference can load the artifact successfully
- tests pass

## Dependencies
- Issue 12

---

# Issue 14

## Title
Implement historical backtesting with naive baseline comparisons

## Branch
`feat/backtesting-and-baselines`

## Goal
Evaluate the predictor on historical windows and compare it against simple baselines.

## Why this matters
The predictor should be judged against realistic baselines, not in isolation.

## Scope
- backtest runner
- naive baselines
- output metrics
- persist backtest summaries

## Required files
- `app/services/backtest.py`
- `tests/test_backtest.py`

## Required baselines
- Kalshi-follow baseline
- spot-vs-strike baseline
- momentum baseline
- no-news baseline

## Implementation tasks
1. Build a backtest runner over historical examples.
2. Implement required baseline strategies.
3. Compare full model versus baselines using:
   - accuracy
   - log loss
   - Brier score if practical
4. Save backtest outputs to DuckDB.
5. Implement CLI support for `backtest`.
6. Add tests for baseline behavior and backtest outputs.

## Acceptance criteria
- `btc-kalshi backtest --days 30` runs successfully
- backtest output includes model and baselines
- metrics are persisted
- tests pass

## Dependencies
- Issue 12
- Issue 13

---

# Issue 15

## Title
Harden reliability, retries, and graceful degradation paths

## Branch
`feat/reliability-and-fallbacks`

## Goal
Ensure the application behaves predictably under dependency failures and partial-data scenarios.

## Why this matters
This tool depends on live external data and should fail safely and informatively.

## Scope
- retry wrappers
- timeouts
- warning surfaces
- neutral fallbacks
- clean CLI exits

## Required files
- `app/utils/retries.py`
- updates across service modules
- tests for degraded behavior

## Implementation tasks
1. Wrap all external HTTP calls in shared retry logic.
2. Ensure every client uses timeouts.
3. Add user-facing warnings for:
   - no live market
   - BTC fetch failure
   - no news found
   - KimiClaw unavailable
   - model artifact missing
4. Ensure the predictor still returns results when non-critical services fail.
5. Add tests for degraded cases.

## Acceptance criteria
- common external failures do not crash the CLI unnecessarily
- warnings are visible and understandable
- neutral fallback behavior is used where appropriate
- tests cover degraded scenarios

## Dependencies
- Issue 4
- Issue 5
- Issue 6
- Issue 7
- Issue 9
- Issue 10

---

# Issue 16

## Title
Implement security and safety controls for research-only operation

## Branch
`feat/security-and-safety-controls`

## Goal
Add controls that keep the tool safely scoped to research-only usage.

## Why this matters
The tool must not drift into execution, unsafe content handling, or misleading advice.

## Scope
- disclaimer enforcement
- article sanitization
- no-trading boundaries
- secret hygiene

## Required files
- updates across services and CLI
- tests for safety-related behavior where practical

## Implementation tasks
1. Ensure every prediction output includes a research-only disclaimer.
2. Sanitize article content before sending to KimiClaw.
3. Truncate article body length to a safe configured max.
4. Strip HTML/script content.
5. Confirm no trade execution modules or flows exist.
6. Ensure no “buy” or “sell” recommendation language appears in outputs.
7. Keep secrets loaded only from env/config.

## Acceptance criteria
- article text is sanitized before model submission
- output contains disclaimer consistently
- project contains no trading/execution code
- secrets are not hardcoded

## Dependencies
- Issue 7
- Issue 10

---

# Issue 17

## Title
Complete README and first-run documentation

## Branch
`docs/readme-and-setup`

## Goal
Write complete user and developer documentation for setup, usage, limitations, and architecture.

## Why this matters
The project should be installable and understandable by another engineer or hiring manager.

## Scope
- project overview
- architecture summary
- install steps
- config docs
- command usage
- limitations and disclaimers

## Required files
- `README.md`
- update `.env.example` if needed

## Implementation tasks
1. Document project purpose and constraints.
2. Document setup using `uv`.
3. Document all required environment variables.
4. Add command examples for:
   - market
   - news
   - predict
   - train
   - backtest
   - explain
5. Document data sources.
6. Document research-only limitation and lack of trade execution.
7. Document degraded behavior when KimiClaw is unavailable.

## Acceptance criteria
- README is complete and accurate
- a new developer can set up the project from the docs
- command examples are included
- limitations are clearly stated

## Dependencies
- Issue 10
- Issue 13
- Issue 14

---

# Issue 18

## Title
Polish CLI output formatting and JSON export consistency

## Branch
`feat/output-polish-and-json-consistency`

## Goal
Make the CLI pleasant to use while keeping machine-readable output stable.

## Why this matters
This improves usability, demos, and downstream automation.

## Scope
- human-readable Rich formatting
- stable JSON output schema
- compact summaries
- warning formatting

## Required files
- update `app/cli.py`
- update schemas/tests as needed

## Implementation tasks
1. Improve terminal formatting for market, news, predict, and explain commands.
2. Ensure `--json` produces a stable schema.
3. Ensure warnings and degraded states appear in both human and JSON modes.
4. Add tests for JSON output shape.

## Acceptance criteria
- human output is readable and concise
- JSON output is stable and valid
- warnings are included in both output modes

## Dependencies
- Issue 10
- Issue 11

---

# Issue 19

## Title
Add comprehensive test fixtures and mocked integration coverage

## Branch
`test/fixtures-and-integration-coverage`

## Goal
Expand test coverage with realistic mocked external payloads and end-to-end command validation.

## Why this matters
This project depends on multiple external providers, so fixture-driven testing is necessary for stable development.

## Scope
- mocked Kalshi payloads
- mocked Coinbase payloads
- mocked RSS/GDELT payloads
- mocked KimiClaw responses
- CLI integration tests

## Required files
- test fixtures under `tests/fixtures/`
- updates to all test files

## Implementation tasks
1. Add representative fixture payloads for all major providers.
2. Add end-to-end tests for:
   - market command
   - news command
   - predict command
   - predict with missing news
   - predict with invalid KimiClaw response
3. Ensure tests do not rely on live network access.

## Acceptance criteria
- integration coverage exists for the main flows
- test suite can run offline using fixtures/mocks
- failure cases are covered

## Dependencies
- Issue 4 through Issue 18

---

# Issue 20

## Title
Final quality gate and release-ready cleanup

## Branch
`chore/final-quality-gate`

## Goal
Perform final cleanup and ensure the project meets the full definition of done.

## Why this matters
This consolidates the project into a portfolio-ready and maintainable state.

## Scope
- final lint/type/test pass
- dead code cleanup
- config cleanup
- versioning or release notes if desired

## Implementation tasks
1. Run all checks and fix remaining issues.
2. Remove dead code and unused dependencies.
3. Confirm the CLI works from a clean local install.
4. Confirm the README matches actual behavior.
5. Validate the final definition of done checklist.

## Acceptance criteria
- `pytest` passes
- `ruff check .` passes
- `mypy app` passes
- clean install works
- project satisfies the build spec definition of done

## Dependencies
- Issue 1 through Issue 19

---

# Recommended Build Order

1. Issue 1 — Bootstrap repository, tooling, and CLI entrypoint
2. Issue 2 — Typed schemas and shared domain models
3. Issue 3 — DuckDB storage layer
4. Issue 4 — Kalshi live market discovery
5. Issue 5 — BTC spot and candle ingestion
6. Issue 6 — News ingestion pipeline
7. Issue 7 — KimiClaw client and scoring
8. Issue 8 — Feature engineering
9. Issue 9 — Baseline prediction engine
10. Issue 10 — CLI commands end to end
11. Issue 11 — Prediction history and explain command
12. Issue 12 — Training dataset builder
13. Issue 13 — Model training and artifacts
14. Issue 14 — Backtesting and baselines
15. Issue 15 — Reliability and graceful degradation
16. Issue 16 — Security and safety controls
17. Issue 17 — README and setup docs
18. Issue 18 — Output polish and JSON consistency
19. Issue 19 — Fixtures and integration coverage
20. Issue 20 — Final quality gate

---

# Done Checklist

The full project is complete when:
- local install works
- `btc-kalshi market` shows the live BTC hourly Kalshi market
- `btc-kalshi predict` returns a research-only ABOVE/BELOW result
- prediction uses market data, BTC data, and news
- KimiClaw scoring is integrated with graceful fallback
- training and backtesting work
- prediction history is saved locally
- tests, lint, and type checks pass
- README is complete
- no trading or execution code exists

