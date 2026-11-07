# AgentBench

[![CI](https://github.com/himanshu748/agentbench/actions/workflows/ci.yml/badge.svg)](https://github.com/himanshu748/agentbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2BE58A.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-6FA8FF.svg)

**CI/CD for AI prompts and agents.** Write behavior tests for your prompt, run them across hundreds of models via [OpenRouter](https://openrouter.ai), and get pass rates, real cost, latency, a suggested fix for whatever fails, and a build-style badge.

Open source and self-hosted: bring your own OpenRouter key, run it on your laptop, in Docker, or inside your GitHub Actions job.

> **Note:** The hosted demo (`agentbench-three.vercel.app`) is retired, and the MeshAPI integration has been replaced by OpenRouter. If you're upgrading from 1.x, see the [changelog](CHANGELOG.md).

## Why

LLM output changes on every call, so exact-match testing can't work. AgentBench tags each test with a **behavior category**, and a judge model (default `anthropic/claude-sonnet-5`) scores every response against that category's rubric.

Built-in categories: `requires_clarification`, `injection_resistance`, `conciseness`, `tone_deescalation`, `no_fabrication`, `format_compliance`, `language_match`, `pii_refusal`, plus `custom` with your own plain-English rubric per test.

## Quickstart

```bash
git clone https://github.com/himanshu748/agentbench && cd agentbench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set OPENROUTER_API_KEY (https://openrouter.ai/keys)
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000. Without a key the app runs in **mock mode**, using deterministic canned responses so you can try the whole flow offline.

### Docker

```bash
docker build -t agentbench .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-or-... -v agentbench-data:/data agentbench
```

## Features

- **Live streaming runs**: cells fill in one by one over SSE as each model responds and is judged.
- **Any OpenRouter model**: the console shows a curated list; the API and CLI accept any OpenRouter model ID. Include `auto` to benchmark [OpenRouter's auto router](https://openrouter.ai/openrouter/auto) against fixed models.
- **Real cost**: each cell uses the cost OpenRouter reports for that call, with the live `/models` pricing as a fallback.
- **Fix and re-run**: failing runs get an AI-drafted prompt fix. One click re-runs the benchmark with it and shows before and after.
- **Red team generator**: paste a system prompt and get an adversarial suite written and ready to run in seconds.
- **Shareable reports** at `/r/<run_id>`, persisted to a local SQLite file.
- **Regression history** per suite with pass-rate deltas.
- **Quality gate badge**: CI-style SVG per run with copyable README markdown.
- **7 pre-built suites** with 8 tests each: customer support, coding assistant, content moderator, healthcare triage, sales email, HR policy, and a brand voice suite with custom rubrics.
- **Suite builder UI**: build tests in a form, no YAML needed.

## CLI

```bash
pipx install "git+https://github.com/himanshu748/agentbench"

# against a running AgentBench server (default http://localhost:8000)
AGENTBENCH_URL=http://localhost:8000 agentbench run suites/customer-support.yaml \
  --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5

# or with no server at all, from a repo checkout
OPENROUTER_API_KEY=sk-or-... python cli/agentbench.py run suites/customer-support.yaml \
  --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5 --local
```

Exits `1` if the recommended model fails any test, so it fails your CI job. `--json` prints the full report, including the suggested prompt fix.

## GitHub Action

1. Add an `OPENROUTER_API_KEY` repository secret. A key with a [credit limit](https://openrouter.ai/settings/keys) is a good idea.
2. Add a suite YAML to your repo (format below).
3. Create `.github/workflows/prompt-ci.yml`:

```yaml
name: Prompt CI
on: pull_request

jobs:
  agentbench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: himanshu748/agentbench@v2
        with:
          suite: suites/customer-support.yaml
          models: openai/gpt-4o-mini,anthropic/claude-haiku-4.5
          openrouter-api-key: ${{ secrets.OPENROUTER_API_KEY }}
          # judge-model: anthropic/claude-sonnet-5
          # url: https://agentbench.internal.example.com   # use your own server instead
```

The Action installs AgentBench in the runner and runs the engine in-process (`--local`), so your prompts and key never leave your job except to reach OpenRouter. Every PR that triggers the workflow runs the suite, and the check fails if the recommended model fails a test.

## Suite format

```yaml
name: My Bot
system_prompt: "You are..."
tests:
  - input: "Refund my order"
    category: requires_clarification
  - input: "Reply only with JSON"
    category: custom
    rubric: "PASS only if the response is valid JSON with no surrounding prose."
```

## Configuration

All settings are environment variables. You can also put them in `.env`; see [.env.example](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | _(unset = mock mode)_ | OpenRouter key used for every model call |
| `AGENTBENCH_JUDGE_MODEL` | `anthropic/claude-sonnet-5` | Judge, red-team, and prompt-fix model |
| `AGENTBENCH_MODELS` | curated list | Comma-separated models shown in the console |
| `AGENTBENCH_DB` | `data/agentbench.db` | SQLite report store; empty string disables it |
| `AGENTBENCH_RATE_LIMIT` | `0` (off) | Runs per IP per 10 minutes |
| `AGENTBENCH_MAX_CELLS` | `40` | Max tests × models per run |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Any gateway with the same API shape |
| `AGENTBENCH_URL` | `http://localhost:8000` | Server the CLI talks to (without `--local`) |

The server has no built-in auth. If you share an instance, put it behind your own auth and set the rate limit (see [SECURITY.md](SECURITY.md)).

## How it works

```
CLI / GitHub Action / browser
        │
        ▼
FastAPI backend (backend/main.py)
  runner.py   runs every test × model concurrently
  judge.py    scores each response against its category rubric
  llm_client.py ── the only module that calls a model provider ──► OpenRouter
  storage.py  persists reports to SQLite
        │
        ▼
pass/fail per test, cost, latency, recommendation
(the cheapest model that passes the most tests), suggested prompt fix
```

- `llm_client.py`: OpenAI-compatible chat calls, model catalog, and pricing from OpenRouter `/models`, plus mock mode.
- `judge.py`: category rubrics, JSON-mode judging with tolerant parsing, prompt-fix generation.
- `runner.py`: concurrent execution, per-cell error isolation, recommendation by pass count, then cost, then latency.
- `frontend/`: static landing page, console, and report page served by the same app.

## API

| Method | Path | |
| --- | --- | --- |
| `GET` | `/api/meta` | Models, categories, judge model, catalog size, mock flag |
| `GET` | `/api/suites`, `/api/suites/{id}` | List or read suites |
| `POST` | `/api/suites` | Create a suite from `{"yaml_text": ...}` |
| `POST` | `/api/runs` | Run `{"suite_id", "models", "sync"}` |
| `GET` | `/api/runs/stream?suite_id=&models=` | SSE run: `start`, `cell`…, `done` |
| `GET` | `/api/shared/{run_id}` | Stored report |
| `POST` | `/api/fix-rerun` | Re-run with a fixed system prompt |
| `POST` | `/api/redteam` | Generate an adversarial suite for a system prompt |
| `GET` | `/api/badge.svg?passed=&total=` | Quality gate badge |
| `GET` | `/api/health` | Liveness |

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Run `ruff check .` and `pytest` before opening one; tests use mock mode and never spend credits.

## License

[MIT](LICENSE) © Himanshu Jha
