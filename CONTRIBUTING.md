# Contributing to AgentBench

Thanks for helping. Bug reports, new behavior categories, suites, and docs fixes are all welcome.

## Dev setup

```bash
git clone https://github.com/himanshu748/agentbench && cd agentbench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn backend.main:app --reload --port 8000   # mock mode without a key
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to hit real models.

## Before you open a PR

```bash
ruff check .
pytest
```

Tests run in mock mode and never call a paid API. If you change model-provider code,
add a test using `httpx.MockTransport` (see `tests/test_core.py`).

## Where things live

| Path | What |
| --- | --- |
| `backend/llm_client.py` | The only module that talks to a model provider (OpenRouter) |
| `backend/judge.py` | Behavior categories, rubrics, judge prompt, prompt-fix generation |
| `backend/runner.py` | Concurrent suite x model execution and the ship recommendation |
| `backend/main.py` | FastAPI routes |
| `backend/storage.py` | SQLite report persistence |
| `cli/agentbench.py` | Stdlib-only CLI used by the GitHub Action |
| `frontend/` | Static landing page, console, and report page |
| `suites/` | Built-in YAML suites |

### Adding a behavior category

Add a rubric to `RUBRICS` in `backend/judge.py`. Keep it one or two sentences that start with
the situation and end with an explicit `PASS only if ...` condition. Add a test case using it to
an existing suite.

## Style

- Small, focused PRs with a clear description of behavior changes.
- Match the surrounding code; comments explain *why*, not *what*.
- UI copy follows [docs/brand.md](docs/brand.md): terse, numbers over adjectives, no em dashes.

By contributing you agree that your contributions are licensed under the MIT License.
