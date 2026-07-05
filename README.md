# AgentBench

**CI/CD for AI prompts and agents.** Behavior tests for your prompt, run across the models on MeshAPI, with pass rates, real cost, latency, a fix for what fails, and a build-style badge.

Live: **https://agentbench-three.vercel.app** &middot; Built for the MeshAPI Hackathon (July 2026)

Every LLM call (test execution, judge scoring, red team generation, prompt fixes) routes through MeshAPI via one integration point: [backend/mesh_client.py](backend/mesh_client.py).

## Why

LLM output is non-deterministic, so exact-match testing is dead. AgentBench tags each test with a **behavior category** and a judge model (claude-sonnet-5, via MeshAPI) scores every response against that category's rubric.

Built-in categories: `requires_clarification`, `injection_resistance`, `conciseness`, `tone_deescalation`, `no_fabrication`, `format_compliance`, `language_match`, `pii_refusal`, plus `custom` with your own plain-English rubric per test.

## Features

- **Live streaming runs**: cells fill in one by one over SSE as each model responds and is judged.
- **Fix and re-run**: failing runs get an AI-drafted prompt fix, one click re-benchmarks it, before/after shown.
- **Red team generator**: paste a system prompt, get an adversarial suite written and runnable in seconds.
- **Benchmark MeshAPI `auto`**: test smart routing against fixed models on your own workload.
- **Shareable reports**: every run persists (Supabase) at `/r/<run_id>`.
- **Regression history** per suite with pass-rate deltas.
- **Quality gate badge**: CI-style SVG per run with copyable README markdown.
- **8 pre-built suites**: customer support, coding assistant, content moderator, healthcare triage, sales email, HR policy, brand voice (custom rubrics).

## CLI and GitHub Action

```bash
# terminal
python3 cli/agentbench.py run suites/customer-support.yaml \
  --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
# exit code 1 if the recommended model fails any test -> fails your CI job
```

```yaml
# .github/workflows/prompt-ci.yml
- uses: himanshujha/agentbench@v1
  with:
    suite: suites/customer-support.yaml
    models: openai/gpt-4o-mini,anthropic/claude-haiku-4.5
```

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

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "MESH_API_KEY=rsk_..." > .env         # from meshapi.ai
uvicorn backend.main:app --port 8000       # open http://localhost:8000
```

No key? Mock mode keeps the whole flow demoable offline.

## Architecture

- FastAPI, one Vercel Python serverless function (`api/index.py`), static frontend served by the same app.
- `mesh_client.py`: the only file that talks to MeshAPI. Model catalog and per-token pricing pulled live from `/v1/models` (531 models, 396 chat).
- `judge.py`: category rubrics, JSON-mode judging, prompt-fix generation.
- `runner.py`: concurrent execution, per-cell timeout isolation, recommendation (pass rate, then cost, then latency).
- `storage.py`: shared reports in Supabase (RLS-guarded anon key).
- Abuse guards: per-IP rate limiting, 40-cell run ceiling, spend-capped MeshAPI key.
