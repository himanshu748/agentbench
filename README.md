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
- **7 pre-built suites**, 8 tests each covering every built-in category: customer support, coding assistant, content moderator, healthcare triage, sales email, HR policy, plus a brand voice suite showcasing custom rubrics.
- **Suite builder UI**: create tests in the console with a form (category dropdown, inline rubric editor for custom tests). No YAML needed.

## CLI and GitHub Action

```bash
pipx install "git+https://github.com/himanshu748/agentbench"   # or: pip install ...

agentbench run suites/customer-support.yaml \
  --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
# exit code 1 if the recommended model fails any test -> fails your CI job
# point at your own server with AGENTBENCH_URL=http://localhost:8000
```

```yaml
# .github/workflows/prompt-ci.yml
- uses: himanshu748/agentbench@v1
  with:
    suite: suites/customer-support.yaml
    models: openai/gpt-4o-mini,anthropic/claude-haiku-4.5
```

### Setup in your own repo

1. Add a suite YAML somewhere in your repo (e.g. `suites/customer-support.yaml`), using the format below.
2. Create `.github/workflows/prompt-ci.yml`:
   ```yaml
   name: Prompt CI
   on: pull_request

   jobs:
     agentbench:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: himanshu748/agentbench@v1
           with:
             suite: suites/customer-support.yaml
             models: openai/gpt-4o-mini,anthropic/claude-haiku-4.5
   ```
3. Push. No secrets needed — the Action calls the hosted `agentbench-three.vercel.app` backend by default, which already holds the MeshAPI key. Every PR that touches your trigger path now runs the suite and fails the check (blocking merge) if the recommended model fails a test.

   Only add a `url:` input (pointing at your own deployment) if you're self-hosting the backend — in that case, your deployment needs its own `MESH_API_KEY`.

## How the CI/CD gate actually works

There's no magic running inside GitHub's runner — it's a thin client hitting a server:

```
GitHub Actions runner (or your laptop)
        │
        │ runs: python cli/agentbench.py run suite.yaml --models ...
        ▼
cli/agentbench.py ── HTTP POST ──► AGENTBENCH_URL
                                     (defaults to the hosted
                                      agentbench-three.vercel.app,
                                      or your own `uvicorn` instance)
                                            │
                                    FastAPI backend does the real work:
                                    calls MeshAPI for each model, scores
                                    every response with the judge model
                                            │
                                    returns JSON: pass/fail per test,
                                    cost, latency, a recommendation
        ◄────────────────────────────────────┘
CLI prints a pass/fail table and exits 1 if the recommended
model failed any test -> that non-zero exit fails your CI job.
```

`action.yml` just checks out your repo and runs that same CLI script — so anywhere the CLI works, the Action works.

**Try it locally without touching GitHub:**

```bash
# against the hosted server (default) — no setup needed
python3 cli/agentbench.py run suites/customer-support.yaml \
  --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
# echo $? -> 0 if the recommended model passed everything, 1 if not

# against your own backend, e.g. to test a suite before pushing
uvicorn backend.main:app --port 8000 &
AGENTBENCH_URL=http://localhost:8000 python3 cli/agentbench.py run \
  suites/customer-support.yaml --models openai/gpt-4o-mini
```

Custom rubric tests and the red-team generator use this same backend, not a CLI-only path — `POST /api/suites` validates and stores a custom-rubric suite, `POST /api/redteam` asks the judge model to write adversarial tests for your system prompt. Both work identically whether you're hitting the hosted server or your own local one; the CLI/Action only ever talks to whatever suite_id or YAML you give it.

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
