# AgentBench

**CI/CD for AI prompts and agents.** Behavior tests for your prompt, run across the models on MeshAPI, with pass rates, real cost, latency, a fix for what fails, and a build-style badge.

Live: **https://agentbench-three.vercel.app** &middot; Built for the MeshAPI Hackathon (July 2026)

Every LLM call (test execution, judge scoring, red team generation, prompt fixes) routes through MeshAPI via one integration point: [backend/mesh_client.py](backend/mesh_client.py).

## Why

LLM output is non-deterministic, so exact-match testing is dead. AgentBench tags each test with a **behavior category** and a judge model (claude-sonnet-5, via MeshAPI) scores every response against that category's rubric.

19 built-in categories. The first 16 are scored by the judge model against a rubric; the three `tool_*` ones are asserted directly in Python (see below):

| | | |
| --- | --- | --- |
| `requires_clarification` | `injection_resistance` | `conciseness` |
| `tone_deescalation` | `no_fabrication` | `format_compliance` |
| `language_match` | `pii_refusal` | `instruction_hierarchy` |
| `context_grounding` | `over_refusal` | `safety_refusal` |
| `numeric_accuracy` | `escalation` | `scope_adherence` |
| `code_correctness` | `tool_selection` | `tool_arguments` |
| `tool_restraint` | | |

Plus `custom` with your own plain-English rubric per test. Run `agentbench categories` to list them from any deployment. `over_refusal` is deliberately the mirror of `safety_refusal`: refusing a benign request is scored as a failure, not a safe default.

## Features

- **Tool calling, asserted not judged**: declare your tools in the suite and test whether the agent picks the right one, fills its arguments from the user's words, and stays its hand when no tool is warranted. See below.
- **Live streaming runs**: cells fill in one by one over SSE as each model responds and is judged.
- **Fix and re-run**: failing runs get an AI-drafted prompt fix, one click re-benchmarks it, before/after shown.
- **Red team generator**: paste a system prompt, get an adversarial suite written and runnable in seconds.
- **Benchmark MeshAPI `auto`**: test smart routing against fixed models on your own workload.
- **Shareable reports**: every run persists (Supabase) at `/r/<run_id>`.
- **Regression history** per suite with pass-rate deltas.
- **Quality gate badge**: CI-style SVG per run with copyable README markdown.
- **Flakiness detection** (`--repeat N`): run each test N times and get a per-cell consistency score. LLM non-determinism is the whole reason exact-match testing fails — this measures it directly, and `--fail-on-flaky` gates on it.
- **Regression gate** (`--baseline`): diff a run against a saved report and fail CI specifically on cells that *newly* broke, independent of the absolute pass rate.
- **CI-native output**: JUnit XML (`--junit`) for test reporters, a JSON artifact (`--output`), a GitHub Actions job summary, and a cost `estimate` before you spend anything.
- **17 pre-built suites** across support, coding, moderation, healthcare, sales, HR, brand voice, RAG grounding, legal intake, fintech, devops runbooks, education, travel, safety guardrails, structured data extraction, and two tool-calling agents — together exercising all 19 categories.
- **Suite builder UI**: create tests in the console with a form (category dropdown, inline rubric editor for custom tests). No YAML needed.

## CLI and GitHub Action

```bash
pipx install "git+https://github.com/himanshu748/agentbench"   # or: pip install ...

agentbench run suites/customer-support.yaml \
  --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
# exit code 1 if the recommended model fails any test -> fails your CI job
# point at your own server with --url or AGENTBENCH_URL=http://localhost:8000

# measure flakiness, gate a PR against a baseline, and export for CI:
agentbench run suites/customer-support.yaml --models auto \
  --repeat 5 --baseline main.json --fail-on-regression \
  --junit results.xml --output run.json

agentbench estimate suites/customer-support.yaml --models auto  # cost before spending
agentbench validate suites/customer-support.yaml                # lint without running
agentbench suites            # what suites the server has
agentbench models            # what you can benchmark against
agentbench categories        # the built-in behavior categories
```

| flag (on `run`) | what it does |
| --- | --- |
| `--models a,b` | comma-separated MeshAPI model ids, up to 4 (`auto` = smart routing) |
| `--repeat N` | run each test N times; reports per-cell flakiness (up to 10) |
| `--min-pass-rate 0.8` | allow a known-flaky slice instead of demanding 100% |
| `--baseline FILE` | diff against a prior `--output` report; shows regressions and fixes |
| `--fail-on-regression` | exit non-zero on any cell that passed in the baseline but now fails |
| `--fail-on-flaky` | exit non-zero if any cell is flaky (needs `--repeat > 1`) |
| `--junit FILE` | write a JUnit XML report for CI test reporters |
| `--output FILE` | write the full JSON report (use as the next run's `--baseline`) |
| `--json` | full JSON report on stdout — the banner goes to stderr, so `\| jq` works |
| `--url` / `AGENTBENCH_URL` | point at your own deployment |
| `--timeout` | per-request timeout, default 180s |

**Exit codes** — CI needs to tell a prompt regression apart from a broken tool:

| code | meaning |
| --- | --- |
| `0` | the recommended model met the pass threshold |
| `1` | behavior tests failed — this is what blocks your merge |
| `2` | the benchmark could not run (bad arguments, unreachable server, server error) |

Running inside GitHub Actions, the CLI also writes a markdown table of the run
(per-model pass rate, cost, latency, and every failing test) to the job summary.

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

## Testing tool calls

An agent is mostly its tool calls, so those get tested too. Declare the tools
once at suite level; each tool test then states what should have happened.

```yaml
name: Support Agent
system_prompt: "You are an order-support agent with tools..."
tools:
  - name: lookup_order
    description: Look up the status of one order.
    parameters:                       # plain JSON Schema
      type: object
      properties:
        order_id: { type: string }
      required: [order_id]
  - name: issue_refund
    description: Refund an order up to $500.
    parameters:
      type: object
      properties:
        order_id:   { type: string }
        amount_usd: { type: number }
tests:
  - input: "Where is my order A-1183?"
    category: tool_selection          # the right tool was called
    expect_tool: lookup_order

  - input: "Refund order C-7781 for $42.00."
    category: tool_arguments          # right tool AND right argument values
    expect_tool: issue_refund
    expect_args:
      order_id: "C-7781"
      amount_usd: 42.0

  - input: "How many days do I have to return something?"
    category: tool_restraint          # no tool call was warranted
```

The tools are sent to MeshAPI in the OpenAI `tools` format with
`tool_choice: auto`, so the model decides whether to call anything — which is
what makes `tool_restraint` a real test rather than a formality.

**These three categories are asserted, not judged.** A tool call is structured
data, so AgentBench compares it directly in Python instead of asking a model.
That makes a tool test cost one MeshAPI call instead of two, and makes its
verdict reproducible rather than subject to a judge's mood.

Argument matching is deliberately lenient where leniency is correct and strict
where it is not:

| | |
| --- | --- |
| `"A-1183"` vs `"a-1183 "` | passes (case and whitespace insensitive) |
| `42` vs `42.0` | passes (compared by value) |
| an extra correct argument | passes (expected args are a subset, not an exact match) |
| a missing or wrong required argument | fails, and the reason names which one |

Behavior tests can sit in the same suite as tool tests. A `tool-agent` suite
that also tests `injection_resistance` catches an agent going off the rails
while it holds real capabilities. Two shipped suites do exactly this:
`suites/tool-agent.yaml` and `suites/api-orchestration.yaml`.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "MESH_API_KEY=rsk_..." > .env         # from meshapi.ai
uvicorn backend.main:app --port 8000       # open http://localhost:8000
```

No key? Mock mode keeps the whole flow demoable offline — every behavior
category has canned good and bad answers, so runs pass and fail realistically
rather than uniformly.

## Tests

AgentBench has its own test suite, and it runs in mock mode — no MeshAPI key, no
network, no spend — so it is safe on forks and pull requests.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

350+ tests covering: judge verdict parsing (including malformed model output),
the recommendation ranking, MeshAPI retry/backoff and response-shape handling,
flakiness aggregation and the repeat path, every abuse guard, the cost estimator,
SSE streaming, badge rendering and escaping, CLI exit codes, JUnit XML output,
baseline regression diffing, and validation of all 15 shipped suite YAMLs against
the same rules a user upload goes through.

`.github/workflows/ci.yml` runs it on 3.10-3.12, byte-compiles the CLI on 3.9
(the floor the GitHub Action's runner needs, since only `cli/` is packaged), and
smoke-tests the CLI end to end against a local `uvicorn` instance.

## Architecture

- FastAPI, one Vercel Python serverless function (`api/index.py`), static frontend served by the same app.
- `mesh_client.py`: the only file that talks to MeshAPI. Model catalog and per-token pricing pulled live from `/v1/models` (531 models, 396 chat). Pooled connections, and bounded exponential-backoff retries on 429/5xx/timeouts so a rate limit reads as a slow request, not a failed behavior test.
- `judge.py`: category rubrics, JSON-mode judging, prompt-fix generation.
- `runner.py`: concurrent execution, per-cell timeout isolation, repeat-and-aggregate for flakiness, recommendation (pass rate, then cost, then latency).
- `storage.py`: shared reports in Supabase (RLS-guarded anon key).
- Abuse guards: per-IP rate limiting, a 40-cell run ceiling, suite size limits, and bounded in-memory caches — applied to every endpoint that can spend credit (`/api/runs`, `/api/runs/stream`, `/api/fix-rerun`, `/api/redteam`).

## License

MIT — see [LICENSE](LICENSE).
