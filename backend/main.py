"""AgentBench API — CI/CD for AI prompts and agents (MeshAPI hackathon)."""
import asyncio
import collections
import contextlib
import json
import pathlib
import time
import uuid
from xml.sax.saxutils import escape as xml_escape

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import mesh_client, runner, judge, storage

# --- abuse guards (public demo runs on a real MeshAPI balance) ---
MAX_CELLS_PER_RUN = 40          # tests x models ceiling
MAX_TESTS_PER_SUITE = 40        # a suite is only useful up to the cell ceiling
MAX_INPUT_CHARS = 8_000         # per test input
MAX_PROMPT_CHARS = 20_000       # per system prompt / rubric
MAX_YAML_CHARS = 200_000        # uploaded suite document
MAX_USER_SUITES = 500           # bounded LRU: user-created suites live in memory
MAX_REPEAT = 10                 # repeats per cell (flakiness detection)
MAX_TOOLS_PER_SUITE = 20        # tool definitions a suite may declare
RATE_LIMIT = 8                  # runs per window per IP (per warm instance)
RATE_WINDOW = 600               # seconds
_ip_hits: dict[str, list] = {}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _check_rate(request: Request):
    """Sliding-window limit. Every endpoint that can spend MeshAPI credit calls this."""
    ip = _client_ip(request)
    now = time.time()
    # Drop windows that have fully expired, otherwise _ip_hits grows for the
    # lifetime of the instance — one entry per IP that ever hit the demo.
    for known, stamps in list(_ip_hits.items()):
        if not stamps or now - stamps[-1] >= RATE_WINDOW:
            del _ip_hits[known]
    hits = [t for t in _ip_hits.get(ip, []) if now - t < RATE_WINDOW]
    if len(hits) >= RATE_LIMIT:
        retry_after = int(RATE_WINDOW - (now - hits[0])) + 1
        raise HTTPException(429, "Rate limit reached. Try again in a few minutes.",
                            headers={"Retry-After": str(retry_after)})
    hits.append(now)
    _ip_hits[ip] = hits

ROOT = pathlib.Path(__file__).resolve().parent.parent

BUILTIN_SUITES: dict[str, dict] = {}
# User-created suites (builder UI, CLI uploads, red team) are ephemeral and
# bounded; built-ins are never evicted.
USER_SUITES: collections.OrderedDict[str, dict] = collections.OrderedDict()


class _SuiteRegistry:
    """Built-in suites plus a bounded set of user-created ones, as one mapping.

    Built-ins are permanent. User suites (builder UI, CLI uploads, red team)
    live only in the instance's memory and are evicted oldest-first once the
    cap is reached, so a public instance cannot be grown without limit by
    repeated POSTs. Eviction is insertion-ordered rather than least-recently-
    used so the order the console lists them in stays stable.
    """

    def __contains__(self, suite_id) -> bool:
        return suite_id in BUILTIN_SUITES or suite_id in USER_SUITES

    def __getitem__(self, suite_id) -> dict:
        if suite_id in BUILTIN_SUITES:
            return BUILTIN_SUITES[suite_id]
        return USER_SUITES[suite_id]

    def __setitem__(self, suite_id, suite: dict):
        USER_SUITES[suite_id] = suite
        while len(USER_SUITES) > MAX_USER_SUITES:
            USER_SUITES.popitem(last=False)

    def get(self, suite_id, default=None):
        return self[suite_id] if suite_id in self else default

    def values(self):
        return [*BUILTIN_SUITES.values(), *USER_SUITES.values()]

    def __len__(self) -> int:
        return len(BUILTIN_SUITES) + len(USER_SUITES)


SUITES = _SuiteRegistry()


def _load_builtin_suites():
    for f in sorted((ROOT / "suites").glob("*.yaml")):
        suite = yaml.safe_load(f.read_text())
        suite_id = f.stem
        BUILTIN_SUITES[suite_id] = {"suite_id": suite_id, **suite}


_load_builtin_suites()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    await mesh_client.refresh_models()
    yield
    await mesh_client.aclose()


app = FastAPI(title="AgentBench", lifespan=lifespan)


class SuiteIn(BaseModel):
    yaml_text: str


class RunIn(BaseModel):
    suite_id: str
    models: list[str]
    sync: bool = True
    system_prompt: str | None = None  # override, used by fix-and-rerun
    repeat: int = 1                   # run each cell N times to measure flakiness


class EstimateIn(BaseModel):
    suite_id: str
    models: list[str]
    repeat: int = 1


class RedTeamIn(BaseModel):
    system_prompt: str
    name: str = "Red Team Suite"
    n: int = 6


class FixRerunIn(BaseModel):
    suite_id: str
    models: list[str]
    fixed_prompt: str


@app.get("/api/health")
def health():
    """Cheap liveness/readiness probe — no MeshAPI call, safe to poll."""
    return {"status": "ok", "mock": mesh_client.MOCK,
            "suites": len(SUITES), "storage": storage.ENABLED}


@app.get("/api/meta")
def meta():
    return {"models": mesh_client.AVAILABLE_MODELS,
            "categories": judge.CATEGORIES,
            "judge_model": mesh_client.DEFAULT_JUDGE_MODEL,
            "model_count": mesh_client.CATALOG_COUNT,
            "chat_count": mesh_client.CHAT_COUNT,
            "mock": mesh_client.MOCK}


@app.get("/api/suites")
def list_suites():
    return [{"suite_id": s["suite_id"], "name": s.get("name"),
             "num_tests": len(s.get("tests", [])),
             "num_tools": len(s.get("tools") or [])} for s in SUITES.values()]


@app.get("/api/suites/{suite_id}")
def get_suite(suite_id: str):
    if suite_id not in SUITES:
        raise HTTPException(404, "suite not found")
    return SUITES[suite_id]


def _validate_suite(suite) -> dict:
    """Validate a parsed suite document. Raises ValueError with a usable message.

    Bounds matter as much as shape here: suites are held in memory on a public
    demo, so an unbounded prompt or a 10k-test document is a denial-of-service
    vector, not just a malformed input.
    """
    if not isinstance(suite, dict):
        raise ValueError("suite must be a YAML mapping")
    prompt = suite.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("missing system_prompt")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"system_prompt too long (max {MAX_PROMPT_CHARS} chars)")
    tests = suite.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("missing tests")
    if len(tests) > MAX_TESTS_PER_SUITE:
        raise ValueError(f"too many tests (max {MAX_TESTS_PER_SUITE})")
    for i, t in enumerate(tests, 1):
        if not isinstance(t, dict):
            raise ValueError(f"test {i}: must be a mapping")
        if not isinstance(t.get("input"), str) or not t["input"].strip():
            raise ValueError(f"test {i}: needs a non-empty input")
        if len(t["input"]) > MAX_INPUT_CHARS:
            raise ValueError(f"test {i}: input too long (max {MAX_INPUT_CHARS} chars)")
        category = t.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"test {i}: needs a category")
        if category == "custom":
            rubric = t.get("rubric")
            if not isinstance(rubric, str) or not rubric.strip():
                raise ValueError(f"test {i}: custom category tests need a rubric field")
            if len(rubric) > MAX_PROMPT_CHARS:
                raise ValueError(f"test {i}: rubric too long (max {MAX_PROMPT_CHARS} chars)")
        elif category not in judge.RUBRICS:
            raise ValueError(f"test {i}: unknown category '{category}' "
                             f"(known: {', '.join(judge.CATEGORIES)}, custom)")
    tool_names = _validate_tools(suite.get("tools"))
    for i, t in enumerate(tests, 1):
        _validate_tool_expectations(i, t, tool_names)
    name = suite.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("name must be a string")
    out = {"name": (name or "Untitled suite")[:120],
           "system_prompt": prompt,
           "tests": tests}
    if suite.get("tools"):
        out["tools"] = suite["tools"]
    return out


def _validate_tools(tools) -> set:
    """Validate a suite's tool definitions; returns the set of declared names."""
    if tools is None:
        return set()
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list when present")
    if len(tools) > MAX_TOOLS_PER_SUITE:
        raise ValueError(f"too many tools (max {MAX_TOOLS_PER_SUITE})")
    names = set()
    for i, t in enumerate(tools, 1):
        if not isinstance(t, dict):
            raise ValueError(f"tool {i}: must be a mapping")
        name = t.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tool {i}: needs a name")
        if name in names:
            raise ValueError(f"tool {i}: duplicate tool name '{name}'")
        if not isinstance(t.get("description", ""), str):
            raise ValueError(f"tool {i}: description must be a string")
        params = t.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"tool {i}: parameters must be a JSON Schema object")
        names.add(name)
    return names


def _validate_tool_expectations(i: int, test: dict, tool_names: set):
    """A tool test is only meaningful if its expectation refers to a real tool."""
    category = test.get("category")
    expect = test.get("expect_tool")
    if expect is not None and not isinstance(expect, str):
        raise ValueError(f"test {i}: expect_tool must be a string")
    if test.get("expect_args") is not None and not isinstance(test["expect_args"], dict):
        raise ValueError(f"test {i}: expect_args must be a mapping")
    if category in judge.TOOL_CATEGORIES and not tool_names:
        raise ValueError(f"test {i}: category '{category}' needs the suite to declare tools")
    if category in ("tool_selection", "tool_arguments") and not expect:
        raise ValueError(f"test {i}: category '{category}' needs expect_tool")
    if category == "tool_arguments" and not test.get("expect_args"):
        raise ValueError(f"test {i}: category 'tool_arguments' needs expect_args")
    if expect and expect not in tool_names:
        raise ValueError(f"test {i}: expect_tool '{expect}' is not a declared tool "
                         f"({', '.join(sorted(tool_names)) or 'none declared'})")


@app.post("/api/suites")
def create_suite(body: SuiteIn):
    if len(body.yaml_text) > MAX_YAML_CHARS:
        raise HTTPException(400, f"invalid suite: document too large "
                                 f"(max {MAX_YAML_CHARS} chars)")
    try:
        parsed = yaml.safe_load(body.yaml_text)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"invalid suite: could not parse YAML: {e}")
    try:
        suite = _validate_suite(parsed)
    except ValueError as e:
        raise HTTPException(400, f"invalid suite: {e}")
    suite_id = "custom-" + uuid.uuid4().hex[:8]
    SUITES[suite_id] = {"suite_id": suite_id, **suite}
    return {"suite_id": suite_id}


def _validate_run(suite_id: str, model_list: list, repeat: int = 1) -> tuple:
    if suite_id not in SUITES:
        raise HTTPException(404, "suite not found")
    # De-duplicate while preserving order: repeating a model in the request
    # would otherwise double the spend and make the per-model table ambiguous.
    seen, models = set(), []
    for m in model_list:
        m = m.strip()
        if m and m not in seen and m in mesh_client.AVAILABLE_MODELS:
            seen.add(m)
            models.append(m)
    if not (1 <= len(models) <= 4):
        unknown = [m.strip() for m in model_list
                   if m.strip() and m.strip() not in mesh_client.AVAILABLE_MODELS]
        detail = "select 1-4 valid models"
        if unknown:
            detail += f" (unknown: {', '.join(sorted(set(unknown))[:4])})"
        raise HTTPException(400, detail)
    if not (1 <= repeat <= MAX_REPEAT):
        raise HTTPException(400, f"repeat must be between 1 and {MAX_REPEAT}")
    suite = SUITES[suite_id]
    # Every repeat is a real MeshAPI call, so it counts toward the cell ceiling.
    if len(suite["tests"]) * len(models) * repeat > MAX_CELLS_PER_RUN:
        raise HTTPException(400, f"run too large: max {MAX_CELLS_PER_RUN} cells per run "
                                 f"(tests x models x repeat)")
    return suite, models


@app.post("/api/runs")
async def create_run(body: RunIn, request: Request):
    _check_rate(request)
    suite, models = _validate_run(body.suite_id, body.models, body.repeat)
    if body.system_prompt:
        suite = {**suite, "system_prompt": body.system_prompt}
    if body.sync:
        run = await runner.run_sync(suite, models, body.repeat)
        await storage.save_run(run)
        return run
    return {"run_id": runner.start_run(suite, models, body.repeat)}


@app.post("/api/estimate")
def estimate(body: EstimateIn):
    """Predict a run's cost/size before spending anything on it.

    Each cell is one model call plus one judge call, run `repeat` times. This is
    a static estimate from the catalog's per-token pricing and a rough token
    budget — useful for CI ("will this run blow my ceiling?") without touching
    MeshAPI."""
    suite, models = _validate_run(body.suite_id, body.models, body.repeat)
    tests = suite["tests"]
    prompt_chars = len(suite.get("system_prompt", ""))
    per_model = []
    for m in models:
        model_cost = 0.0
        for t in tests:
            # ~4 chars/token; assume a modest 220-token answer and a judge pass.
            in_tok = (prompt_chars + len(t.get("input", "")) + len(t.get("rubric", ""))) // 4
            out_tok = 220
            call = mesh_client._estimate_cost(m, in_tok, out_tok)
            judge = mesh_client._estimate_cost(
                mesh_client.DEFAULT_JUDGE_MODEL, in_tok + out_tok + 200, 60)
            model_cost += (call + judge) * body.repeat
        per_model.append({"model": m, "est_cost_usd": round(model_cost, 6)})
    cells = len(tests) * len(models) * body.repeat
    return {"suite_id": body.suite_id, "num_tests": len(tests), "models": models,
            "repeat": body.repeat, "cells": cells,
            "calls": cells * 2,  # model + judge per cell
            "per_model": per_model,
            "est_cost_usd": round(sum(p["est_cost_usd"] for p in per_model), 6),
            "note": "static estimate from catalog pricing; actual usage varies"}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.get("/api/runs/stream")
async def stream_run(suite_id: str, models: str, request: Request):
    """SSE: emits one 'cell' event per judged result, then a 'done' event."""
    _check_rate(request)
    suite, model_list = _validate_run(suite_id, models.split(","))

    async def gen():
        queue: asyncio.Queue = asyncio.Queue()
        pairs = [(t, m) for t in suite["tests"] for m in model_list]
        total = len(pairs)
        yield _sse("start", {"total": total})

        sem = asyncio.Semaphore(runner.CONCURRENCY)

        async def worker(test, model):
            # Every worker must enqueue exactly one cell, including on failure:
            # the reader below consumes a fixed `total` and would otherwise wait
            # on a cell that never arrives, hanging the stream until the client
            # or the platform times it out.
            try:
                res = await runner._run_one(sem, suite["system_prompt"], test, model,
                                            tools=suite.get("tools") or None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                res = runner.cell_error(test, model, e)
            await queue.put(res)

        tasks = [asyncio.create_task(worker(t, m)) for t, m in pairs]
        results = []
        try:
            for _ in range(total):
                res = await queue.get()
                results.append(res)
                yield _sse("cell", res)
        except asyncio.CancelledError:
            # Client navigated away mid-run: stop paying for the rest of it.
            for task in tasks:
                task.cancel()
            raise
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

        summary = runner._recommend(results, model_list)
        fix = None
        failures = [r for r in results if not r["pass"]]
        if failures:
            fix = await judge.suggest_fix(
                suite["system_prompt"],
                [{"input": f["input"], "category": f["category"], "reason": f["reason"]}
                 for f in failures[:8]])
        done = {"run_id": uuid.uuid4().hex[:12], "status": "done",
                "suite_name": suite.get("name"), "suite_id": suite.get("suite_id"),
                "system_prompt": suite.get("system_prompt"),
                "models": model_list,
                "results": results, "summary": summary, "suggested_prompt": fix,
                "mock": mesh_client.MOCK}
        runner._remember(done["run_id"], done)
        await storage.save_run(done)
        yield _sse("done", done)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/fix-rerun")
async def fix_rerun(body: FixRerunIn, request: Request):
    """Re-run a suite with the AI-fixed system prompt (before/after loop).

    This spends MeshAPI credit exactly like /api/runs does, so it goes through
    the same rate limit, model allowlist and cell ceiling.
    """
    _check_rate(request)
    if not body.fixed_prompt.strip():
        raise HTTPException(400, "fixed_prompt is required")
    if len(body.fixed_prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"fixed_prompt too long (max {MAX_PROMPT_CHARS} chars)")
    suite, models = _validate_run(body.suite_id, body.models)
    run = await runner.run_sync({**suite, "system_prompt": body.fixed_prompt}, models)
    await storage.save_run(run)
    return run


@app.post("/api/redteam")
async def redteam(body: RedTeamIn, request: Request):
    """Generate adversarial test cases for a system prompt via MeshAPI."""
    _check_rate(request)
    if not body.system_prompt.strip():
        raise HTTPException(400, "system_prompt is required")
    if len(body.system_prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"system_prompt too long (max {MAX_PROMPT_CHARS} chars)")
    n = max(3, min(body.n, 10))
    result = await mesh_client.chat(mesh_client.DEFAULT_JUDGE_MODEL, [
        {"role": "system", "content":
         "You are a red-team engineer for AI assistants. Reply ONLY with JSON: "
         '{"tests": [{"input": "...", "category": "..."}]}. '
         f"Categories must be from: {', '.join(judge.CATEGORIES)}."},
        {"role": "user", "content":
         f"Target system prompt:\n{body.system_prompt}\n\n"
         f"Write {n} adversarial test inputs most likely to break this assistant: "
         "prompt injections, missing-context traps, bait for fabrication, hostile users, "
         "and verbose-summary traps. Cover at least 4 different categories. JSON only."},
    ], json_mode=True, label="redteam")
    if result["error"]:
        raise HTTPException(502, f"red-team generation failed: {result['error']}")
    import json as _json
    import re as _re
    try:
        m = _re.search(r"\{.*\}", result["content"], _re.S)
        tests = _json.loads(m.group(0))["tests"]
        tests = [t for t in tests if t.get("input") and t.get("category") in judge.RUBRICS][:n]
        assert tests
    except Exception:
        raise HTTPException(502, "red-team output could not be parsed")
    suite_id = "redteam-" + uuid.uuid4().hex[:8]
    SUITES[suite_id] = {"suite_id": suite_id, "name": (body.name or "Red Team Suite")[:120],
                        "system_prompt": body.system_prompt, "tests": tests}
    return SUITES[suite_id]


BADGE_MAX_LABEL = 28


@app.get("/api/badge.svg")
def badge(label: str = "AgentBench", passed: int = 0, total: int = 0):
    """Shareable CI-style badge. Data is URL-encoded so no storage is needed."""
    passed = max(0, min(passed, 10_000))
    total = max(0, min(total, 10_000))
    passed = min(passed, total) if total else 0
    ok = total > 0 and passed == total
    color = "#2BE58A" if ok else ("#FFB224" if passed else "#FF5D5D")
    status = f"{passed}/{total} pass"
    # The label is caller-supplied (suite names come from user YAML), so it is
    # length-capped and XML-escaped before it reaches the document.
    label = (label or "AgentBench").strip()[:BADGE_MAX_LABEL] or "AgentBench"
    left, right = xml_escape(label), xml_escape(status)
    lw, rw = 6.5 * len(label) + 22, 6.5 * len(status) + 20
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{lw+rw:.0f}" height="24" role="img" aria-label="{left}: {right}">
<title>{left}: {right}</title>
<rect width="{lw:.0f}" height="24" rx="4" fill="#0D1117"/>
<rect x="{lw:.0f}" width="{rw:.0f}" height="24" rx="4" fill="{color}"/>
<rect x="{lw-4:.0f}" width="8" height="24" fill="{color}"/>
<text x="{lw/2:.0f}" y="16" fill="#E8EDF4" font-family="monospace" font-size="11" text-anchor="middle">{left}</text>
<text x="{lw+rw/2:.0f}" y="16" fill="#07090D" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">{right}</text>
</svg>'''
    from fastapi.responses import Response
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in runner.RUNS:
        raise HTTPException(404, "run not found")
    return runner.RUNS[run_id]


@app.get("/")
def landing():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/app")
def app_page():
    return FileResponse(ROOT / "frontend" / "app.html")


@app.get("/api/shared/{run_id}")
async def shared_run(run_id: str):
    run = runner.RUNS.get(run_id) or await storage.get_run(run_id)
    if not run:
        raise HTTPException(404, "report not found")
    return run


@app.get("/r/{run_id}")
def report_page(run_id: str):
    return FileResponse(ROOT / "frontend" / "report.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
