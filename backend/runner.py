"""Test execution engine: runs suite × models concurrently, judges each result."""
import asyncio
import collections
import uuid

from . import mesh_client, judge

# In-memory store, bounded so a long-lived instance cannot grow without limit.
# Durable sharing lives in storage.py (Supabase); this is just the hot cache.
MAX_RUNS_CACHED = 200
RUNS: collections.OrderedDict[str, dict] = collections.OrderedDict()

CONCURRENCY = 6


def _remember(run_id: str, run: dict) -> dict:
    RUNS[run_id] = run
    RUNS.move_to_end(run_id)
    while len(RUNS) > MAX_RUNS_CACHED:
        RUNS.popitem(last=False)
    return run


def cell_error(test: dict, model: str, exc: BaseException) -> dict:
    """A result cell standing in for a test whose execution raised.

    Every consumer (SSE stream, sync run, CLI table, report page) expects the
    full cell shape, so a crash has to become a failing cell rather than a
    missing one — a missing cell would stall the stream's progress count.
    """
    return {"input": test.get("input", ""), "category": test.get("category", "unknown"),
            "model": model, "latency_ms": 0, "cost_usd": 0.0, "response": "",
            "status": "error", "pass": False,
            "reason": f"Test execution failed: {type(exc).__name__}: {exc}"[:300]}


def _mock_tag(test: dict, attempt: int = 0) -> str:
    # In mock mode the client needs the category hint embedded in the input, and
    # an attempt marker so repeated runs vary (the mock seed is derived from the
    # message text). Neither marker is added in live mode, so real requests are
    # untouched.
    if not mesh_client.MOCK:
        return test["input"]
    tag = f"{test['input']}\n[category:{test['category']}]"
    return f"{tag}\n[attempt:{attempt}]" if attempt else tag


async def _run_one(sem: asyncio.Semaphore, system_prompt: str, test: dict, model: str,
                   attempt: int = 0) -> dict:
    async with sem:
        resp = await mesh_client.chat(model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _mock_tag(test, attempt)},
        ], label=f"test:{test['category']}")
    base = {
        "input": test["input"], "category": test["category"], "model": model,
        "latency_ms": resp["latency_ms"], "cost_usd": resp["cost_usd"],
        "response": resp["content"],
    }
    if resp["error"]:
        return {**base, "status": "error", "pass": False,
                "reason": f"Model call failed: {resp['error']}"}
    verdict = await judge.judge(test["category"], test["input"], resp["content"],
                                custom_rubric=test.get("rubric"))
    return {**base, "status": "done", "pass": verdict["pass"],
            "confidence": verdict["confidence"], "reason": verdict["reason"],
            "cost_usd": resp["cost_usd"] + verdict.get("judge_cost_usd", 0.0)}


def _aggregate_repeats(cells: list) -> dict:
    """Collapse N repeats of one (test, model) cell into a single verdict.

    LLM output is non-deterministic, so the same test can pass on one call and
    fail on the next. Running each cell N times and aggregating turns that
    variance into a measurement: `runs`, `passes`, `pass_rate` and a `flaky`
    flag (some but not all repeats passed). The headline `pass` is a majority
    vote, so a single repeat behaves exactly like the old single-run path.
    """
    runs = len(cells)
    passes = sum(1 for c in cells if c["pass"])
    verdict_pass = passes * 2 >= runs  # majority; ties count as pass
    flaky = 0 < passes < runs
    base = dict(cells[0])
    # Prefer a representative reason from the losing side of a flaky/failed cell,
    # so the surfaced reason explains why it can fail.
    losers = [c for c in cells if c["pass"] != verdict_pass]
    base["reason"] = (losers[0] if losers else cells[0]).get("reason", "")
    base.update({
        "pass": verdict_pass,
        "status": "done" if all(c.get("status") == "done" for c in cells) else "error",
        "runs": runs,
        "passes": passes,
        "pass_rate": round(passes / runs, 4) if runs else 0.0,
        "flaky": flaky,
        "latency_ms": int(sum(c["latency_ms"] for c in cells) / runs),
        "cost_usd": round(sum(c["cost_usd"] for c in cells), 8),
        "confidence": round(sum(c.get("confidence", 0.0) for c in cells) / runs, 3),
        "attempts": [{"pass": c["pass"], "reason": c.get("reason", ""),
                      "latency_ms": c["latency_ms"]} for c in cells],
    })
    return base


def _flakiness_summary(results: list) -> dict:
    """Aggregate flakiness across every cell in a run."""
    repeated = [r for r in results if r.get("runs", 1) > 1]
    if not repeated:
        return {"measured": False, "repeat": 1, "flaky_count": 0,
                "consistency": 1.0, "flaky_cells": []}
    flaky = [r for r in repeated if r.get("flaky")]
    flaky_cells = sorted(
        ({"input": r["input"], "category": r["category"], "model": r["model"],
          "passes": r["passes"], "runs": r["runs"], "pass_rate": r["pass_rate"]}
         for r in flaky),
        key=lambda c: abs(c["pass_rate"] - 0.5))  # most-uncertain first
    return {"measured": True, "repeat": repeated[0]["runs"],
            "flaky_count": len(flaky), "total_cells": len(repeated),
            "consistency": round(1 - len(flaky) / len(repeated), 4),
            "flaky_cells": flaky_cells[:20]}


def _recommend(results: list, models: list) -> dict:
    if not models:
        return {"per_model": [], "recommended": None, "reason": "no models were run"}
    stats = []
    for m in models:
        rs = [r for r in results if r["model"] == m]
        passed = sum(1 for r in rs if r["pass"])
        cost = sum(r["cost_usd"] for r in rs)
        avg_latency = sum(r["latency_ms"] for r in rs) / max(len(rs), 1)
        flaky = sum(1 for r in rs if r.get("flaky"))
        stats.append({"model": m, "passed": passed, "total": len(rs),
                      "pass_rate": round(passed / len(rs), 4) if rs else 0.0,
                      "flaky": flaky,
                      "total_cost_usd": round(cost, 6),
                      "avg_latency_ms": int(avg_latency)})
    # Rank by pass *rate* (not raw count) so a model whose cells all errored out
    # can never outrank one that actually answered, then cheapest, then fastest.
    best = sorted(stats, key=lambda s: (-s["pass_rate"], s["total_cost_usd"],
                                        s["avg_latency_ms"]))[0]
    best_reason = (f"{best['model']}: {best['passed']}/{best['total']} passed, "
                   f"${best['total_cost_usd']:.4f} total, {best['avg_latency_ms']}ms avg")
    return {"per_model": stats, "recommended": best["model"], "reason": best_reason}


async def execute_run(run_id: str, suite: dict, models: list, repeat: int = 1):
    run = RUNS[run_id]
    try:
        repeat = max(1, int(repeat))
        sem = asyncio.Semaphore(CONCURRENCY)
        # One task per (test, model, attempt). Attempts of the same cell are
        # aggregated afterwards into a single flakiness-aware result.
        plan = [(t, m, a) for t in suite["tests"] for m in models
                for a in range(repeat)]
        tasks = [_run_one(sem, suite["system_prompt"], t, m, a) for t, m, a in plan]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        raw = [r if not isinstance(r, BaseException) else cell_error(t, m, r)
               for r, (t, m, a) in zip(settled, plan)]

        if repeat > 1:
            grouped: dict = {}
            for cell in raw:
                grouped.setdefault((cell["input"], cell["category"], cell["model"]),
                                   []).append(cell)
            results = [_aggregate_repeats(cells) for cells in grouped.values()]
        else:
            results = raw

        run["results"] = results
        run["repeat"] = repeat
        run["summary"] = _recommend(results, models)
        run["flakiness"] = _flakiness_summary(results)

        # P1: suggested prompt fix for failing cases (one call, strongest model)
        failures = [r for r in results if not r["pass"]]
        if failures:
            run["suggested_prompt"] = await judge.suggest_fix(
                suite["system_prompt"],
                [{"input": f["input"], "category": f["category"], "reason": f["reason"]}
                 for f in failures[:8]])
        run["status"] = "done"
    except Exception as e:
        run["status"] = "error"
        run["error"] = str(e)


def _new_run(suite: dict, models: list, repeat: int = 1) -> str:
    run_id = uuid.uuid4().hex[:12]
    _remember(run_id, {"run_id": run_id, "status": "running", "suite_name": suite.get("name"),
                       "suite_id": suite.get("suite_id"),
                       "system_prompt": suite.get("system_prompt"),
                       "models": models, "repeat": repeat, "results": [], "summary": None,
                       "flakiness": None, "suggested_prompt": None, "mock": mesh_client.MOCK})
    return run_id


def start_run(suite: dict, models: list, repeat: int = 1) -> str:
    run_id = _new_run(suite, models, repeat)
    asyncio.create_task(execute_run(run_id, suite, models, repeat))
    return run_id


async def run_sync(suite: dict, models: list, repeat: int = 1) -> dict:
    """Execute a run to completion and return it (serverless-friendly)."""
    run_id = _new_run(suite, models, repeat)
    await execute_run(run_id, suite, models, repeat)
    return RUNS[run_id]
