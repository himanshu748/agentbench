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


def _mock_tag(test: dict) -> str:
    # In mock mode the client needs the category hint embedded in the input.
    return f"{test['input']}\n[category:{test['category']}]" if mesh_client.MOCK else test["input"]


async def _run_one(sem: asyncio.Semaphore, system_prompt: str, test: dict, model: str) -> dict:
    async with sem:
        resp = await mesh_client.chat(model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _mock_tag(test)},
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


def _recommend(results: list, models: list) -> dict:
    if not models:
        return {"per_model": [], "recommended": None, "reason": "no models were run"}
    stats = []
    for m in models:
        rs = [r for r in results if r["model"] == m]
        passed = sum(1 for r in rs if r["pass"])
        cost = sum(r["cost_usd"] for r in rs)
        avg_latency = sum(r["latency_ms"] for r in rs) / max(len(rs), 1)
        stats.append({"model": m, "passed": passed, "total": len(rs),
                      "pass_rate": round(passed / len(rs), 4) if rs else 0.0,
                      "total_cost_usd": round(cost, 6),
                      "avg_latency_ms": int(avg_latency)})
    # Rank by pass *rate* (not raw count) so a model whose cells all errored out
    # can never outrank one that actually answered, then cheapest, then fastest.
    best = sorted(stats, key=lambda s: (-s["pass_rate"], s["total_cost_usd"],
                                        s["avg_latency_ms"]))[0]
    best_reason = (f"{best['model']}: {best['passed']}/{best['total']} passed, "
                   f"${best['total_cost_usd']:.4f} total, {best['avg_latency_ms']}ms avg")
    return {"per_model": stats, "recommended": best["model"], "reason": best_reason}


async def execute_run(run_id: str, suite: dict, models: list):
    run = RUNS[run_id]
    try:
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [_run_one(sem, suite["system_prompt"], t, m)
                 for t in suite["tests"] for m in models]
        pairs = [(t, m) for t in suite["tests"] for m in models]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        run["results"] = [r if not isinstance(r, BaseException) else cell_error(t, m, r)
                          for r, (t, m) in zip(settled, pairs)]
        run["summary"] = _recommend(run["results"], models)

        # P1: suggested prompt fix for failing cases (one call, strongest model)
        failures = [r for r in run["results"] if not r["pass"]]
        if failures:
            run["suggested_prompt"] = await judge.suggest_fix(
                suite["system_prompt"],
                [{"input": f["input"], "category": f["category"], "reason": f["reason"]}
                 for f in failures[:8]])
        run["status"] = "done"
    except Exception as e:
        run["status"] = "error"
        run["error"] = str(e)


def _new_run(suite: dict, models: list) -> str:
    run_id = uuid.uuid4().hex[:12]
    _remember(run_id, {"run_id": run_id, "status": "running", "suite_name": suite.get("name"),
                       "suite_id": suite.get("suite_id"),
                       "system_prompt": suite.get("system_prompt"),
                       "models": models, "results": [], "summary": None,
                       "suggested_prompt": None, "mock": mesh_client.MOCK})
    return run_id


def start_run(suite: dict, models: list) -> str:
    run_id = _new_run(suite, models)
    asyncio.create_task(execute_run(run_id, suite, models))
    return run_id


async def run_sync(suite: dict, models: list) -> dict:
    """Execute a run to completion and return it (serverless-friendly)."""
    run_id = _new_run(suite, models)
    await execute_run(run_id, suite, models)
    return RUNS[run_id]
