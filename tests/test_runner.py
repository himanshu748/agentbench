"""Execution engine: ranking, error cells, and the bounded run cache."""
import pytest

from backend import runner
from tests.conftest import make_suite


def cell(model, passed, cost=0.001, latency=100, category="conciseness"):
    return {"input": "i", "category": category, "model": model, "pass": passed,
            "cost_usd": cost, "latency_ms": latency, "status": "done", "reason": "r"}


class TestRecommend:
    def test_highest_pass_rate_wins(self):
        results = [cell("a", True), cell("a", True), cell("b", True), cell("b", False)]
        s = runner._recommend(results, ["a", "b"])
        assert s["recommended"] == "a"

    def test_cost_breaks_a_pass_rate_tie(self):
        results = [cell("cheap", True, cost=0.001), cell("pricey", True, cost=0.05)]
        assert runner._recommend(results, ["cheap", "pricey"])["recommended"] == "cheap"

    def test_latency_breaks_a_cost_tie(self):
        results = [cell("slow", True, cost=0.01, latency=900),
                   cell("fast", True, cost=0.01, latency=90)]
        assert runner._recommend(results, ["slow", "fast"])["recommended"] == "fast"

    def test_pass_rate_beats_raw_count(self):
        # `many` passes 2 of 4; `few` passes 1 of 1. Ranking on raw count would
        # pick `many` despite it failing half its tests.
        results = ([cell("many", True), cell("many", True),
                    cell("many", False), cell("many", False), cell("few", True)])
        s = runner._recommend(results, ["many", "few"])
        assert s["recommended"] == "few"

    def test_per_model_stats_are_complete(self):
        s = runner._recommend([cell("a", True), cell("a", False)], ["a"])
        stat = s["per_model"][0]
        assert stat == {"model": "a", "passed": 1, "total": 2, "pass_rate": 0.5,
                        "total_cost_usd": 0.002, "avg_latency_ms": 100}

    def test_model_with_no_results_does_not_crash(self):
        s = runner._recommend([], ["ghost"])
        assert s["per_model"][0]["total"] == 0
        assert s["per_model"][0]["pass_rate"] == 0.0

    def test_no_models_returns_empty_summary(self):
        s = runner._recommend([], [])
        assert s["recommended"] is None
        assert s["per_model"] == []

    def test_reason_mentions_the_winner(self):
        s = runner._recommend([cell("a", True)], ["a"])
        assert s["recommended"] in s["reason"]


class TestCellError:
    def test_shape_matches_a_normal_cell(self):
        err = runner.cell_error({"input": "hi", "category": "conciseness"},
                                "openai/gpt-4o-mini", RuntimeError("boom"))
        for key in ("input", "category", "model", "latency_ms", "cost_usd",
                    "response", "status", "pass", "reason"):
            assert key in err
        assert err["pass"] is False
        assert err["status"] == "error"
        assert "RuntimeError" in err["reason"]

    def test_tolerates_a_malformed_test_dict(self):
        err = runner.cell_error({}, "m", ValueError("x"))
        assert err["category"] == "unknown"
        assert err["input"] == ""

    def test_reason_is_truncated(self):
        err = runner.cell_error({}, "m", ValueError("y" * 900))
        assert len(err["reason"]) <= 300

    def test_error_cells_count_as_failures_in_ranking(self):
        results = [runner.cell_error({"category": "c"}, "broken", RuntimeError("x")),
                   cell("working", True)]
        assert runner._recommend(results, ["broken", "working"])["recommended"] == "working"


class TestRunCache:
    def test_cache_is_bounded(self, monkeypatch):
        monkeypatch.setattr(runner, "MAX_RUNS_CACHED", 3)
        for i in range(10):
            runner._remember(f"run{i}", {"run_id": f"run{i}"})
        assert len(runner.RUNS) == 3
        assert "run0" not in runner.RUNS
        assert "run9" in runner.RUNS

    def test_oldest_is_evicted_first(self, monkeypatch):
        monkeypatch.setattr(runner, "MAX_RUNS_CACHED", 2)
        for name in ("first", "second", "third"):
            runner._remember(name, {"run_id": name})
        assert list(runner.RUNS) == ["second", "third"]


@pytest.mark.asyncio
class TestExecuteRun:
    async def test_full_run_produces_one_cell_per_test_model_pair(self, mock_mode):
        suite = make_suite(n_tests=3)
        run = await runner.run_sync(suite, ["openai/gpt-4o-mini", "openai/gpt-4o"])
        assert run["status"] == "done"
        assert len(run["results"]) == 6
        assert {r["model"] for r in run["results"]} == {"openai/gpt-4o-mini", "openai/gpt-4o"}

    async def test_every_cell_has_a_verdict_and_cost(self, mock_mode):
        run = await runner.run_sync(make_suite(2), ["openai/gpt-4o-mini"])
        for r in run["results"]:
            assert isinstance(r["pass"], bool)
            assert r["cost_usd"] >= 0
            assert r["reason"]

    async def test_a_crashing_cell_becomes_an_error_cell(self, monkeypatch, mock_mode):
        calls = {"n": 0}
        real = runner._run_one

        async def flaky(sem, prompt, test, model):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("cell exploded")
            return await real(sem, prompt, test, model)

        monkeypatch.setattr(runner, "_run_one", flaky)
        run = await runner.run_sync(make_suite(3), ["openai/gpt-4o-mini"])
        # The run still completes, with the crash reported as a failing cell.
        assert run["status"] == "done"
        assert len(run["results"]) == 3
        errors = [r for r in run["results"] if r["status"] == "error"]
        assert len(errors) == 1
        assert "cell exploded" in errors[0]["reason"]

    async def test_failures_trigger_a_suggested_fix(self, monkeypatch, mock_mode):
        async def always_fail(*a, **k):
            return {"pass": False, "confidence": 0.9, "reason": "nope",
                    "judge_cost_usd": 0.0}

        monkeypatch.setattr(runner.judge, "judge", always_fail)
        run = await runner.run_sync(make_suite(2), ["openai/gpt-4o-mini"])
        assert run["suggested_prompt"]

    async def test_a_fully_passing_run_suggests_no_fix(self, monkeypatch, mock_mode):
        async def always_pass(*a, **k):
            return {"pass": True, "confidence": 0.9, "reason": "good",
                    "judge_cost_usd": 0.0}

        monkeypatch.setattr(runner.judge, "judge", always_pass)
        run = await runner.run_sync(make_suite(2), ["openai/gpt-4o-mini"])
        assert run["suggested_prompt"] is None

    async def test_concurrency_is_capped(self, monkeypatch, mock_mode):
        peak = {"now": 0, "max": 0}
        real_chat = runner.mesh_client.chat

        async def counting_chat(*a, **k):
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
            try:
                return await real_chat(*a, **k)
            finally:
                peak["now"] -= 1

        monkeypatch.setattr(runner.mesh_client, "chat", counting_chat)
        await runner.run_sync(make_suite(8), ["openai/gpt-4o-mini"])
        # The judge call happens outside the semaphore, so allow 2x headroom.
        assert peak["max"] <= runner.CONCURRENCY * 2

    async def test_run_is_stored_and_retrievable(self, mock_mode):
        run = await runner.run_sync(make_suite(1), ["openai/gpt-4o-mini"])
        assert runner.RUNS[run["run_id"]] is run
