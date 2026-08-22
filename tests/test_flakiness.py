"""Flakiness detection: repeated runs, aggregation, and the summary."""
import pytest

from backend import runner
from tests.conftest import make_suite


def attempt(model, passed, cost=0.001, latency=100, category="conciseness",
            status="done", reason="r"):
    return {"input": "i", "category": category, "model": model, "pass": passed,
            "cost_usd": cost, "latency_ms": latency, "status": status, "reason": reason,
            "confidence": 0.9}


class TestAggregateRepeats:
    def test_all_pass_is_not_flaky(self):
        agg = runner._aggregate_repeats([attempt("m", True)] * 4)
        assert agg["pass"] is True
        assert agg["flaky"] is False
        assert (agg["passes"], agg["runs"], agg["pass_rate"]) == (4, 4, 1.0)

    def test_all_fail_is_not_flaky(self):
        agg = runner._aggregate_repeats([attempt("m", False)] * 3)
        assert agg["pass"] is False
        assert agg["flaky"] is False
        assert agg["pass_rate"] == 0.0

    def test_mixed_is_flaky(self):
        cells = [attempt("m", True), attempt("m", False), attempt("m", True)]
        agg = runner._aggregate_repeats(cells)
        assert agg["flaky"] is True
        assert agg["passes"] == 2 and agg["runs"] == 3

    def test_majority_decides_headline_verdict(self):
        assert runner._aggregate_repeats(
            [attempt("m", True), attempt("m", True), attempt("m", False)])["pass"] is True
        assert runner._aggregate_repeats(
            [attempt("m", False), attempt("m", False), attempt("m", True)])["pass"] is False

    def test_ties_count_as_pass(self):
        agg = runner._aggregate_repeats([attempt("m", True), attempt("m", False)])
        assert agg["pass"] is True

    def test_cost_is_summed_latency_averaged(self):
        cells = [attempt("m", True, cost=0.01, latency=100),
                 attempt("m", True, cost=0.02, latency=300)]
        agg = runner._aggregate_repeats(cells)
        assert agg["cost_usd"] == pytest.approx(0.03)
        assert agg["latency_ms"] == 200

    def test_reason_explains_the_losing_side(self):
        # A flaky cell whose majority is pass should surface a failing reason.
        cells = [attempt("m", True, reason="ok"), attempt("m", True, reason="ok"),
                 attempt("m", False, reason="leaked the key")]
        assert runner._aggregate_repeats(cells)["reason"] == "leaked the key"

    def test_attempts_are_recorded(self):
        agg = runner._aggregate_repeats([attempt("m", True), attempt("m", False)])
        assert len(agg["attempts"]) == 2
        assert {a["pass"] for a in agg["attempts"]} == {True, False}

    def test_any_errored_attempt_marks_cell_errored(self):
        cells = [attempt("m", True), attempt("m", False, status="error")]
        assert runner._aggregate_repeats(cells)["status"] == "error"


class TestFlakinessSummary:
    def test_single_run_is_not_measured(self):
        fk = runner._flakiness_summary([attempt("m", True)])
        assert fk["measured"] is False
        assert fk["repeat"] == 1

    def test_counts_flaky_cells(self):
        results = [
            runner._aggregate_repeats([attempt("m", True)] * 3),          # stable
            runner._aggregate_repeats([attempt("m", True), attempt("m", False),
                                       attempt("m", True)]),               # flaky
        ]
        fk = runner._flakiness_summary(results)
        assert fk["measured"] is True
        assert fk["flaky_count"] == 1
        assert fk["total_cells"] == 2
        assert fk["consistency"] == 0.5

    def test_flaky_cells_sorted_most_uncertain_first(self):
        near = runner._aggregate_repeats([attempt("m", True)] * 1 + [attempt("m", False)] * 3)
        even = runner._aggregate_repeats([attempt("m", True)] * 2 + [attempt("m", False)] * 2)
        fk = runner._flakiness_summary([near, even])
        # The 50/50 cell is the most uncertain and should come first.
        assert fk["flaky_cells"][0]["pass_rate"] == 0.5


@pytest.mark.asyncio
class TestRepeatedRun:
    async def test_repeat_collapses_to_one_cell_per_pair(self, mock_mode):
        run = await runner.run_sync(make_suite(3), ["openai/gpt-4o-mini"], repeat=4)
        assert run["repeat"] == 4
        assert len(run["results"]) == 3  # not 12
        assert all(r["runs"] == 4 for r in run["results"])

    async def test_repeat_one_matches_the_single_run_shape(self, mock_mode):
        run = await runner.run_sync(make_suite(2), ["openai/gpt-4o-mini"], repeat=1)
        assert run["flakiness"]["measured"] is False
        assert "runs" not in run["results"][0]  # unaggregated single cells

    async def test_repeats_actually_vary(self, mock_mode):
        # With a wide-enough suite, mock variation should produce at least one
        # flaky cell across repeats — proving repeats aren't identical.
        run = await runner.run_sync(make_suite(8), ["openai/gpt-4o-mini"], repeat=6)
        rates = {r["pass_rate"] for r in run["results"]}
        assert rates - {0.0, 1.0}, "expected at least one partially-passing cell"

    async def test_flakiness_summary_is_attached(self, mock_mode):
        run = await runner.run_sync(make_suite(4), ["openai/gpt-4o-mini"], repeat=3)
        assert run["flakiness"]["measured"] is True
        assert run["flakiness"]["repeat"] == 3
