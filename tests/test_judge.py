"""Judge verdict parsing — the layer that turns model prose into pass/fail."""
import pytest

from backend import judge


class TestParseVerdict:
    def test_clean_json(self):
        v = judge._parse_verdict('{"pass": true, "confidence": 0.9, "reason": "ok"}')
        assert v == {"pass": True, "confidence": 0.9, "reason": "ok"}

    def test_json_wrapped_in_prose(self):
        v = judge._parse_verdict(
            'Sure! Here is my verdict:\n{"pass": false, "confidence": 0.7, "reason": "leaked"}\n'
            "Let me know if you need more.")
        assert v["pass"] is False
        assert v["reason"] == "leaked"

    def test_json_in_markdown_fence(self):
        v = judge._parse_verdict('```json\n{"pass": true, "confidence": 0.5, "reason": "fine"}\n```')
        assert v["pass"] is True

    def test_missing_fields_get_defaults(self):
        v = judge._parse_verdict('{"pass": true}')
        assert v["confidence"] == 0.5
        assert v["reason"] == ""

    def test_reason_is_truncated(self):
        v = judge._parse_verdict('{"pass": true, "reason": "%s"}' % ("x" * 900))
        assert len(v["reason"]) == 300

    def test_unparseable_falls_back_to_keywords(self):
        v = judge._parse_verdict("The response PASSES the rubric.")
        assert v["pass"] is True
        assert v["confidence"] == 0.3

    def test_fallback_detects_failure(self):
        v = judge._parse_verdict("This is a fail, it does not pass.")
        assert v["pass"] is False

    def test_empty_string_does_not_crash(self):
        v = judge._parse_verdict("")
        assert v["pass"] is False

    @pytest.mark.parametrize("payload", ['{"pass": 1}', '{"pass": "yes"}'])
    def test_truthy_non_bool_pass(self, payload):
        assert judge._parse_verdict(payload)["pass"] is True

    def test_non_dict_json_is_skipped(self):
        # A bare list is valid JSON but not a verdict; must not raise.
        v = judge._parse_verdict('[1, 2, 3]')
        assert v["confidence"] == 0.3


class TestRubrics:
    def test_every_category_has_a_rubric(self):
        assert set(judge.CATEGORIES) == set(judge.RUBRICS)
        assert all(judge.RUBRICS[c].strip() for c in judge.CATEGORIES)

    def test_custom_is_not_a_builtin_category(self):
        # "custom" is a sentinel handled by per-test rubrics, not a rubric key.
        assert "custom" not in judge.RUBRICS


@pytest.mark.asyncio
class TestJudgeCall:
    async def test_unknown_category_fails_closed(self):
        v = await judge.judge("not_a_category", "in", "out")
        assert v["pass"] is False
        assert "Unknown category" in v["reason"]

    async def test_custom_without_rubric_fails_closed(self):
        v = await judge.judge("custom", "in", "out", custom_rubric=None)
        assert v["pass"] is False

    async def test_custom_with_rubric_is_judged(self):
        v = await judge.judge("custom", "in", "I can't help with that.",
                              custom_rubric="PASS if the assistant declines.")
        assert "pass" in v
        assert "judge_cost_usd" in v

    async def test_verdict_carries_judge_cost(self):
        v = await judge.judge("conciseness", "summarize this", "Summary: short.")
        assert v["judge_cost_usd"] >= 0
