"""Regression tests for mock-mode behavior categories and judge verdicts."""
import asyncio

import pytest

from backend import judge, llm_client, runner


@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    monkeypatch.setattr(llm_client, "MOCK", True)


def test_every_builtin_category_has_a_canned_pair():
    assert set(llm_client._MOCK_BEHAVIOR) == set(judge.CATEGORIES)


@pytest.mark.parametrize("category", judge.CATEGORIES)
def test_mock_judge_passes_good_and_fails_bad(category):
    good, bad = llm_client._MOCK_BEHAVIOR[category]
    passed = asyncio.run(judge.judge(category, "test input", good))
    failed = asyncio.run(judge.judge(category, "test input", bad))
    assert passed["pass"] is True, (category, passed["reason"])
    assert failed["pass"] is False, (category, failed["reason"])


@pytest.mark.parametrize(
    "category", ["format_compliance", "language_match", "pii_refusal"]
)
def test_mock_test_execution_uses_a_canned_answer(category):
    tagged = runner._mock_tag({"input": "test input", "category": category})
    result = asyncio.run(
        llm_client.chat("test-model", [{"role": "user", "content": tagged}],
                         label=f"test:{category}")
    )
    good, bad = llm_client._MOCK_BEHAVIOR[category]
    assert result["content"] in (good, bad), (
        category, result["content"]
    )
