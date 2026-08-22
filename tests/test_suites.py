"""The shipped suite YAMLs are product surface: validate them like user input."""
import pathlib

import pytest
import yaml

from backend import judge, main

SUITE_DIR = pathlib.Path(__file__).resolve().parent.parent / "suites"
SUITE_FILES = sorted(SUITE_DIR.glob("*.yaml"))


def test_suites_directory_is_not_empty():
    assert SUITE_FILES, "no suite YAMLs found to validate"


@pytest.mark.parametrize("path", SUITE_FILES, ids=lambda p: p.stem)
class TestShippedSuite:
    def test_parses_as_yaml(self, path):
        assert isinstance(yaml.safe_load(path.read_text()), dict)

    def test_passes_the_same_validation_as_an_upload(self, path):
        """A built-in that would be rejected from /api/suites is a broken build."""
        main._validate_suite(yaml.safe_load(path.read_text()))

    def test_has_a_name(self, path):
        assert yaml.safe_load(path.read_text()).get("name")

    def test_fits_under_the_run_cell_ceiling(self, path):
        """Every built-in must be runnable against at least one model."""
        n = len(yaml.safe_load(path.read_text())["tests"])
        assert n <= main.MAX_CELLS_PER_RUN

    def test_custom_tests_carry_a_rubric(self, path):
        for t in yaml.safe_load(path.read_text())["tests"]:
            if t["category"] == "custom":
                assert t.get("rubric", "").strip()

    def test_no_duplicate_inputs(self, path):
        inputs = [t["input"] for t in yaml.safe_load(path.read_text())["tests"]]
        assert len(inputs) == len(set(inputs)), "duplicate test inputs waste a run"


def test_every_builtin_category_is_exercised_somewhere():
    """The README promises the built-ins cover every category — hold it to that."""
    covered = set()
    for path in SUITE_FILES:
        for t in yaml.safe_load(path.read_text())["tests"]:
            covered.add(t["category"])
    missing = set(judge.CATEGORIES) - covered
    assert not missing, f"categories never exercised by any built-in suite: {missing}"


def test_all_suites_load_into_the_registry():
    loaded = {s["suite_id"] for s in main.BUILTIN_SUITES.values()}
    assert loaded == {p.stem for p in SUITE_FILES}


def test_custom_rubrics_are_demonstrated():
    """The brand-voice suite is the documented showcase for custom rubrics."""
    tests = main.BUILTIN_SUITES["brand-voice"]["tests"]
    assert any(t["category"] == "custom" for t in tests)


def test_suite_count_is_advertised():
    """The product claims a certain number of built-in suites; hold it to that."""
    assert len(SUITE_FILES) >= 15


def test_category_count_is_advertised():
    assert len(judge.CATEGORIES) == 19          # 16 judged + 3 tool-calling
    assert len(judge.TOOL_CATEGORIES) == 3


def test_new_domain_suites_are_present():
    ids = {p.stem for p in SUITE_FILES}
    for expected in ("rag-grounding", "legal-intake", "fintech-support", "devops-runbook",
                     "education-tutor", "travel-concierge", "safety-guardrails",
                     "data-extraction"):
        assert expected in ids, f"missing suite: {expected}"
