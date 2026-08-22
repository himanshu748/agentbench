"""Shared fixtures. Every test runs in MOCK mode: no MeshAPI key, no network."""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

os.environ.pop("MESH_API_KEY", None)  # force mock mode before backend import

from fastapi.testclient import TestClient  # noqa: E402

from backend import main, mesh_client, runner  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    """Each test starts with empty rate-limit, run and user-suite state."""
    main._ip_hits.clear()
    main.USER_SUITES.clear()
    runner.RUNS.clear()
    yield
    main._ip_hits.clear()
    main.USER_SUITES.clear()
    runner.RUNS.clear()


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def mock_mode():
    assert mesh_client.MOCK, "tests must run without MESH_API_KEY set"
    return True


def make_suite(n_tests=2, category="requires_clarification"):
    return {"name": "Fixture Suite", "system_prompt": "You are a test bot.",
            "tests": [{"input": f"test input {i}", "category": category}
                      for i in range(n_tests)]}
