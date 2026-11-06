import json

import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def test_meta_and_health(client):
    meta = client.get("/api/meta").json()
    assert meta["mock"] is True
    assert meta["provider"] == "openrouter"
    assert "auto" in meta["models"]
    assert "pii_refusal" in meta["categories"]
    assert client.get("/api/health").json() == {"ok": True, "mock": True}


def test_builtin_suites_load(client):
    suites = client.get("/api/suites").json()
    assert len(suites) == 7
    assert all(s["num_tests"] > 0 for s in suites)


@pytest.mark.parametrize("yaml_text,err", [
    ("tests: []\nsystem_prompt: x", "missing tests"),
    ("tests:\n  - input: hi\n    category: conciseness", "missing system_prompt"),
    ("system_prompt: x\ntests:\n  - input: hi\n    category: custom", "rubric"),
    ("system_prompt: x\ntests:\n  - input: hi\n    category: nope", "unknown category"),
])
def test_create_suite_validation(client, yaml_text, err):
    r = client.post("/api/suites", json={"yaml_text": yaml_text})
    assert r.status_code == 400 and err in r.json()["detail"]


def test_sync_run_and_shared_report(client):
    r = client.post("/api/runs", json={"suite_id": "customer-support",
                                       "models": ["openai/gpt-4o-mini", "auto"]})
    assert r.status_code == 200
    run = r.json()
    assert run["status"] == "done" and run["mock"] is True
    assert len(run["results"]) == 2 * len(main.SUITES["customer-support"]["tests"])
    assert run["summary"]["recommended"] in ("openai/gpt-4o-mini", "auto")
    shared = client.get(f"/api/shared/{run['run_id']}")
    assert shared.status_code == 200 and shared.json()["run_id"] == run["run_id"]


def test_shared_report_survives_restart(client):
    run = client.post("/api/runs", json={"suite_id": "hr-policy",
                                         "models": ["openai/gpt-4o-mini"]}).json()
    main.runner.RUNS.clear()  # simulate a restart: only SQLite has it now
    assert client.get(f"/api/shared/{run['run_id']}").json()["suite_name"] == run["suite_name"]


def test_run_rejects_unknown_and_too_many_models(client):
    r = client.post("/api/runs", json={"suite_id": "customer-support", "models": ["not/a-model"]})
    assert r.status_code == 400 and "not/a-model" in r.json()["detail"]
    too_many = main.llm_client.AVAILABLE_MODELS[:5]
    r = client.post("/api/runs", json={"suite_id": "customer-support", "models": too_many})
    assert r.status_code == 400


def test_stream_emits_start_cells_done(client):
    with client.stream("GET", "/api/runs/stream",
                       params={"suite_id": "coding-assistant", "models": "openai/gpt-4o-mini"}) as r:
        body = "".join(r.iter_text())
    events = [blk.split("\n")[0].removeprefix("event: ") for blk in body.strip().split("\n\n")]
    n = len(main.SUITES["coding-assistant"]["tests"])
    assert events[0] == "start" and events[-1] == "done" and events.count("cell") == n
    done = json.loads(body.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert len(done["results"]) == n


def test_fix_rerun_validates_models(client):
    r = client.post("/api/fix-rerun", json={"suite_id": "customer-support",
                                            "models": ["not/a-model"], "fixed_prompt": "x"})
    assert r.status_code == 400


def test_badge(client):
    r = client.get("/api/badge.svg", params={"passed": 5, "total": 5})
    assert r.headers["content-type"].startswith("image/svg+xml") and "5/5 pass" in r.text


def test_pages(client):
    for path in ("/", "/app", "/r/abc123", "/static/logo.svg"):
        assert client.get(path).status_code == 200
