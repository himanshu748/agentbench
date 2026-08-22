"""HTTP surface: validation, abuse guards, streaming, badge."""
import json

import pytest
import yaml

from backend import judge, main, mesh_client


def sse_events(text: str):
    """Parse an SSE body into [(event, data-dict), ...]."""
    out = []
    for block in text.strip().split("\n\n"):
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name:
            out.append((name, payload))
    return out


class TestMeta:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["suites"] >= 7

    def test_meta_exposes_catalog_and_categories(self, client):
        body = client.get("/api/meta").json()
        assert body["mock"] is True
        assert set(body["categories"]) == set(judge.CATEGORIES)
        assert "auto" in body["models"]

    def test_landing_and_console_render(self, client):
        assert client.get("/").status_code == 200
        assert client.get("/app").status_code == 200
        assert client.get("/r/anything").status_code == 200


class TestSuiteListing:
    def test_builtin_suites_are_listed(self, client):
        suites = client.get("/api/suites").json()
        ids = {s["suite_id"] for s in suites}
        assert "customer-support" in ids
        assert all(s["num_tests"] > 0 for s in suites)

    def test_get_suite_returns_tests(self, client):
        s = client.get("/api/suites/customer-support").json()
        assert s["system_prompt"] and s["tests"]

    def test_unknown_suite_is_404(self, client):
        assert client.get("/api/suites/nope").status_code == 404


class TestSuiteValidation:
    def post(self, client, doc):
        text = doc if isinstance(doc, str) else yaml.safe_dump(doc)
        return client.post("/api/suites", json={"yaml_text": text})

    def test_valid_suite_is_accepted_and_listed(self, client):
        r = self.post(client, {"name": "Mine", "system_prompt": "p",
                               "tests": [{"input": "hi", "category": "conciseness"}]})
        assert r.status_code == 200
        suite_id = r.json()["suite_id"]
        assert suite_id in {s["suite_id"] for s in client.get("/api/suites").json()}

    def test_custom_category_requires_a_rubric(self, client):
        r = self.post(client, {"system_prompt": "p",
                               "tests": [{"input": "hi", "category": "custom"}]})
        assert r.status_code == 400
        assert "rubric" in r.json()["detail"]

    def test_custom_category_with_rubric_is_accepted(self, client):
        r = self.post(client, {"system_prompt": "p",
                               "tests": [{"input": "hi", "category": "custom",
                                          "rubric": "PASS if lowercase."}]})
        assert r.status_code == 200

    def test_unknown_category_is_rejected_with_a_useful_message(self, client):
        r = self.post(client, {"system_prompt": "p",
                               "tests": [{"input": "hi", "category": "vibes"}]})
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "vibes" in detail and "conciseness" in detail

    @pytest.mark.parametrize("doc,expected", [
        ({"tests": [{"input": "a", "category": "conciseness"}]}, "system_prompt"),
        ({"system_prompt": "p"}, "tests"),
        ({"system_prompt": "p", "tests": []}, "tests"),
        ({"system_prompt": "   ", "tests": [{"input": "a", "category": "conciseness"}]},
         "system_prompt"),
        ({"system_prompt": "p", "tests": [{"category": "conciseness"}]}, "input"),
        ({"system_prompt": "p", "tests": [{"input": "a"}]}, "category"),
    ])
    def test_malformed_suites_are_rejected(self, client, doc, expected):
        r = self.post(client, doc)
        assert r.status_code == 400
        assert expected in r.json()["detail"]

    def test_non_mapping_yaml_is_rejected(self, client):
        assert self.post(client, "- just\n- a list\n").status_code == 400

    def test_broken_yaml_is_rejected_not_crashed(self, client):
        r = self.post(client, "name: [unclosed\n")
        assert r.status_code == 400
        assert "YAML" in r.json()["detail"]

    def test_too_many_tests_is_rejected(self, client):
        doc = {"system_prompt": "p",
               "tests": [{"input": f"t{i}", "category": "conciseness"}
                         for i in range(main.MAX_TESTS_PER_SUITE + 1)]}
        r = self.post(client, doc)
        assert r.status_code == 400
        assert "too many tests" in r.json()["detail"]

    def test_oversized_prompt_is_rejected(self, client):
        r = self.post(client, {"system_prompt": "x" * (main.MAX_PROMPT_CHARS + 1),
                               "tests": [{"input": "a", "category": "conciseness"}]})
        assert r.status_code == 400

    def test_oversized_input_is_rejected(self, client):
        r = self.post(client, {"system_prompt": "p",
                               "tests": [{"input": "x" * (main.MAX_INPUT_CHARS + 1),
                                          "category": "conciseness"}]})
        assert r.status_code == 400

    def test_oversized_document_is_rejected_before_parsing(self, client):
        r = client.post("/api/suites", json={"yaml_text": "x" * (main.MAX_YAML_CHARS + 1)})
        assert r.status_code == 400
        assert "too large" in r.json()["detail"]

    def test_yaml_cannot_construct_python_objects(self, client):
        # safe_load only: an object tag must be a parse error, never execution.
        r = self.post(client, "!!python/object/apply:os.system ['echo pwned']\n")
        assert r.status_code == 400

    def test_user_suites_are_evicted_once_capped(self, client, monkeypatch):
        monkeypatch.setattr(main, "MAX_USER_SUITES", 3)
        ids = [self.post(client, {"system_prompt": "p",
                                  "tests": [{"input": "a", "category": "conciseness"}]}
                         ).json()["suite_id"] for _ in range(5)]
        assert len(main.USER_SUITES) == 3
        assert client.get(f"/api/suites/{ids[0]}").status_code == 404
        assert client.get(f"/api/suites/{ids[-1]}").status_code == 200


class TestRunGuards:
    def test_unknown_suite_is_404(self, client):
        r = client.post("/api/runs", json={"suite_id": "nope", "models": ["auto"]})
        assert r.status_code == 404

    def test_unknown_model_is_rejected_by_name(self, client):
        r = client.post("/api/runs",
                        json={"suite_id": "customer-support", "models": ["totally/fake"]})
        assert r.status_code == 400
        assert "totally/fake" in r.json()["detail"]

    def test_empty_model_list_is_rejected(self, client):
        r = client.post("/api/runs", json={"suite_id": "customer-support", "models": []})
        assert r.status_code == 400

    def test_more_than_four_models_is_rejected(self, client):
        models = mesh_client.AVAILABLE_MODELS[:5]
        r = client.post("/api/runs", json={"suite_id": "customer-support", "models": models})
        assert r.status_code == 400

    def test_duplicate_models_are_collapsed(self, client):
        # Four copies of one model is one model, not a rejected 4-model run.
        suite, models = main._validate_run("customer-support", ["auto", "auto", "auto"])
        assert models == ["auto"]

    def test_cell_ceiling_is_enforced(self, client, monkeypatch):
        monkeypatch.setattr(main, "MAX_CELLS_PER_RUN", 4)
        r = client.post("/api/runs", json={"suite_id": "customer-support",
                                           "models": ["auto"], "sync": True})
        assert r.status_code == 400
        assert "too large" in r.json()["detail"]

    def test_rate_limit_returns_429_with_retry_after(self, client, monkeypatch):
        monkeypatch.setattr(main, "RATE_LIMIT", 2)
        body = {"suite_id": "customer-support", "models": ["auto"], "sync": True}
        monkeypatch.setattr(main, "MAX_CELLS_PER_RUN", 0)  # fail fast, after the rate check
        for _ in range(2):
            assert client.post("/api/runs", json=body).status_code == 400
        r = client.post("/api/runs", json=body)
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) > 0

    def test_rate_limit_state_does_not_leak_forever(self, client, monkeypatch):
        monkeypatch.setattr(main, "RATE_WINDOW", 0)  # every window is already expired
        monkeypatch.setattr(main, "MAX_CELLS_PER_RUN", 0)
        for _ in range(5):
            client.post("/api/runs", json={"suite_id": "customer-support",
                                           "models": ["auto"], "sync": True})
        assert len(main._ip_hits) <= 1


class TestRunExecution:
    def test_sync_run_returns_a_full_report(self, client):
        r = client.post("/api/runs", json={"suite_id": "brand-voice",
                                           "models": ["openai/gpt-4o-mini"], "sync": True})
        assert r.status_code == 200
        run = r.json()
        assert run["status"] == "done"
        assert run["summary"]["recommended"] == "openai/gpt-4o-mini"
        assert len(run["results"]) == len(main.SUITES["brand-voice"]["tests"])

    def test_system_prompt_override_is_applied(self, client):
        r = client.post("/api/runs", json={"suite_id": "customer-support",
                                           "models": ["auto"], "sync": True,
                                           "system_prompt": "Overridden prompt."})
        assert r.status_code == 200

    def test_completed_run_is_fetchable(self, client):
        run = client.post("/api/runs", json={"suite_id": "brand-voice",
                                             "models": ["auto"], "sync": True}).json()
        assert client.get(f"/api/runs/{run['run_id']}").json()["run_id"] == run["run_id"]
        assert client.get(f"/api/shared/{run['run_id']}").status_code == 200

    def test_unknown_run_is_404(self, client):
        assert client.get("/api/runs/deadbeef").status_code == 404
        assert client.get("/api/shared/deadbeef").status_code == 404


class TestStreaming:
    def test_stream_emits_start_cells_and_done(self, client):
        with client.stream("GET", "/api/runs/stream",
                           params={"suite_id": "brand-voice",
                                   "models": "openai/gpt-4o-mini"}) as r:
            assert r.status_code == 200
            events = sse_events("".join(r.iter_text()))
        names = [n for n, _ in events]
        assert names[0] == "start"
        assert names[-1] == "done"
        total = events[0][1]["total"]
        assert names.count("cell") == total
        assert len(events[-1][1]["results"]) == total

    def test_stream_completes_even_when_a_cell_raises(self, client, monkeypatch):
        """Regression: a raising worker used to leave the reader awaiting a cell
        that never arrived, hanging the stream until the platform timed it out."""
        from backend import runner
        real = runner._run_one
        calls = {"n": 0}

        async def flaky(sem, prompt, test, model, attempt=0):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise RuntimeError("worker exploded")
            return await real(sem, prompt, test, model, attempt)

        monkeypatch.setattr(runner, "_run_one", flaky)
        with client.stream("GET", "/api/runs/stream",
                           params={"suite_id": "brand-voice",
                                   "models": "openai/gpt-4o-mini"}) as r:
            events = sse_events("".join(r.iter_text()))
        names = [n for n, _ in events]
        assert names[-1] == "done", "stream must finish rather than stall"
        assert names.count("cell") == events[0][1]["total"]
        errors = [d for n, d in events if n == "cell" and d["status"] == "error"]
        assert errors and "worker exploded" in errors[0]["reason"]

    def test_stream_validates_before_streaming(self, client):
        r = client.get("/api/runs/stream", params={"suite_id": "nope", "models": "auto"})
        assert r.status_code == 404


class TestFixRerun:
    def test_fix_rerun_applies_the_new_prompt(self, client):
        r = client.post("/api/fix-rerun", json={"suite_id": "brand-voice",
                                                "models": ["auto"],
                                                "fixed_prompt": "A better prompt."})
        assert r.status_code == 200
        assert r.json()["status"] == "done"

    def test_fix_rerun_rejects_an_empty_prompt(self, client):
        r = client.post("/api/fix-rerun", json={"suite_id": "brand-voice",
                                                "models": ["auto"], "fixed_prompt": "  "})
        assert r.status_code == 400

    def test_fix_rerun_enforces_the_model_allowlist(self, client):
        """Regression: fix-rerun used to skip _validate_run entirely, so any
        model string and any suite size got through the abuse guards."""
        r = client.post("/api/fix-rerun", json={"suite_id": "brand-voice",
                                                "models": ["totally/fake"],
                                                "fixed_prompt": "p"})
        assert r.status_code == 400

    def test_fix_rerun_enforces_the_cell_ceiling(self, client, monkeypatch):
        monkeypatch.setattr(main, "MAX_CELLS_PER_RUN", 1)
        r = client.post("/api/fix-rerun", json={"suite_id": "brand-voice",
                                                "models": ["auto"], "fixed_prompt": "p"})
        assert r.status_code == 400

    def test_fix_rerun_is_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(main, "RATE_LIMIT", 1)
        body = {"suite_id": "brand-voice", "models": ["auto"], "fixed_prompt": "p"}
        client.post("/api/fix-rerun", json=body)
        assert client.post("/api/fix-rerun", json=body).status_code == 429


class TestRedTeam:
    def test_empty_prompt_is_rejected(self, client):
        assert client.post("/api/redteam", json={"system_prompt": " "}).status_code == 400

    def test_oversized_prompt_is_rejected(self, client):
        r = client.post("/api/redteam",
                        json={"system_prompt": "x" * (main.MAX_PROMPT_CHARS + 1)})
        assert r.status_code == 400

    def test_is_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(main, "RATE_LIMIT", 1)
        client.post("/api/redteam", json={"system_prompt": "You are a bot."})
        r = client.post("/api/redteam", json={"system_prompt": "You are a bot."})
        assert r.status_code == 429


class TestBadge:
    def test_default_badge_renders(self, client):
        r = client.get("/api/badge.svg", params={"passed": 5, "total": 5})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/svg+xml")
        assert "5/5 pass" in r.text
        assert "AgentBench" in r.text

    def test_label_is_rendered(self, client):
        r = client.get("/api/badge.svg", params={"label": "Support Bot",
                                                 "passed": 3, "total": 4})
        assert "Support Bot" in r.text
        assert "3/4 pass" in r.text

    @pytest.mark.parametrize("passed,total,color", [
        (5, 5, "#2BE58A"),   # all green
        (3, 5, "#FFB224"),   # partial
        (0, 5, "#FF5D5D"),   # all red
    ])
    def test_color_reflects_outcome(self, client, passed, total, color):
        assert color in client.get("/api/badge.svg",
                                   params={"passed": passed, "total": total}).text

    def test_label_is_xml_escaped(self, client):
        r = client.get("/api/badge.svg",
                       params={"label": "<script>&", "passed": 1, "total": 1})
        assert "<script>" not in r.text
        assert "&lt;script&gt;&amp;" in r.text

    def test_label_is_length_capped(self, client):
        r = client.get("/api/badge.svg", params={"label": "z" * 500, "passed": 1, "total": 1})
        assert "z" * (main.BADGE_MAX_LABEL + 1) not in r.text

    def test_absurd_counts_are_clamped(self, client):
        r = client.get("/api/badge.svg", params={"passed": 10 ** 9, "total": -4})
        assert r.status_code == 200
        assert "0/0 pass" in r.text

    def test_passed_cannot_exceed_total(self, client):
        assert "3/3 pass" in client.get("/api/badge.svg",
                                        params={"passed": 99, "total": 3}).text

    def test_empty_label_falls_back_to_brand(self, client):
        assert "AgentBench" in client.get("/api/badge.svg", params={"label": "  "}).text


class TestEstimate:
    def test_estimate_returns_cost_and_cells(self, client):
        r = client.post("/api/estimate", json={"suite_id": "customer-support",
                                               "models": ["openai/gpt-4o-mini"], "repeat": 1})
        assert r.status_code == 200
        body = r.json()
        assert body["cells"] == 8
        assert body["calls"] == 16  # model + judge per cell
        assert body["est_cost_usd"] > 0

    def test_estimate_scales_with_repeat(self, client):
        one = client.post("/api/estimate", json={"suite_id": "customer-support",
                                                 "models": ["auto"], "repeat": 1}).json()
        three = client.post("/api/estimate", json={"suite_id": "customer-support",
                                                   "models": ["auto"], "repeat": 3}).json()
        assert three["cells"] == one["cells"] * 3
        assert three["est_cost_usd"] == pytest.approx(one["est_cost_usd"] * 3, rel=1e-6)

    def test_estimate_rejects_unknown_suite(self, client):
        assert client.post("/api/estimate",
                           json={"suite_id": "nope", "models": ["auto"]}).status_code == 404

    def test_estimate_spends_nothing(self, client):
        """A pre-flight estimate must never create a run or touch the cache."""
        from backend import runner
        before = len(runner.RUNS)
        client.post("/api/estimate", json={"suite_id": "customer-support", "models": ["auto"]})
        assert len(runner.RUNS) == before


class TestRepeat:
    def test_repeat_produces_flakiness_and_aggregated_cells(self, client):
        r = client.post("/api/runs", json={"suite_id": "brand-voice", "models": ["auto"],
                                           "sync": True, "repeat": 3})
        run = r.json()
        assert run["repeat"] == 3
        assert len(run["results"]) == 5  # aggregated, not 15
        assert run["flakiness"]["measured"] is True
        assert all(c["runs"] == 3 for c in run["results"])

    def test_repeat_counts_toward_the_cell_ceiling(self, client, monkeypatch):
        monkeypatch.setattr(main, "MAX_CELLS_PER_RUN", 10)
        # brand-voice has 5 tests; x1 model x3 repeat = 15 > 10.
        r = client.post("/api/runs", json={"suite_id": "brand-voice", "models": ["auto"],
                                           "sync": True, "repeat": 3})
        assert r.status_code == 400
        assert "too large" in r.json()["detail"]

    def test_repeat_over_max_is_rejected(self, client):
        r = client.post("/api/runs", json={"suite_id": "brand-voice", "models": ["auto"],
                                           "sync": True, "repeat": main.MAX_REPEAT + 1})
        assert r.status_code == 400
        assert "repeat" in r.json()["detail"]

    def test_repeat_zero_is_rejected(self, client):
        r = client.post("/api/runs", json={"suite_id": "brand-voice", "models": ["auto"],
                                           "sync": True, "repeat": 0})
        assert r.status_code == 400
