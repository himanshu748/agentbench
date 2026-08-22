"""CLI behaviour: exit codes, error reporting, CI integration."""
import json
import urllib.error
import io

import pytest

from cli import agentbench as cli


def make_run(models=("m1",), passed_per_model=None, suggested=None):
    passed_per_model = passed_per_model or {"m1": (2, 2)}
    results, per_model = [], []
    for m in models:
        p, t = passed_per_model[m]
        for i in range(t):
            results.append({"input": f"i{i}", "category": "conciseness", "model": m,
                            "pass": i < p, "status": "done", "latency_ms": 100,
                            "cost_usd": 0.001, "reason": "because"})
        per_model.append({"model": m, "passed": p, "total": t,
                          "pass_rate": p / t if t else 0,
                          "total_cost_usd": 0.002, "avg_latency_ms": 100})
    best = max(per_model, key=lambda s: s["pass_rate"])
    return {"run_id": "abc123", "status": "done", "suite_name": "Suite",
            "models": list(models), "results": results,
            "suggested_prompt": suggested,
            "summary": {"per_model": per_model, "recommended": best["model"],
                        "reason": "because"}}


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the HTTP layer with a scripted responder; record every call."""
    def _install(responses):
        calls = []

        def api(base, path, payload=None, timeout=180.0):
            calls.append((path, payload))
            value = responses[path]
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(cli, "api", api)
        return calls
    return _install


def run_cli(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["agentbench", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


class TestExitCodes:
    def test_all_passing_exits_zero(self, monkeypatch, fake_api, capsys):
        fake_api({"/api/runs": make_run(passed_per_model={"m1": (2, 2)})})
        assert run_cli(["run", "some-suite", "--models", "m1"], monkeypatch) == cli.EXIT_OK
        assert "SHIP" in capsys.readouterr().out

    def test_a_failing_test_exits_one(self, monkeypatch, fake_api, capsys):
        fake_api({"/api/runs": make_run(passed_per_model={"m1": (1, 2)})})
        code = run_cli(["run", "some-suite", "--models", "m1"], monkeypatch)
        assert code == cli.EXIT_TESTS_FAILED
        assert "BLOCK" in capsys.readouterr().out

    def test_server_error_exits_two_not_one(self, monkeypatch, fake_api, capsys):
        """CI needs to tell 'the prompt regressed' apart from 'the tool broke'."""
        fake_api({"/api/runs": cli.ApiError("boom")})
        code = run_cli(["run", "some-suite", "--models", "m1"], monkeypatch)
        assert code == cli.EXIT_ERROR
        assert "boom" in capsys.readouterr().err

    def test_run_reported_as_errored_exits_two(self, monkeypatch, fake_api):
        fake_api({"/api/runs": {"status": "error", "error": "upstream died"}})
        assert run_cli(["run", "s", "--models", "m1"], monkeypatch) == cli.EXIT_ERROR

    def test_empty_results_exit_two_instead_of_crashing(self, monkeypatch, fake_api):
        """Regression: an empty result set used to raise inside the table render."""
        fake_api({"/api/runs": {"status": "done", "models": [], "results": [],
                                "summary": None}})
        assert run_cli(["run", "s", "--models", "m1"], monkeypatch) == cli.EXIT_ERROR

    def test_empty_models_argument_exits_two(self, monkeypatch, fake_api):
        fake_api({})
        assert run_cli(["run", "s", "--models", " , "], monkeypatch) == cli.EXIT_ERROR


class TestPassThreshold:
    def test_threshold_allows_partial_failure(self, monkeypatch, fake_api):
        fake_api({"/api/runs": make_run(passed_per_model={"m1": (8, 10)})})
        code = run_cli(["run", "s", "--models", "m1", "--min-pass-rate", "0.8"], monkeypatch)
        assert code == cli.EXIT_OK

    def test_below_threshold_still_fails(self, monkeypatch, fake_api):
        fake_api({"/api/runs": make_run(passed_per_model={"m1": (7, 10)})})
        code = run_cli(["run", "s", "--models", "m1", "--min-pass-rate", "0.8"], monkeypatch)
        assert code == cli.EXIT_TESTS_FAILED

    def test_default_requires_everything_to_pass(self, monkeypatch, fake_api):
        fake_api({"/api/runs": make_run(passed_per_model={"m1": (9, 10)})})
        assert run_cli(["run", "s", "--models", "m1"], monkeypatch) == cli.EXIT_TESTS_FAILED

    @pytest.mark.parametrize("bad", ["1.5", "-0.2"])
    def test_out_of_range_threshold_is_rejected(self, monkeypatch, fake_api, bad):
        fake_api({})
        assert run_cli(["run", "s", "--models", "m1", "--min-pass-rate", bad],
                       monkeypatch) == 2


class TestSuiteResolution:
    def test_existing_file_is_uploaded(self, monkeypatch, fake_api, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text("name: x\nsystem_prompt: p\ntests: []\n")
        calls = fake_api({"/api/suites": {"suite_id": "custom-1"},
                          "/api/runs": make_run()})
        run_cli(["run", str(path), "--models", "m1"], monkeypatch)
        assert calls[0][0] == "/api/suites"
        assert calls[1][1]["suite_id"] == "custom-1"

    def test_bare_id_is_used_directly(self, monkeypatch, fake_api):
        calls = fake_api({"/api/runs": make_run()})
        run_cli(["run", "customer-support", "--models", "m1"], monkeypatch)
        assert calls[0][0] == "/api/runs"
        assert calls[0][1]["suite_id"] == "customer-support"

    def test_missing_yaml_file_is_a_clear_error(self, monkeypatch, fake_api, capsys):
        fake_api({})
        code = run_cli(["run", "no/such/file.yaml", "--models", "m1"], monkeypatch)
        assert code == cli.EXIT_ERROR
        assert "not found" in capsys.readouterr().err


class TestApiErrors:
    def _urlopen(self, monkeypatch, raiser):
        monkeypatch.setattr(cli.urllib.request, "urlopen", raiser)

    def test_http_error_detail_is_surfaced(self, monkeypatch):
        def boom(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(json.dumps({"detail": "invalid suite: missing tests"}).encode()))
        self._urlopen(monkeypatch, boom)
        with pytest.raises(cli.ApiError, match="invalid suite: missing tests"):
            cli.api("http://x", "/api/suites", {"yaml_text": ""})

    def test_pydantic_error_list_is_flattened(self, monkeypatch):
        def boom(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 422, "Unprocessable", {},
                io.BytesIO(json.dumps({"detail": [{"msg": "field required"}]}).encode()))
        self._urlopen(monkeypatch, boom)
        with pytest.raises(cli.ApiError, match="field required"):
            cli.api("http://x", "/api/runs", {})

    def test_connection_failure_suggests_a_fix(self, monkeypatch):
        def boom(req, timeout=None):
            raise urllib.error.URLError("Connection refused")
        self._urlopen(monkeypatch, boom)
        with pytest.raises(cli.ApiError, match="AGENTBENCH_URL"):
            cli.api("http://x", "/api/meta")

    def test_non_json_response_is_reported(self, monkeypatch):
        class R:
            def read(self): return b"<html>oops</html>"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda req, timeout=None: R())
        with pytest.raises(cli.ApiError, match="non-JSON"):
            cli.api("http://x", "/api/meta")


class TestOutput:
    def test_table_survives_a_missing_reason(self, capsys):
        run = make_run()
        run["results"][0].pop("reason")
        cli.print_table(run)
        assert "conciseness" in capsys.readouterr().out

    def test_table_handles_zero_results(self, capsys):
        cli.print_table({"models": [], "results": []})
        assert capsys.readouterr().out == ""

    def test_json_mode_emits_parseable_json(self, monkeypatch, fake_api, capsys):
        fake_api({"/api/runs": make_run()})
        run_cli(["run", "s", "--models", "m1", "--json"], monkeypatch)
        assert json.loads(capsys.readouterr().out)["run_id"] == "abc123"

    def test_report_url_is_printed(self, monkeypatch, fake_api, capsys):
        fake_api({"/api/runs": make_run()})
        run_cli(["run", "s", "--models", "m1"], monkeypatch)
        assert "/r/abc123" in capsys.readouterr().out


class TestStepSummary:
    def test_summary_is_written_when_github_sets_the_path(self, monkeypatch, fake_api,
                                                          tmp_path):
        target = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        fake_api({"/api/runs": make_run(passed_per_model={"m1": (1, 2)})})
        run_cli(["run", "s", "--models", "m1"], monkeypatch)
        text = target.read_text()
        assert "AgentBench" in text
        assert "| `m1` | 1/2 |" in text
        assert "Failing tests" in text

    def test_passing_run_marks_success(self, monkeypatch, fake_api, tmp_path):
        target = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        fake_api({"/api/runs": make_run()})
        run_cli(["run", "s", "--models", "m1"], monkeypatch)
        assert "Passed" in target.read_text()

    def test_no_summary_written_outside_github(self, monkeypatch, fake_api, tmp_path):
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        fake_api({"/api/runs": make_run()})
        run_cli(["run", "s", "--models", "m1"], monkeypatch)  # must not raise

    def test_unwritable_summary_never_fails_the_build(self, monkeypatch, fake_api, tmp_path):
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "no" / "such" / "dir.md"))
        fake_api({"/api/runs": make_run()})
        assert run_cli(["run", "s", "--models", "m1"], monkeypatch) == cli.EXIT_OK

    def test_pipes_in_a_reason_do_not_break_the_table(self, monkeypatch, fake_api, tmp_path):
        target = tmp_path / "s.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        run = make_run(passed_per_model={"m1": (0, 1)})
        run["results"][0]["reason"] = "a | b | c"
        fake_api({"/api/runs": run})
        run_cli(["run", "s", "--models", "m1"], monkeypatch)
        assert r"a \| b \| c" in target.read_text()


class TestServerSelection:
    def test_url_flag_wins_over_env(self, monkeypatch, fake_api):
        monkeypatch.setenv("AGENTBENCH_URL", "http://from-env")
        seen = {}

        def api(base, path, payload=None, timeout=180.0):
            seen["base"] = base
            return make_run()

        monkeypatch.setattr(cli, "api", api)
        run_cli(["--url", "http://from-flag", "run", "s", "--models", "m1"], monkeypatch)
        assert seen["base"] == "http://from-flag"

    def test_env_is_used_when_no_flag(self, monkeypatch):
        monkeypatch.setenv("AGENTBENCH_URL", "http://from-env/")
        seen = {}

        def api(base, path, payload=None, timeout=180.0):
            seen["base"] = base
            return make_run()

        monkeypatch.setattr(cli, "api", api)
        run_cli(["run", "s", "--models", "m1"], monkeypatch)
        assert seen["base"] == "http://from-env"  # trailing slash stripped


class TestSubcommands:
    def test_suites_lists_ids(self, monkeypatch, fake_api, capsys):
        fake_api({"/api/suites": [{"suite_id": "customer-support", "name": "CS",
                                   "num_tests": 8}]})
        assert run_cli(["suites"], monkeypatch) == cli.EXIT_OK
        assert "customer-support" in capsys.readouterr().out

    def test_models_lists_the_catalog(self, monkeypatch, fake_api, capsys):
        fake_api({"/api/meta": {"models": ["auto", "openai/gpt-4o"],
                                "judge_model": "j", "model_count": 500, "chat_count": 400}})
        assert run_cli(["models"], monkeypatch) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "openai/gpt-4o" in out and "500" in out


class TestColor:
    def test_no_color_env_disables_escapes(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert cli._supports_color(io.StringIO()) is False

    def test_non_tty_gets_no_escapes(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        assert cli._supports_color(io.StringIO()) is False
