#!/usr/bin/env python3
"""Regression tests for agentbench CLI colour handling (issue #63).

Submitted to the repo as tests/test_cli_color.py. Pure stdlib + pytest.
"""
import io
import json
import sys
import types

import pytest

from cli import agentbench as cli


RUN = {
    "results": [
        {"model": "m1", "category": "safety", "pass": True, "status": "ok",
         "latency_ms": 12, "cost_usd": 0.001, "reason": "all good"},
    ],
    "models": ["m1"],
    "summary": {
        "recommended": "m1", "reason": "highest pass rate",
        "per_model": [{"model": "m1", "passed": 1, "total": 1}],
    },
}


class FakeApi:
    """#1 replaced the api() function with HttpApi/LocalApi objects."""
    label = "test"

    def __call__(self, path, payload=None):
        return RUN


class FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def render(monkeypatch, env, tty, extra_args=()):
    # #1 replaced the module-level api() helper with HttpApi / LocalApi classes.
    monkeypatch.setattr(cli, "HttpApi", lambda base: FakeApi())
    monkeypatch.setattr(sys, "argv", ["agentbench", "run", "no-such-suite", "--models", "m1", *extra_args])
    monkeypatch.setattr(sys, "stdout", stream := FakeStream(tty))
    monkeypatch.setattr(sys, "stderr", errstream := FakeStream(tty))
    for key in ("NO_COLOR", "FORCE_COLOR", "GITHUB_ACTIONS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return stream.getvalue(), errstream.getvalue(), exc.value.code


def test_no_ansi_when_not_a_tty(monkeypatch):
    out, err, code = render(monkeypatch, {}, tty=False)
    assert "\033[" not in out
    assert code == 0


def test_no_ansi_when_no_color_is_set(monkeypatch):
    out, err, code = render(monkeypatch, {"NO_COLOR": "1"}, tty=True)
    assert "\033[" not in out
    assert code == 0


def test_no_color_flag_disables_even_with_force_color(monkeypatch):
    out, err, code = render(monkeypatch, {"FORCE_COLOR": "1"}, tty=True, extra_args=("--no-color",))
    assert "\033[" not in out
    assert code == 0


def test_force_color_keeps_ansi_in_ci(monkeypatch):
    out, err, code = render(monkeypatch, {"FORCE_COLOR": "1"}, tty=False)
    assert "\033[" in out
    assert code == 0


def test_github_actions_keeps_ansi(monkeypatch):
    out, err, code = render(monkeypatch, {"GITHUB_ACTIONS": "true"}, tty=False)
    assert "\033[" in out
    assert code == 0


def test_ansi_on_a_real_tty(monkeypatch):
    out, err, code = render(monkeypatch, {}, tty=True)
    assert "\033[" in out
    assert code == 0


def test_no_color_beats_force_color(monkeypatch):
    assert cli.color_enabled({"NO_COLOR": "1", "FORCE_COLOR": "1"}, types.SimpleNamespace(isatty=lambda: True)) is False


def test_empty_no_color_is_treated_as_unset(monkeypatch):
    assert cli.color_enabled({"NO_COLOR": ""}, types.SimpleNamespace(isatty=lambda: True)) is True


def test_flag_beats_everything(monkeypatch):
    assert cli.color_enabled({"GITHUB_ACTIONS": "true"}, types.SimpleNamespace(isatty=lambda: True), no_color=True) is False


def test_palette_off_is_all_empty():
    assert cli.palette(False) == ("", "", "", "", "", "")


def test_palette_on_has_escapes():
    assert all("\033[" in value for value in cli.palette(True))


def test_report_text_unchanged_when_colour_off(monkeypatch):
    out, err, code = render(monkeypatch, {"NO_COLOR": "1"}, tty=True)
    assert "agentbench" in err  # status line moved to stderr in #1
    assert "PASS" in out
    assert "SHIP -> m1" in out
    assert "all good" in out
    assert code == 0


def test_json_output_stays_machine_readable(monkeypatch):
    out, err, code = render(monkeypatch, {"NO_COLOR": "1"}, tty=True, extra_args=("--json",))
    # The status line is now on stderr, so stdout is pure JSON.
    assert json.loads(out)["summary"]["recommended"] == "m1"
    assert code == 0

def test_status_line_on_stderr_has_no_ansi_when_not_a_tty(monkeypatch):
    out, err, code = render(monkeypatch, {}, tty=False)
    assert "agentbench" in err
    assert "\033[" not in out and "\033[" not in err
    assert code == 0


def test_status_line_on_stderr_honours_no_color(monkeypatch):
    out, err, code = render(monkeypatch, {"GITHUB_ACTIONS": "true"}, tty=True,
                            extra_args=("--no-color",))
    assert "agentbench" in err
    assert "\033[" not in out and "\033[" not in err
    assert code == 0
