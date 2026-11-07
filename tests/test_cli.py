import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_cli_local_mode_runs_suite_yaml():
    env = {**os.environ, "OPENROUTER_API_KEY": ""}
    p = subprocess.run([sys.executable, "cli/agentbench.py", "run", "suites/customer-support.yaml",
                        "--models", "openai/gpt-4o-mini,anthropic/claude-haiku-4.5", "--local"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode in (0, 1), p.stderr
    assert "SHIP ->" in p.stdout and "MOCK MODE" in p.stdout


def test_cli_reports_unreachable_server():
    env = {**os.environ, "AGENTBENCH_URL": "http://127.0.0.1:9"}
    p = subprocess.run([sys.executable, "cli/agentbench.py", "run", "customer-support",
                        "--models", "auto"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode != 0 and "cannot reach AgentBench" in p.stderr
