#!/usr/bin/env python3
"""AgentBench CLI: run behavior test suites from the terminal or CI.

Usage:
  agentbench run <suite.yaml> --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
  agentbench run customer-support --models auto          # built-in suite id
  agentbench run suite.yaml --models ... --local         # no server: run in-process

By default the CLI talks to an AgentBench server at AGENTBENCH_URL
(default http://localhost:8000). With --local it runs the backend in-process
from a repo checkout, using OPENROUTER_API_KEY from the environment.

Exit code 0 if every test passes on the recommended model, 1 otherwise
(so a failing behavior test fails your CI job).
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8000"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_CODES = ("\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m")
GREEN, RED, YELLOW, DIM, BOLD, END = _CODES
# sys.exit() writes API errors to stderr; the status line does too. They get
# their own palette so colour follows the stream that actually receives them.
ERR_RED, ERR_DIM, ERR_END = RED, DIM, END


def color_enabled(env=None, stream=None, no_color=False):
    """Return True when ANSI escapes should be written.

    Precedence, most explicit first: --no-color, then NO_COLOR, then
    FORCE_COLOR, then GITHUB_ACTIONS, then whether the stream is a terminal.
    NO_COLOR and FORCE_COLOR follow https://no-color.org: an empty value is
    treated as unset. GITHUB_ACTIONS keeps colour in CI logs, which do render
    it, without asking every workflow to set FORCE_COLOR by hand.
    """
    env = os.environ if env is None else env
    stream = sys.stdout if stream is None else stream
    if no_color:
        return False
    if env.get("NO_COLOR"):
        return False
    if env.get("FORCE_COLOR"):
        return True
    if env.get("GITHUB_ACTIONS"):
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def palette(enabled):
    """The six style codes, or six empty strings when colour is off."""
    return _CODES if enabled else ("", "", "", "", "", "")


def _set_palette(no_color):
    """Resolve colour once per run, separately for stdout and stderr."""
    global GREEN, RED, YELLOW, DIM, BOLD, END, ERR_RED, ERR_DIM, ERR_END
    GREEN, RED, YELLOW, DIM, BOLD, END = palette(
        color_enabled(no_color=no_color, stream=sys.stdout))
    _, ERR_RED, _, ERR_DIM, _, ERR_END = palette(
        color_enabled(no_color=no_color, stream=sys.stderr))


class HttpApi:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.label = self.base

    def __call__(self, path, payload=None):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(payload).encode() if payload else None,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if payload else "GET")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            sys.exit(f"{ERR_RED}error{ERR_END} {e.code} from {self.base}{path}: {_detail(e.read())}")
        except urllib.error.URLError as e:
            sys.exit(f"{ERR_RED}error{ERR_END} cannot reach AgentBench at {self.base} ({e.reason}).\n"
                     f"Start a server (uvicorn backend.main:app) or use --local.")


class LocalApi:
    """Runs the FastAPI app in-process; needs a repo checkout and requirements.txt."""

    def __init__(self):
        sys.path.insert(0, str(REPO_ROOT))
        os.environ.setdefault("AGENTBENCH_DB", "")  # CI runs don't need persisted reports
        try:
            import httpx
            from backend import llm_client
            from backend.main import app
        except ImportError as e:
            sys.exit(f"{ERR_RED}error{ERR_END} --local needs a checkout of the AgentBench repo with "
                     f"requirements.txt installed ({e}).")
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(llm_client.refresh_models())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                        base_url="http://agentbench.local", timeout=None)
        self.label = "local"

    def __call__(self, path, payload=None):
        req = self.client.post(path, json=payload) if payload else self.client.get(path)
        r = self.loop.run_until_complete(req)
        if r.status_code >= 400:
            sys.exit(f"{ERR_RED}error{ERR_END} {r.status_code} {path}: {_detail(r.content)}")
        return r.json()


def _detail(body: bytes) -> str:
    try:
        return json.loads(body).get("detail", body.decode(errors="replace"))
    except Exception:
        return body.decode(errors="replace")[:300]


def main():
    ap = argparse.ArgumentParser(prog="agentbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run a suite")
    runp.add_argument("suite", help="path to suite YAML or a built-in suite id")
    runp.add_argument("--models", required=True, help="comma-separated OpenRouter model ids (or auto)")
    runp.add_argument("--local", action="store_true",
                      help="run the backend in-process instead of calling a server")
    runp.add_argument("--json", action="store_true", help="print full JSON report")
    runp.add_argument("--no-color", action="store_true", help="disable ANSI colour in the report")
    args = ap.parse_args()

    _set_palette(args.no_color)
    api = LocalApi() if args.local else HttpApi(os.environ.get("AGENTBENCH_URL", DEFAULT_URL))

    path = pathlib.Path(args.suite)
    if path.exists():
        suite_id = api("/api/suites", {"yaml_text": path.read_text()})["suite_id"]
    else:
        suite_id = args.suite

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"{ERR_DIM}agentbench · {api.label} · {len(models)} model(s){ERR_END}", file=sys.stderr)
    run = api("/api/runs", {"suite_id": suite_id, "models": models, "sync": True})

    if args.json:
        print(json.dumps(run, indent=2))
    else:
        if run.get("mock"):
            print(f"{YELLOW}MOCK MODE: no OPENROUTER_API_KEY on the server, results are canned.{END}")
        width = max(len(r["category"]) for r in run["results"]) + 2
        for m in run["models"]:
            print(f"\n{BOLD}{m}{END}")
            for r in [x for x in run["results"] if x["model"] == m]:
                mark = (f"{GREEN}PASS{END}" if r["pass"] else
                        f"{YELLOW}ERR {END}" if r["status"] == "error" else f"{RED}FAIL{END}")
                print(f"  {mark}  {r['category']:<{width}} {DIM}{r['latency_ms']}ms  "
                      f"${r['cost_usd']:.5f}  {r['reason'][:70]}{END}")
        s = run["summary"]
        print(f"\n{BOLD}SHIP -> {GREEN}{s['recommended']}{END}  {DIM}{s['reason']}{END}")
        if run.get("suggested_prompt"):
            print(f"\n{YELLOW}Suggested prompt fix available (run with --json to see it).{END}")

    best = next(p for p in run["summary"]["per_model"] if p["model"] == run["summary"]["recommended"])
    sys.exit(0 if best["passed"] == best["total"] else 1)


if __name__ == "__main__":
    main()
