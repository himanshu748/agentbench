#!/usr/bin/env python3
"""AgentBench CLI: run behavior test suites from the terminal or CI.

Usage:
  agentbench run <suite.yaml> --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
  agentbench run customer-support --models auto        # built-in suite id
  agentbench suites                                    # list available suites
  agentbench models                                    # list benchmarkable models
  AGENTBENCH_URL=https://agentbench-three.vercel.app agentbench run suite.yaml ...

Exit codes:
  0  the recommended model met the pass threshold
  1  behavior tests failed  -> fails your CI job
  2  the benchmark could not run (bad arguments, network, server error)

Stdlib only: this file is what the GitHub Action executes, so it must run on a
bare `actions/setup-python` runner with nothing installed.
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://agentbench-three.vercel.app"

EXIT_OK, EXIT_TESTS_FAILED, EXIT_ERROR = 0, 1, 2


def _supports_color(stream) -> bool:
    # Honour the NO_COLOR convention and never emit escapes into a pipe or a
    # CI log file that renders them literally.
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("GITHUB_ACTIONS") == "true":
        return True
    return hasattr(stream, "isatty") and stream.isatty()


if _supports_color(sys.stdout) or _supports_color(sys.stderr):
    GREEN, RED, YELLOW, DIM, BOLD, END = (
        "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m")
else:
    GREEN = RED = YELLOW = DIM = BOLD = END = ""


class ApiError(Exception):
    """A request to the AgentBench server failed in a way worth reporting plainly."""


def api(base: str, path: str, payload=None, timeout: float = 180.0):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json",
                 "User-Agent": "agentbench-cli/1.1"},
        method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        # FastAPI puts the useful message in {"detail": ...}; surface it instead
        # of a bare "HTTP Error 400: Bad Request".
        detail = ""
        try:
            body = json.loads(e.read().decode() or "{}")
            detail = body.get("detail") or body.get("message") or ""
            if isinstance(detail, list):  # pydantic validation errors
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
        except Exception:
            pass
        raise ApiError(f"{base + path} returned HTTP {e.code}"
                       + (f": {detail}" if detail else "")) from None
    except urllib.error.URLError as e:
        raise ApiError(f"could not reach {base} ({e.reason}). "
                       "Set AGENTBENCH_URL or --url to point at a running server.") from None
    except TimeoutError:
        raise ApiError(f"request to {base + path} timed out after {timeout:.0f}s. "
                       "Try fewer models or a smaller suite.") from None
    except json.JSONDecodeError:
        raise ApiError(f"{base + path} returned a non-JSON response") from None


def resolve_suite(base: str, suite_arg: str, timeout: float) -> str:
    """A path uploads the YAML and returns its new id; anything else is an id."""
    path = pathlib.Path(suite_arg)
    if not path.exists():
        if suite_arg.endswith((".yaml", ".yml")):
            raise ApiError(f"suite file not found: {suite_arg}")
        return suite_arg
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ApiError(f"could not read {suite_arg}: {e}") from None
    return api(base, "/api/suites", {"yaml_text": text}, timeout)["suite_id"]


def print_table(run: dict):
    results = run.get("results") or []
    width = max((len(r["category"]) for r in results), default=8) + 2
    for m in run.get("models", []):
        rows = [x for x in results if x["model"] == m]
        print(f"\n{BOLD}{m}{END}")
        for r in rows:
            mark = (f"{GREEN}PASS{END}" if r["pass"] else
                    f"{YELLOW}ERR {END}" if r.get("status") == "error" else f"{RED}FAIL{END}")
            print(f"  {mark}  {r['category']:<{width}} {DIM}{r['latency_ms']}ms  "
                  f"${r['cost_usd']:.5f}  {(r.get('reason') or '')[:70]}{END}")


def write_step_summary(run: dict, best: dict, ok: bool):
    """Render a markdown report into the GitHub Actions job summary, if present."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    s = run["summary"]
    lines = [f"## AgentBench — {run.get('suite_name') or 'suite'}", "",
             ("✅ **Passed**" if ok else "❌ **Failed**")
             + f" — recommended model `{s['recommended']}` "
               f"scored {best['passed']}/{best['total']}", "",
             "| model | passed | cost | avg latency |",
             "| --- | --- | --- | --- |"]
    for p in s["per_model"]:
        lines.append(f"| `{p['model']}` | {p['passed']}/{p['total']} | "
                     f"${p['total_cost_usd']:.4f} | {p['avg_latency_ms']}ms |")
    failures = [r for r in run.get("results") or [] if not r["pass"]]
    if failures:
        lines += ["", "<details><summary>Failing tests</summary>", ""]
        for f in failures[:20]:
            # Escape pipes so a judge reason cannot break out of the table cell.
            reason = (f.get("reason") or "").replace("|", "\\|")[:160]
            lines.append(f"- `{f['model']}` · **{f['category']}** — {reason}")
        lines += ["", "</details>"]
    try:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass  # a summary that cannot be written must never fail the build


def cmd_run(args, base: str) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ApiError("--models must list at least one model id")

    suite_id = resolve_suite(base, args.suite, args.timeout)
    # Banner goes to stderr so `--json` output stays pipeable into jq.
    print(f"{DIM}agentbench · {base} · {len(models)} model(s){END}", file=sys.stderr)
    run = api(base, "/api/runs",
              {"suite_id": suite_id, "models": models, "sync": True}, args.timeout)

    if run.get("status") == "error":
        raise ApiError(f"run failed on the server: {run.get('error', 'unknown error')}")
    summary = run.get("summary")
    if not summary or not summary.get("per_model"):
        raise ApiError("run returned no results")

    if args.json:
        print(json.dumps(run, indent=2))
    else:
        print_table(run)

    best = next(p for p in summary["per_model"] if p["model"] == summary["recommended"])
    rate = best["passed"] / best["total"] if best["total"] else 0.0
    ok = rate >= args.min_pass_rate

    if not args.json:
        verdict = f"{GREEN}SHIP{END}" if ok else f"{RED}BLOCK{END}"
        print(f"\n{BOLD}{verdict} -> {summary['recommended']}{END}  {DIM}{summary['reason']}{END}")
        if args.min_pass_rate < 1.0:
            print(f"{DIM}threshold: {args.min_pass_rate:.0%} · actual: {rate:.0%}{END}")
        if run.get("suggested_prompt"):
            print(f"\n{YELLOW}Suggested prompt fix available (re-run with --json to see it).{END}")
        if run.get("run_id"):
            print(f"{DIM}report: {base}/r/{run['run_id']}{END}")

    write_step_summary(run, best, ok)
    return EXIT_OK if ok else EXIT_TESTS_FAILED


def cmd_suites(args, base: str) -> int:
    for s in api(base, "/api/suites", timeout=args.timeout):
        print(f"{s['suite_id']:<24} {DIM}{s['num_tests']:>2} tests{END}  {s.get('name') or ''}")
    return EXIT_OK


def cmd_models(args, base: str) -> int:
    meta = api(base, "/api/meta", timeout=args.timeout)
    for m in meta["models"]:
        print(m)
    print(f"{DIM}judge: {meta['judge_model']} · catalog: {meta['model_count']} "
          f"models ({meta['chat_count']} chat){END}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="agentbench",
        description="Behavior tests for AI prompts, run across models via MeshAPI.")
    ap.add_argument("--url", default=None,
                    help=f"AgentBench server (env AGENTBENCH_URL, default {DEFAULT_URL})")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout in seconds (default 180)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="run a suite")
    runp.add_argument("suite", help="path to suite YAML or a built-in suite id")
    runp.add_argument("--models", required=True,
                      help="comma-separated MeshAPI model ids (or auto)")
    runp.add_argument("--json", action="store_true", help="print full JSON report")
    runp.add_argument("--min-pass-rate", type=float, default=1.0, metavar="RATE",
                      help="pass rate the recommended model must reach, 0-1 "
                           "(default 1.0: every test must pass)")
    runp.set_defaults(func=cmd_run)

    sp = sub.add_parser("suites", help="list suites available on the server")
    sp.set_defaults(func=cmd_suites)

    mp = sub.add_parser("models", help="list models available to benchmark")
    mp.set_defaults(func=cmd_models)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    base = (args.url or os.environ.get("AGENTBENCH_URL") or DEFAULT_URL).rstrip("/")
    if getattr(args, "min_pass_rate", 1.0) < 0 or getattr(args, "min_pass_rate", 1.0) > 1:
        ap.error("--min-pass-rate must be between 0 and 1")
    try:
        sys.exit(args.func(args, base))
    except ApiError as e:
        print(f"{RED}agentbench: {e}{END}", file=sys.stderr)
        sys.exit(EXIT_ERROR)
    except KeyboardInterrupt:
        print(f"\n{DIM}interrupted{END}", file=sys.stderr)
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
