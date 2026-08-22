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
    repeated = (run.get("repeat") or 1) > 1
    for m in run.get("models", []):
        rows = [x for x in results if x["model"] == m]
        print(f"\n{BOLD}{m}{END}")
        for r in rows:
            if r.get("flaky"):
                mark = f"{YELLOW}FLAKY{END}"
            elif r["pass"]:
                mark = f"{GREEN}PASS {END}"
            elif r.get("status") == "error":
                mark = f"{YELLOW}ERR  {END}"
            else:
                mark = f"{RED}FAIL {END}"
            rate = ""
            if repeated:
                rate = f"{DIM}{r.get('passes', 0)}/{r.get('runs', 1)}{END} "
            print(f"  {mark} {rate} {r['category']:<{width}} {DIM}{r['latency_ms']}ms  "
                  f"${r['cost_usd']:.5f}  {(r.get('reason') or '')[:64]}{END}")


def print_flakiness(run: dict):
    fk = run.get("flakiness") or {}
    if not fk.get("measured"):
        return
    n, total = fk["flaky_count"], fk.get("total_cells", 0)
    color = GREEN if n == 0 else YELLOW
    print(f"\n{BOLD}Flakiness{END} {DIM}(each cell x{fk['repeat']}){END}: "
          f"{color}{n}/{total} cells flaky{END}  "
          f"{DIM}consistency {fk['consistency']:.0%}{END}")
    for c in fk.get("flaky_cells", [])[:8]:
        print(f"  {YELLOW}~{END} {c['category']:<22} {DIM}{c['passes']}/{c['runs']} "
              f"passed · {c['model']}{END}")


def print_regression(diff: dict):
    reg, fixes = diff["regressions"], diff["fixes"]
    delta = diff["current_pass_rate"] - diff["baseline_pass_rate"]
    arrow = f"{GREEN}+{delta:.0%}{END}" if delta >= 0 else f"{RED}{delta:.0%}{END}"
    print(f"\n{BOLD}vs baseline{END} {DIM}{diff['baseline_pass_rate']:.0%} -> "
          f"{diff['current_pass_rate']:.0%}{END} ({arrow})")
    if reg:
        print(f"  {RED}{len(reg)} regression(s):{END}")
        for r in reg[:8]:
            print(f"    {RED}-{END} {r['category']:<22} {DIM}{r['model']}{END}")
    if fixes:
        print(f"  {GREEN}{len(fixes)} fix(es):{END}")
        for r in fixes[:8]:
            print(f"    {GREEN}+{END} {r['category']:<22} {DIM}{r['model']}{END}")
    if not reg and not fixes:
        print(f"  {DIM}no per-cell changes{END}")


def write_step_summary(run: dict, best: dict, ok: bool, diff: dict = None):
    """Render a markdown report into the GitHub Actions job summary, if present."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    s = run["summary"]
    repeated = (run.get("repeat") or 1) > 1
    lines = [f"## AgentBench — {run.get('suite_name') or 'suite'}", "",
             ("✅ **Passed**" if ok else "❌ **Failed**")
             + f" — recommended model `{s['recommended']}` "
               f"scored {best['passed']}/{best['total']}", ""]
    header = "| model | passed | " + ("flaky | " if repeated else "") + "cost | avg latency |"
    sep = "| --- | --- | " + ("--- | " if repeated else "") + "--- | --- |"
    lines += [header, sep]
    for p in s["per_model"]:
        flaky = f"{p.get('flaky', 0)} | " if repeated else ""
        lines.append(f"| `{p['model']}` | {p['passed']}/{p['total']} | {flaky}"
                     f"${p['total_cost_usd']:.4f} | {p['avg_latency_ms']}ms |")
    if diff:
        lines += ["", f"**vs baseline:** {diff['baseline_pass_rate']:.0%} → "
                  f"{diff['current_pass_rate']:.0%} · "
                  f"{len(diff['regressions'])} regression(s), {len(diff['fixes'])} fix(es)"]
    fk = run.get("flakiness") or {}
    if fk.get("measured"):
        lines += ["", f"**Flakiness:** {fk['flaky_count']}/{fk.get('total_cells', 0)} "
                  f"cells flaky across {fk['repeat']} runs "
                  f"(consistency {fk['consistency']:.0%})"]
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


def _xml_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_junit(run: dict, path: str):
    """Emit a JUnit XML report so CI systems render per-test pass/fail natively.

    One <testcase> per result cell; a failing cell carries a <failure>, an
    errored cell an <error>, and a flaky cell is marked skipped-with-note so it
    shows up without hard-failing a suite that only gates on pass rate."""
    results = run.get("results") or []
    failures = sum(1 for r in results if not r["pass"] and r.get("status") != "error")
    errors = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("flaky"))
    total_time = sum(r.get("latency_ms", 0) for r in results) / 1000.0
    name = run.get("suite_name") or "AgentBench"
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<testsuite name="{_xml_escape(name)}" tests="{len(results)}" '
             f'failures="{failures}" errors="{errors}" skipped="{skipped}" '
             f'time="{total_time:.3f}">']
    for r in results:
        case = (f'{r["category"]} [{r["model"]}]')
        classname = _xml_escape(name)
        t = r.get("latency_ms", 0) / 1000.0
        reason = _xml_escape((r.get("reason") or "")[:500])
        lines.append(f'  <testcase classname="{classname}" name="{_xml_escape(case)}" '
                     f'time="{t:.3f}">')
        if r.get("status") == "error":
            lines.append(f'    <error message="{reason}"></error>')
        elif not r["pass"]:
            lines.append(f'    <failure message="{reason}"></failure>')
        elif r.get("flaky"):
            lines.append(f'    <skipped message="flaky: '
                         f'{r.get("passes",0)}/{r.get("runs",1)} passed"></skipped>')
        lines.append('  </testcase>')
    lines.append('</testsuite>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _cell_key(r: dict) -> tuple:
    return (r.get("model"), r.get("category"), r.get("input"))


def regression_diff(run: dict, baseline: dict) -> dict:
    """Compare a run against a saved baseline report, per cell.

    Returns the cells that newly fail (regressions), newly pass (fixes) and the
    pass-rate delta per model. This is what makes the tool a real CI gate: it can
    fail a PR specifically because a test that used to pass now doesn't."""
    base_by_key = {_cell_key(r): r for r in baseline.get("results") or []}
    regressions, fixes = [], []
    for r in run.get("results") or []:
        prev = base_by_key.get(_cell_key(r))
        if prev is None:
            continue
        if prev["pass"] and not r["pass"]:
            regressions.append(r)
        elif not prev["pass"] and r["pass"]:
            fixes.append(r)
    def rate(report):
        rs = report.get("results") or []
        return sum(1 for r in rs if r["pass"]) / len(rs) if rs else 0.0
    return {"regressions": regressions, "fixes": fixes,
            "baseline_pass_rate": round(rate(baseline), 4),
            "current_pass_rate": round(rate(run), 4),
            "new_cells": sum(1 for r in run.get("results") or []
                             if _cell_key(r) not in base_by_key)}


def cmd_run(args, base: str) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ApiError("--models must list at least one model id")

    baseline = None
    if args.baseline:
        try:
            baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ApiError(f"could not read baseline {args.baseline}: {e}") from None

    suite_id = resolve_suite(base, args.suite, args.timeout)
    # Banner goes to stderr so `--json` output stays pipeable into jq.
    extra = f" · x{args.repeat}" if args.repeat > 1 else ""
    print(f"{DIM}agentbench · {base} · {len(models)} model(s){extra}{END}", file=sys.stderr)
    run = api(base, "/api/runs",
              {"suite_id": suite_id, "models": models, "sync": True,
               "repeat": args.repeat}, args.timeout)

    if run.get("status") == "error":
        raise ApiError(f"run failed on the server: {run.get('error', 'unknown error')}")
    summary = run.get("summary")
    if not summary or not summary.get("per_model"):
        raise ApiError("run returned no results")

    if args.json:
        print(json.dumps(run, indent=2))
    else:
        print_table(run)
        print_flakiness(run)

    best = next(p for p in summary["per_model"] if p["model"] == summary["recommended"])
    rate = best["passed"] / best["total"] if best["total"] else 0.0
    ok = rate >= args.min_pass_rate

    diff = None
    if baseline:
        diff = regression_diff(run, baseline)
        if not args.json:
            print_regression(diff)
        if args.fail_on_regression and diff["regressions"]:
            ok = False

    if args.fail_on_flaky and (run.get("flakiness") or {}).get("flaky_count"):
        ok = False

    # Side-effect outputs happen regardless of pass/fail so CI always has them.
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"{DIM}wrote report -> {args.output}{END}", file=sys.stderr)
    if args.junit:
        write_junit(run, args.junit)
        print(f"{DIM}wrote JUnit XML -> {args.junit}{END}", file=sys.stderr)

    if not args.json:
        verdict = f"{GREEN}SHIP{END}" if ok else f"{RED}BLOCK{END}"
        print(f"\n{BOLD}{verdict} -> {summary['recommended']}{END}  {DIM}{summary['reason']}{END}")
        if args.min_pass_rate < 1.0:
            print(f"{DIM}threshold: {args.min_pass_rate:.0%} · actual: {rate:.0%}{END}")
        if run.get("suggested_prompt"):
            print(f"\n{YELLOW}Suggested prompt fix available (re-run with --json to see it).{END}")
        if run.get("run_id"):
            print(f"{DIM}report: {base}/r/{run['run_id']}{END}")

    write_step_summary(run, best, ok, diff)
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


def cmd_categories(args, base: str) -> int:
    for c in api(base, "/api/meta", timeout=args.timeout)["categories"]:
        print(c)
    print("custom", f"{DIM}(bring your own rubric per test){END}")
    return EXIT_OK


def cmd_validate(args, base: str) -> int:
    """Validate a suite YAML against the server without running it (no spend)."""
    path = pathlib.Path(args.suite)
    if not path.exists():
        raise ApiError(f"suite file not found: {args.suite}")
    try:
        suite_id = api(base, "/api/suites",
                       {"yaml_text": path.read_text(encoding="utf-8")}, args.timeout)["suite_id"]
    except ApiError as e:
        print(f"{RED}invalid{END} {args.suite}: {e}", file=sys.stderr)
        return EXIT_TESTS_FAILED
    suite = api(base, f"/api/suites/{suite_id}", timeout=args.timeout)
    print(f"{GREEN}valid{END} {args.suite} {DIM}· {len(suite['tests'])} tests{END}")
    return EXIT_OK


def cmd_estimate(args, base: str) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ApiError("--models must list at least one model id")
    suite_id = resolve_suite(base, args.suite, args.timeout)
    est = api(base, "/api/estimate",
              {"suite_id": suite_id, "models": models, "repeat": args.repeat}, args.timeout)
    if args.json:
        print(json.dumps(est, indent=2))
        return EXIT_OK
    print(f"{BOLD}{est['cells']} cells{END} {DIM}({est['num_tests']} tests × "
          f"{len(est['models'])} model(s) × {est['repeat']} repeat) · "
          f"{est['calls']} MeshAPI calls{END}")
    for p in est["per_model"]:
        print(f"  {p['model']:<32} ~${p['est_cost_usd']:.4f}")
    print(f"{BOLD}~${est['est_cost_usd']:.4f} total{END}  {DIM}{est['note']}{END}")
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
    runp.add_argument("--repeat", type=int, default=1, metavar="N",
                      help="run each test N times to measure flakiness (default 1)")
    runp.add_argument("--output", metavar="FILE",
                      help="write the full JSON report to FILE (a CI artifact / baseline)")
    runp.add_argument("--junit", metavar="FILE",
                      help="write a JUnit XML report to FILE for CI test reporters")
    runp.add_argument("--baseline", metavar="FILE",
                      help="compare against a prior --output report and show regressions")
    runp.add_argument("--fail-on-regression", action="store_true",
                      help="exit non-zero if any cell that passed in the baseline now fails")
    runp.add_argument("--fail-on-flaky", action="store_true",
                      help="exit non-zero if any cell is flaky (needs --repeat > 1)")
    runp.set_defaults(func=cmd_run)

    estp = sub.add_parser("estimate", help="estimate a run's cost/size without spending")
    estp.add_argument("suite", help="path to suite YAML or a built-in suite id")
    estp.add_argument("--models", required=True, help="comma-separated model ids (or auto)")
    estp.add_argument("--repeat", type=int, default=1, metavar="N",
                      help="repeats per cell (default 1)")
    estp.add_argument("--json", action="store_true", help="print full JSON")
    estp.set_defaults(func=cmd_estimate)

    valp = sub.add_parser("validate", help="validate a suite YAML without running it")
    valp.add_argument("suite", help="path to a suite YAML")
    valp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("suites", help="list suites available on the server")
    sp.set_defaults(func=cmd_suites)

    mp = sub.add_parser("models", help="list models available to benchmark")
    mp.set_defaults(func=cmd_models)

    cp = sub.add_parser("categories", help="list built-in behavior categories")
    cp.set_defaults(func=cmd_categories)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    base = (args.url or os.environ.get("AGENTBENCH_URL") or DEFAULT_URL).rstrip("/")
    if getattr(args, "min_pass_rate", 1.0) < 0 or getattr(args, "min_pass_rate", 1.0) > 1:
        ap.error("--min-pass-rate must be between 0 and 1")
    if getattr(args, "repeat", 1) < 1:
        ap.error("--repeat must be at least 1")
    if getattr(args, "fail_on_flaky", False) and getattr(args, "repeat", 1) == 1:
        ap.error("--fail-on-flaky needs --repeat > 1 to have anything to measure")
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
