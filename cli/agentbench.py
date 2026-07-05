#!/usr/bin/env python3
"""AgentBench CLI: run behavior test suites from the terminal or CI.

Usage:
  python agentbench.py run <suite.yaml> --models openai/gpt-4o-mini,anthropic/claude-haiku-4.5
  python agentbench.py run customer-support --models auto        # built-in suite id
  AGENTBENCH_URL=https://agentbench-three.vercel.app python agentbench.py run suite.yaml ...

Exit code 0 if every test passes on the recommended model, 1 otherwise
(so a failing behavior test fails your CI job).
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.request

BASE = os.environ.get("AGENTBENCH_URL", "https://agentbench-three.vercel.app").rstrip("/")

GREEN, RED, YELLOW, DIM, BOLD, END = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"


def api(path, payload=None):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(prog="agentbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run a suite")
    runp.add_argument("suite", help="path to suite YAML or a built-in suite id")
    runp.add_argument("--models", required=True, help="comma-separated MeshAPI model ids (or auto)")
    runp.add_argument("--json", action="store_true", help="print full JSON report")
    args = ap.parse_args()

    path = pathlib.Path(args.suite)
    if path.exists():
        suite_id = api("/api/suites", {"yaml_text": path.read_text()})["suite_id"]
    else:
        suite_id = args.suite

    models = args.models.split(",")
    print(f"{DIM}agentbench · {BASE} · {len(models)} model(s){END}")
    run = api("/api/runs", {"suite_id": suite_id, "models": models, "sync": True})

    if args.json:
        print(json.dumps(run, indent=2))
    else:
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
