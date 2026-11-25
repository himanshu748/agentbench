# HarnessTwin Hard: cloud preflight

This isolated branch is a verified infrastructure preflight, not a published coding-agent benchmark. The AgentBench main branch is unchanged.

Official SWE-bench Pro V2 commit: 66f92766bba642462d4bbe5479e83f91f9211862 (v2.0.0). First task is the official HARD-51 Ansible instance b748edea.

Two independent Hugging Face Jobs ran the official verifier:
- Empty patch: 41/46 tests; reward 0 (expected). https://huggingface.co/jobs/HIMANSHUKUMARJHA/6b0747edb6a90bbe997e674e
- Official reference patch: 46/46 tests; reward 1 (expected). https://huggingface.co/jobs/HIMANSHUKUMARJHA/6b074829b6a90bbe997e67ad

These are controls only. Zero GMI model calls and zero agent attempts. No pass@1 or resolved-rate claim.

Reproduce the controls on a connected HF account, inside fresh official task images, e.g.:
```bash
TASK=instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86
hf jobs run --flavor cpu-basic --timeout 10m "ghcr.io/scaleapi/swe-bench_pro-v2:$TASK" bash -lc "$(cat preflight_control.sh)" _ nop "$TASK"
hf jobs run --flavor cpu-basic --timeout 10m "ghcr.io/scaleapi/swe-bench_pro-v2:$TASK" bash -lc "$(cat preflight_control.sh)" _ oracle "$TASK"
```
Never expose the oracle or official hidden tests to an agent. HF controls use CPU Basic, not the official one-CPU agent budget.

Pinned registry versions: Pi 1.6.1, OpenCode 1.19.6, mini-SWE-agent 2.4.6. Exact GMI account ID and billing rates are intentionally unspecified until authenticated discovery. This branch contains the public-safe preflight code and evidence only; the separate local replacement package contains the unvalidated Docker runner prototype.

Upstream candidate: correct the existing DeepSeek V4 Flash cost record. Models.dev validation passes before and after; bun test fails both before and after; v1 check identifies only the intended three changes. No PR submitted. See upstream-candidate.patch and https://huggingface.co/jobs/HIMANSHUKUMARJHA/6b0748bfb6a90bbe997e6902. Account-specific rates and maintainer approval pending.
