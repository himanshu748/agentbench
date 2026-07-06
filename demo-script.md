# AgentBench — Demo Video Script (~90s)

**Track:** Agents & Automation | **Hackathon:** MeshAPI | **Deadline:** 2026-07-12

---

### [0:00–0:12] Hook — the problem
*(Screen: a prompt/agent config file, someone editing it)*

> "Every time you tweak a prompt or swap models, you're flying blind. Did it get better? Worse? Did it break for the edge case a customer hit last month? Most teams just... ship and hope."

### [0:12–0:25] Introduce AgentBench
*(Screen: agentbench-three.vercel.app landing page)*

> "This is AgentBench — CI/CD for AI prompts. You write test suites once, run them against any model through MeshAPI, and let an LLM judge score the outputs — so a prompt change gets tested before it ships, the same way code does."

### [0:25–0:45] Live console walkthrough
*(Screen: /app console — pick a suite, e.g. "customer-support", select 2-3 models, hit Run)*

> "Here's the console. I pick a test suite — this one's customer support — select a few models to compare, and run. Under the hood, every call routes through MeshAPI, so I'm benchmarking GPT, Claude, and open models side by side with one API key."

*(Screen: SSE live-streaming results filling in row by row)*

> "Results stream in live. Each response gets graded by a judge model against a rubric — not just 'did it match a string,' but did it actually satisfy the intent."

### [0:45–1:00] Judge scoring + regression
*(Screen: scored grid, red/green cells, maybe a "fix and rerun" click)*

> "Red cells are failures. I can see exactly which model, which test case, and why — then fix the prompt and rerun in one click to confirm the regression is gone."

### [1:00–1:15] CI/CD integration — the killer feature
*(Screen: action.yml / a GitHub Actions run, or terminal running the CLI with exit code)*

> "And this isn't just a dashboard — it's a real CI gate. Drop our GitHub Action into any repo's workflow, point it at your suite, and every pull request that touches a prompt gets tested automatically. If the recommended model fails a test, the check fails and the merge is blocked."

### [1:15–1:25] Shareable reports + close
*(Screen: /r/{run_id} shared report page, then back to landing)*

> "Every run gets a shareable, permanent report link — so 'trust me, I tested it' becomes an actual artifact you can point to. AgentBench: ship prompt changes with the same confidence you ship code."

*(End card: logo + agentbench-three.vercel.app + GitHub repo link)*

---

## B-roll shot list
1. Landing page hero (brand green #2BE58A, Space Grotesk type)
2. Suite picker + model multi-select
3. Live SSE streaming grid (rows populating)
4. Judge score breakdown for one failing test case
5. "Fix and rerun" click → green after red
6. GitHub Action YAML snippet (`action.yml`) + a real Actions run log
7. CLI in terminal: `agentbench run suites/customer-support.yaml` → exit code
8. Shared report page `/r/{run_id}`
9. Badge embed (e.g. in a README)

## Talking points to have ready for Q&A / judges
- Every LLM call — including the judge — goes through MeshAPI's OpenAI-compatible endpoint, pulling live pricing from `/v1/models` (531 models).
- Real run cost ~$0.005 for 5 tests × 3 models + judging — cheap enough to run per-PR.
- 8 built-in suites (customer support, coding assistant, healthcare triage, HR policy, brand voice, sales assistant, content moderation) + custom rubric support.
- Fully deployed and live, not just localhost: https://agentbench-three.vercel.app
