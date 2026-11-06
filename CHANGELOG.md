# Changelog

## 2.0.0

The hosted demo is retired. AgentBench is now a self-hosted, open source project.

### Breaking
- Model provider switched from MeshAPI to **OpenRouter**. Set `OPENROUTER_API_KEY`
  (replaces `MESH_API_KEY`); `MESH_BASE_URL` / `MESH_JUDGE_MODEL` become
  `OPENROUTER_BASE_URL` / `AGENTBENCH_JUDGE_MODEL`. `auto` now maps to `openrouter/auto`.
- The CLI defaults to `http://localhost:8000` instead of the hosted demo.
- The GitHub Action runs the engine inside the job and needs an `openrouter-api-key` input
  (or `url` of your own server). Use `himanshu748/agentbench@v2`.
- Report storage moved from Supabase to a local SQLite file (`AGENTBENCH_DB`).
- Vercel deployment files removed. Python 3.10+ required.

### Added
- `agentbench run --local`: run the full engine in-process, no server needed.
- Any OpenRouter model id is accepted by the API and CLI; `AGENTBENCH_MODELS` sets the console list.
- Cost per cell uses the cost OpenRouter reports for each call.
- Dockerfile, `/api/health`, `.env.example`, test suite, CI, contributor docs.
- `AGENTBENCH_RATE_LIMIT` and `AGENTBENCH_MAX_CELLS` settings (rate limit off by default).

### Fixed
- `/api/fix-rerun` now validates models and respects the run size limit.
- Unknown model ids return a clear 400 instead of being silently dropped.

## 1.0.0

Initial release built for the MeshAPI Hackathon (July 2026).
