# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub private vulnerability reporting](https://github.com/himanshu748/agentbench/security/advisories/new),
not in public issues. Include steps to reproduce and the impact you expect. You should get a
response within a week.

## Running AgentBench safely

- AgentBench spends money on whatever OpenRouter key you configure. If you expose a server to
  people you don't trust, set `AGENTBENCH_RATE_LIMIT`, keep `AGENTBENCH_MAX_CELLS` low, and use an
  OpenRouter key with a [credit limit](https://openrouter.ai/settings/keys).
- The server has no authentication. Put it behind your own auth proxy or private network if it
  holds a key.
- Never commit `.env`. In GitHub Actions pass the key as a secret
  (`openrouter-api-key: ${{ secrets.OPENROUTER_API_KEY }}`).
