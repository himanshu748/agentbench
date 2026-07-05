"""Shared report persistence (Supabase REST, anon key + RLS).

The anon key is a publishable client key by design; the table only allows
insert and select through row level security policies.
"""
import os

import httpx

from . import mesh_client  # noqa: F401  (ensures .env is loaded first)

SUPA_URL = os.environ.get("SUPA_URL", "")
SUPA_KEY = os.environ.get("SUPA_KEY", "")
ENABLED = bool(SUPA_URL and SUPA_KEY)
TABLE = f"{SUPA_URL}/rest/v1/agentbench_runs"
HEADERS = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}",
           "Content-Type": "application/json"}


async def save_run(run: dict):
    """Best-effort persist; a storage failure never breaks a benchmark."""
    if not ENABLED:
        return
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            await c.post(TABLE, headers={**HEADERS, "Prefer": "resolution=ignore-duplicates"},
                         json={"id": run["run_id"], "suite_name": run.get("suite_name"),
                               "payload": run})
    except Exception:
        pass


async def get_run(run_id: str) -> dict | None:
    if not ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(TABLE, headers=HEADERS,
                            params={"id": f"eq.{run_id}", "select": "payload"})
            rows = r.json()
            return rows[0]["payload"] if rows else None
    except Exception:
        return None
