"""Shared report persistence (Supabase REST, anon key + RLS).

The anon key is a publishable client key by design; the table only allows
insert and select through row level security policies.
"""
import os

import httpx

SUPA_URL = os.environ.get("SUPA_URL", "https://tmsfudajqumspruyssov.supabase.co")
SUPA_KEY = os.environ.get(
    "SUPA_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtc2Z1ZGFqcXVtc3BydXlzc292Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODI3MDEwMjYsImV4cCI6MjA5ODI3NzAyNn0.JjHmxrJMx-G6Dj14FCgiSY3V3Gsivl0cLnq-K3ibvgg",
)
TABLE = f"{SUPA_URL}/rest/v1/agentbench_runs"
HEADERS = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}",
           "Content-Type": "application/json"}


async def save_run(run: dict):
    """Best-effort persist; a storage failure never breaks a benchmark."""
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            await c.post(TABLE, headers={**HEADERS, "Prefer": "resolution=ignore-duplicates"},
                         json={"id": run["run_id"], "suite_name": run.get("suite_name"),
                               "payload": run})
    except Exception:
        pass


async def get_run(run_id: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(TABLE, headers=HEADERS,
                            params={"id": f"eq.{run_id}", "select": "payload"})
            rows = r.json()
            return rows[0]["payload"] if rows else None
    except Exception:
        return None
