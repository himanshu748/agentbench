"""Run report persistence in a local SQLite file, so /r/<run_id> links survive restarts.

Config (env):
  AGENTBENCH_DB  Path to the SQLite file (default data/agentbench.db in the repo
                 root). Set to an empty string to disable persistence.
"""
import asyncio
import json
import os
import pathlib
import sqlite3

from . import llm_client  # noqa: F401  (ensures .env is loaded first)

_DEFAULT = pathlib.Path(__file__).resolve().parent.parent / "data" / "agentbench.db"
DB_PATH = os.environ.get("AGENTBENCH_DB", str(_DEFAULT))
ENABLED = bool(DB_PATH)


def _connect() -> sqlite3.Connection:
    if DB_PATH != ":memory:":
        pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, suite_name TEXT, "
                 "created_at TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)")
    return conn


def _save(run: dict):
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO runs (id, suite_name, payload) VALUES (?, ?, ?)",
                     (run["run_id"], run.get("suite_name"), json.dumps(run)))
    conn.close()


def _get(run_id: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


async def save_run(run: dict):
    """Best-effort persist; a storage failure never breaks a benchmark."""
    if not ENABLED:
        return
    try:
        await asyncio.to_thread(_save, run)
    except Exception:
        pass


async def get_run(run_id: str) -> dict | None:
    if not ENABLED:
        return None
    try:
        return await asyncio.to_thread(_get, run_id)
    except Exception:
        return None
