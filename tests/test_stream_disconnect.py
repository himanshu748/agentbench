import asyncio

import pytest

from backend import main


def test_closing_stream_cancels_pending_cells(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        calls = []

        async def run_one(_sem, _prompt, test, _model):
            calls.append(test["input"])
            if test["input"] == "first":
                await started.wait()
                return {"input": "first", "pass": True, "status": "done"}
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        class Request:
            async def is_disconnected(self):
                return False

        suite = {"system_prompt": "prompt", "tests": [{"input": "first"}, {"input": "pending"}]}
        monkeypatch.setattr(main, "_validate_run", lambda *_: (suite, ["model"]))
        monkeypatch.setattr(main.runner, "_run_one", run_one)
        response = await main.stream_run("suite", "model", Request())
        stream = response.body_iterator
        assert (await anext(stream)).startswith("event: start")
        assert (await anext(stream)).startswith("event: cell")
        await stream.aclose()
        assert cancelled.is_set()
        assert calls == ["first", "pending"]

    asyncio.run(scenario())


def test_disconnected_stream_skips_fix_and_storage(monkeypatch):
    async def scenario():
        cancelled = asyncio.Event()
        started = asyncio.Event()
        stored = []
        fixes = []

        async def run_one(_sem, _prompt, test, _model):
            if test["input"] == "first":
                await started.wait()
                return {"input": "first", "pass": False, "status": "done", "category": "x", "reason": "x"}
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def save_run(run):
            stored.append(run)

        async def suggest_fix(*args):
            fixes.append(args)

        class Request:
            calls = 0

            async def is_disconnected(self):
                self.calls += 1
                return self.calls > 1

        suite = {"system_prompt": "prompt", "tests": [{"input": "first"}, {"input": "pending"}]}
        monkeypatch.setattr(main, "_validate_run", lambda *_: (suite, ["model"]))
        monkeypatch.setattr(main.runner, "_run_one", run_one)
        monkeypatch.setattr(main.storage, "save_run", save_run)
        monkeypatch.setattr(main.judge, "suggest_fix", suggest_fix)
        response = await main.stream_run("suite", "model", Request())
        stream = response.body_iterator
        assert (await anext(stream)).startswith("event: start")
        async def collect():
            return [event async for event in stream]

        events = await asyncio.wait_for(collect(), timeout=1)
        assert not any(event.startswith("event: done") for event in events)
        assert not stored and not fixes
        assert cancelled.is_set()
        assert not main.runner.RUNS

    monkeypatch.setattr(main.runner, "RUNS", {})
    asyncio.run(scenario())


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_asgi_disconnect_cancels_inflight_cell(monkeypatch, spec_version):
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        disconnect = asyncio.Event()
        sent = []
        stored = []
        calls = []

        async def chat(*args, **kwargs):
            calls.append((args, kwargs))
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        async def save_run(run):
            stored.append(run)

        suite = {"system_prompt": "prompt", "tests": [
            {"input": "first", "category": "x"}, {"input": "second", "category": "x"},
        ]}
        monkeypatch.setattr(main, "_validate_run", lambda *_: (suite, ["model"]))
        monkeypatch.setattr(main.runner, "CONCURRENCY", 1)
        monkeypatch.setattr(main.llm_client, "chat", chat)
        monkeypatch.setattr(main.storage, "save_run", save_run)
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1", "method": "GET", "scheme": "http",
            "path": "/api/runs/stream", "raw_path": b"/api/runs/stream",
            "query_string": b"suite_id=suite&models=model", "headers": [],
            "client": ("127.0.0.1", 1234), "server": ("localhost", 80),
        }
        app_task = asyncio.create_task(main.app(scope, receive, send))
        await asyncio.wait_for(started.wait(), timeout=2)
        disconnect.set()
        await asyncio.wait_for(app_task, timeout=2)
        assert cancelled.is_set()
        assert len(calls) == 1
        assert not stored
        assert not any(b"event: done" in message.get("body", b"") for message in sent)

    asyncio.run(scenario())


def test_disconnect_cancels_suggested_fix(monkeypatch):
    async def scenario():
        fixing = asyncio.Event()
        cancelled = asyncio.Event()
        disconnect = asyncio.Event()
        stored = []

        async def run_one(_sem, _prompt, _test, _model):
            return {"input": "x", "model": "model", "pass": False, "status": "done",
                    "category": "x", "reason": "x", "cost_usd": 0, "latency_ms": 1}

        async def suggest_fix(*_args):
            fixing.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(_message):
            pass

        async def save_run(run):
            stored.append(run)

        suite = {"system_prompt": "prompt", "tests": [{"input": "x"}]}
        monkeypatch.setattr(main, "_validate_run", lambda *_: (suite, ["model"]))
        monkeypatch.setattr(main.runner, "_run_one", run_one)
        monkeypatch.setattr(main.judge, "suggest_fix", suggest_fix)
        monkeypatch.setattr(main.storage, "save_run", save_run)
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "method": "GET", "scheme": "http",
            "path": "/api/runs/stream", "raw_path": b"/api/runs/stream",
            "query_string": b"suite_id=suite&models=model", "headers": [],
            "client": ("127.0.0.1", 1234), "server": ("localhost", 80),
        }
        app_task = asyncio.create_task(main.app(scope, receive, send))
        await asyncio.wait_for(fixing.wait(), timeout=2)
        disconnect.set()
        await asyncio.wait_for(app_task, timeout=2)
        assert cancelled.is_set()
        assert not stored

    asyncio.run(scenario())
