"""Report persistence is best-effort: it must never break a benchmark."""
import httpx
import pytest

from backend import storage


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    async def handle_async_request(self, request):
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
def supabase(monkeypatch):
    def _install(handler):
        transport = _Transport(handler)
        monkeypatch.setattr(storage, "ENABLED", True)
        monkeypatch.setattr(storage, "TABLE", "http://supabase.test/rest/v1/agentbench_runs")
        original = httpx.AsyncClient

        def factory(*a, **k):
            k.pop("transport", None)
            return original(*a, transport=transport, **k)

        monkeypatch.setattr(storage.httpx, "AsyncClient", factory)
        return transport
    return _install


RUN = {"run_id": "abc", "suite_name": "S", "results": []}


@pytest.mark.asyncio
class TestSaveRun:
    async def test_disabled_storage_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(storage, "ENABLED", False)
        await storage.save_run(RUN)  # must not raise or make a request

    async def test_run_is_posted(self, supabase):
        t = supabase(lambda req: httpx.Response(201, json={}, request=req))
        await storage.save_run(RUN)
        assert len(t.requests) == 1
        assert t.requests[0].method == "POST"

    async def test_a_storage_error_never_propagates(self, supabase):
        """A benchmark that succeeded must not fail because sharing did."""
        def boom(req):
            raise httpx.ConnectError("supabase down", request=req)
        supabase(boom)
        await storage.save_run(RUN)  # swallowed

    async def test_a_5xx_never_propagates(self, supabase):
        supabase(lambda req: httpx.Response(500, json={"message": "nope"}, request=req))
        await storage.save_run(RUN)


@pytest.mark.asyncio
class TestGetRun:
    async def test_disabled_storage_returns_none(self, monkeypatch):
        monkeypatch.setattr(storage, "ENABLED", False)
        assert await storage.get_run("abc") is None

    async def test_found_row_returns_the_payload(self, supabase):
        supabase(lambda req: httpx.Response(200, json=[{"payload": RUN}], request=req))
        assert await storage.get_run("abc") == RUN

    async def test_missing_row_returns_none(self, supabase):
        supabase(lambda req: httpx.Response(200, json=[], request=req))
        assert await storage.get_run("abc") is None

    async def test_transport_failure_returns_none(self, supabase):
        def boom(req):
            raise httpx.ConnectError("down", request=req)
        supabase(boom)
        assert await storage.get_run("abc") is None
