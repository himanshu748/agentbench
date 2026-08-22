"""MeshAPI client: response shapes, retry policy, pricing, catalog parsing."""
import httpx
import pytest

from backend import mesh_client


class TestExtractContent:
    def test_standard_openai_shape(self):
        data = {"choices": [{"message": {"content": "hello"}}]}
        assert mesh_client._extract_content(data) == "hello"

    def test_content_parts_list_is_joined(self):
        data = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
        assert mesh_client._extract_content(data) == "ab"

    def test_null_content_becomes_empty_string(self):
        data = {"choices": [{"message": {"content": None}}]}
        assert mesh_client._extract_content(data) == ""

    def test_refusal_is_used_when_content_is_absent(self):
        data = {"choices": [{"message": {"refusal": "I can't do that."}}]}
        assert mesh_client._extract_content(data) == "I can't do that."

    def test_message_missing_entirely(self):
        assert mesh_client._extract_content({"choices": [{}]}) == ""

    @pytest.mark.parametrize("data", [{}, {"choices": []}, {"choices": None}])
    def test_no_choices_raises_a_reportable_error(self, data):
        """Regression: this used to be a raw KeyError/IndexError escaping as a 500."""
        with pytest.raises(ValueError):
            mesh_client._extract_content(data)


class TestCatalogParsing:
    def test_bare_list(self):
        assert mesh_client._catalog_entries([{"id": "a"}]) == [{"id": "a"}]

    def test_openai_data_envelope(self):
        """Regression: OpenAI-compatible /v1/models wraps the list in {"data": ...}."""
        assert mesh_client._catalog_entries({"data": [{"id": "a"}]}) == [{"id": "a"}]

    def test_models_envelope(self):
        assert mesh_client._catalog_entries({"models": [{"id": "a"}]}) == [{"id": "a"}]

    @pytest.mark.parametrize("payload", [None, "nope", 42, {}, {"data": None}])
    def test_unexpected_shapes_yield_nothing(self, payload):
        assert mesh_client._catalog_entries(payload) == []

    def test_non_dict_entries_are_dropped(self):
        assert mesh_client._catalog_entries([{"id": "a"}, "junk", None]) == [{"id": "a"}]

    @pytest.mark.parametrize("keys", [
        {"prompt_usd_per_1m": 1.0, "completion_usd_per_1m": 2.0},
        {"input_usd_per_1m": 1.0, "output_usd_per_1m": 2.0},
        {"prompt": 1.0, "completion": 2.0},
    ])
    def test_pricing_key_spellings(self, keys):
        assert mesh_client._pick_price(keys, mesh_client._PROMPT_PRICE_KEYS) == 1.0
        assert mesh_client._pick_price(keys, mesh_client._COMPLETION_PRICE_KEYS) == 2.0

    def test_missing_pricing_key_raises(self):
        with pytest.raises(KeyError):
            mesh_client._pick_price({}, mesh_client._PROMPT_PRICE_KEYS)


class TestCost:
    def test_known_model_uses_its_price(self):
        cost = mesh_client._estimate_cost("openai/gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.15 + 0.60)

    def test_unknown_model_falls_back_to_a_default(self):
        assert mesh_client._estimate_cost("who/knows", 1_000_000, 0) == pytest.approx(1.0)

    def test_zero_tokens_costs_nothing(self):
        assert mesh_client._estimate_cost("openai/gpt-4o", 0, 0) == 0


@pytest.mark.asyncio
class TestMockMode:
    async def test_mock_returns_a_complete_result(self):
        r = await mesh_client.chat("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert r["error"] is None
        assert r["content"]
        assert r["latency_ms"] > 0
        assert r["cost_usd"] >= 0

    async def test_mock_is_deterministic_per_model_and_input(self):
        msgs = [{"role": "user", "content": "same input\n[category:no_fabrication]"}]
        a = await mesh_client.chat("openai/gpt-4o", msgs)
        b = await mesh_client.chat("openai/gpt-4o", msgs)
        assert a["content"] == b["content"]

    async def test_mock_json_mode_returns_a_verdict(self):
        r = await mesh_client.chat("openai/gpt-4o", [{"role": "user", "content": "x"}],
                                   json_mode=True)
        import json
        assert "pass" in json.loads(r["content"])


class _StubTransport(httpx.AsyncBaseTransport):
    """Replays a scripted list of (status, json) responses, counting requests."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        status, payload = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(payload, Exception):
            raise payload
        return httpx.Response(status, json=payload, request=request)


@pytest.fixture
def live_client(monkeypatch):
    """Run chat() in live mode against a stub transport instead of MeshAPI."""
    def _install(script):
        transport = _StubTransport(script)
        stub = httpx.AsyncClient(transport=transport)
        monkeypatch.setattr(mesh_client, "MOCK", False)
        monkeypatch.setattr(mesh_client, "MESH_API_KEY", "test-key")
        monkeypatch.setattr(mesh_client, "_client", lambda: stub)
        monkeypatch.setattr(mesh_client, "RETRY_BASE_DELAY", 0.0)
        return transport
    return _install


OK_BODY = {"choices": [{"message": {"content": "hi"}}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.mark.asyncio
class TestLiveChat:
    MSGS = [{"role": "user", "content": "hi"}]

    async def test_successful_call_reports_tokens_and_cost(self, live_client):
        live_client([(200, OK_BODY)])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert r["error"] is None
        assert (r["input_tokens"], r["output_tokens"]) == (10, 5)
        assert r["cost_usd"] > 0

    async def test_gateway_reported_cost_wins_over_the_estimate(self, live_client):
        body = {**OK_BODY, "usage": {**OK_BODY["usage"], "cost_usd": 0.42}}
        live_client([(200, body)])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert r["cost_usd"] == 0.42

    async def test_rate_limit_is_retried_then_succeeds(self, live_client):
        t = live_client([(429, {"error": {"message": "slow down"}}), (200, OK_BODY)])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert r["error"] is None
        assert t.calls == 2

    async def test_server_error_is_retried(self, live_client):
        t = live_client([(503, {}), (503, {}), (200, OK_BODY)])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert r["error"] is None
        assert t.calls == 3

    async def test_retries_are_bounded(self, live_client, monkeypatch):
        monkeypatch.setattr(mesh_client, "MAX_ATTEMPTS", 3)
        t = live_client([(503, {})])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert t.calls == 3
        assert "after 3 attempts" in r["error"]

    async def test_timeouts_are_retried(self, live_client):
        t = live_client([(0, httpx.ConnectTimeout("timed out")), (200, OK_BODY)])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert r["error"] is None
        assert t.calls == 2

    async def test_client_error_is_not_retried(self, live_client):
        """A bad model id will never succeed — retrying just wastes the run."""
        t = live_client([(400, {"error": {"code": "bad_model", "message": "unknown model"}})])
        r = await mesh_client.chat("nope/nope", self.MSGS)
        assert t.calls == 1
        assert "unknown model" in r["error"]

    async def test_malformed_success_body_is_an_error_not_a_crash(self, live_client):
        live_client([(200, {"choices": []})])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert r["error"] and r["content"] == ""

    async def test_errors_never_raise(self, live_client):
        live_client([(500, {})])
        r = await mesh_client.chat("openai/gpt-4o-mini", self.MSGS)
        assert isinstance(r, dict) and r["content"] == "" and r["error"]


@pytest.mark.asyncio
class TestMockCoverage:
    """Mock mode is the offline demo path: every category must be winnable."""

    async def _judge_verdict(self, category, response):
        from backend import judge
        return await judge.judge(category, "some input", response)

    @pytest.mark.parametrize("category", sorted(mesh_client._MOCK_BEHAVIOR))
    async def test_good_answer_passes_its_category(self, category):
        """Regression: format_compliance, language_match and pii_refusal had no
        canned answers, so they fell through to a default the mock judge always
        rejected — three of eight categories could never pass offline."""
        good = mesh_client._MOCK_BEHAVIOR[category][0]
        assert (await self._judge_verdict(category, good))["pass"] is True

    @pytest.mark.parametrize("category", sorted(mesh_client._MOCK_BEHAVIOR))
    async def test_bad_answer_fails_its_category(self, category):
        bad = mesh_client._MOCK_BEHAVIOR[category][1]
        assert (await self._judge_verdict(category, bad))["pass"] is False

    async def test_every_builtin_category_has_a_canned_pair(self):
        from backend import judge
        assert set(judge.CATEGORIES) == set(mesh_client._MOCK_BEHAVIOR)

    async def test_custom_rubrics_are_winnable_too(self):
        from backend import judge
        good = mesh_client._MOCK_DEFAULT[0]
        v = await judge.judge("custom", "in", good, custom_rubric="PASS if helpful.")
        assert v["pass"] is True

    async def test_custom_rubric_bad_answer_fails(self):
        from backend import judge
        v = await judge.judge("custom", "in", mesh_client._MOCK_DEFAULT[1],
                              custom_rubric="PASS if helpful.")
        assert v["pass"] is False

    async def test_pairs_are_distinguishable(self):
        for category, (good, bad) in mesh_client._MOCK_BEHAVIOR.items():
            assert good != bad, category
            assert good[:60] not in bad, category
