import asyncio

from backend import judge, llm_client, runner


def test_parse_verdict_variants():
    assert judge._parse_verdict('{"pass": true, "confidence": 0.9, "reason": "ok"}')["pass"] is True
    v = judge._parse_verdict('Sure! {"pass": false, "reason": "leaked"} done')
    assert v["pass"] is False and v["reason"] == "leaked"
    assert judge._parse_verdict("FAIL: it leaked")["confidence"] == 0.3


def test_recommend_prefers_pass_rate_then_cost():
    results = [
        {"model": "a", "pass": True, "cost_usd": 0.02, "latency_ms": 100},
        {"model": "a", "pass": True, "cost_usd": 0.02, "latency_ms": 100},
        {"model": "b", "pass": True, "cost_usd": 0.01, "latency_ms": 100},
        {"model": "b", "pass": False, "cost_usd": 0.01, "latency_ms": 100},
        {"model": "c", "pass": True, "cost_usd": 0.001, "latency_ms": 900},
        {"model": "c", "pass": True, "cost_usd": 0.001, "latency_ms": 900},
    ]
    assert runner._recommend(results, ["a", "b", "c"])["recommended"] == "c"


def test_apply_openrouter_catalog(monkeypatch):
    monkeypatch.setattr(llm_client, "PRICE_TABLE", dict(llm_client.PRICE_TABLE))
    monkeypatch.setattr(llm_client, "AVAILABLE_MODELS", ["openai/gpt-4o-mini", "gone/model", "auto"])
    monkeypatch.setattr(llm_client, "CATALOG_IDS", set())
    llm_client.apply_catalog([
        {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"},
         "architecture": {"output_modalities": ["text", "image"]}},
        {"id": "black-forest-labs/flux", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"output_modalities": ["image"]}},
    ])
    assert llm_client.PRICE_TABLE["openai/gpt-4o-mini"] == (0.15, 0.6)
    assert "openrouter/auto" not in llm_client.PRICE_TABLE  # -1 = dynamic pricing
    assert llm_client.AVAILABLE_MODELS == ["openai/gpt-4o-mini", "auto"]
    assert (llm_client.CATALOG_COUNT, llm_client.CHAT_COUNT) == (3, 2)
    assert llm_client.is_known("black-forest-labs/flux") and not llm_client.is_known("gone/model")


def test_parse_completion_prefers_reported_cost():
    data = {"choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0042}}
    r = llm_client.parse_completion("openai/gpt-4o-mini", data, 12)
    assert r["content"] == "hi" and r["cost_usd"] == 0.0042 and r["latency_ms"] == 12
    data = {"choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    r = llm_client.parse_completion("openai/gpt-4o-mini", data, 1)
    assert r["content"] == "" and r["cost_usd"] == llm_client.PRICE_TABLE["openai/gpt-4o-mini"][0]


def test_chat_sends_openrouter_request(monkeypatch):
    import httpx
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                                                   "cost": 0.001}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(llm_client.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llm_client, "MOCK", False)
    monkeypatch.setattr(llm_client, "API_KEY", "sk-or-test")
    r = asyncio.run(llm_client.chat("auto", [{"role": "user", "content": "x"}], json_mode=True))
    assert r["error"] is None and r["content"] == "ok"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-or-test"
    assert b'"model":"openrouter/auto"' in seen["body"].replace(b" ", b"")
    assert b"json_object" in seen["body"]


def test_chat_surfaces_provider_errors(monkeypatch):
    import httpx
    real_client = httpx.AsyncClient
    def handler(req):
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})
    monkeypatch.setattr(llm_client.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llm_client, "MOCK", False)
    r = asyncio.run(llm_client.chat("openai/gpt-4o-mini", [{"role": "user", "content": "x"}]))
    assert r["content"] == "" and "Insufficient credits" in r["error"]
