import json

import httpx
import pytest

from agent.llm import MockProvider, OpenAICompatProvider, get_provider


def provider_with(handler) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "https://api.example.com/v1",
        "sk-test",
        "test-model",
        transport=httpx.MockTransport(handler),
    )


async def test_provider_posts_chat_payload_and_extracts_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = request.read()
        return httpx.Response(200, json={"choices": [{"message": {"content": "最终回答"}}]})

    provider = provider_with(handler)
    text = await provider.complete("系统提示", "用户问题")

    assert text == "最终回答"
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    body = json.loads(seen["body"])
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "系统提示"}
    assert body["messages"][1] == {"role": "user", "content": "用户问题"}


async def test_provider_http_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    provider = provider_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete("s", "u")


def test_get_provider_defaults_to_mock(monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "")
    assert isinstance(get_provider(), MockProvider)


def test_get_provider_openai_with_key(monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "sk-x")
    provider = get_provider()
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.base_url == settings.llm_base_url
    assert provider.model == settings.llm_model
