from dataclasses import dataclass, field

import httpx


class LLMProvider:
    name = "base"

    async def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Offline deterministic provider: the heuristic pipeline runs instead."""

    name = "mock"

    async def complete(self, system: str, user: str) -> str:
        return ""


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.name = "openai"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._transport = transport

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


def get_provider() -> LLMProvider:
    from agent.config import settings

    if settings.llm_provider == "openai" and settings.llm_api_key:
        return OpenAICompatProvider(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    return MockProvider()


@dataclass
class StepLog:
    tool: str
    args: dict = field(default_factory=dict)
    summary: str = ""
    ok: bool = True
