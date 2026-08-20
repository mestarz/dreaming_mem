"""Small LLM boundary plus optional OpenAI-compatible HTTP implementation."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str:
        """Return the model's text response for one prompt."""


@dataclass(slots=True)
class CallableLLM:
    """Adapt a sync or async ``callable(prompt) -> str``."""

    function: Callable[[str], str | Awaitable[str]]

    async def complete(self, prompt: str) -> str:
        result = self.function(prompt)
        if isawaitable(result):
            result = await result
        if not isinstance(result, str):
            raise TypeError("LLM callable must return a string")
        return result


@dataclass(slots=True)
class InvokeLLMAdapter:
    """Adapt Agent Memory-style objects exposing ``await invoke(messages=...)``."""

    llm: Any

    async def complete(self, prompt: str) -> str:
        response = await self.llm.invoke(messages=[{"role": "user", "content": prompt}])
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            raise TypeError("invoke() response must be text or expose a text .content")
        return content


@dataclass(slots=True)
class OpenAICompatibleLLM:
    """Dependency-free adapter for an OpenAI-compatible chat-completions API."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    top_p: float | None = None

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._complete_blocking, prompt)

    def _complete_blocking(self, prompt: str) -> str:
        base = self.base_url.rstrip("/")
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response has no choices[0].message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("LLM response content is not text")
        return content


@dataclass(slots=True)
class OllamaChatLLM:
    """Ollama-native chat adapter with explicit thinking and context controls."""

    base_url: str
    model: str
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    top_p: float | None = None
    think: bool = False
    num_ctx: int | None = None
    num_predict: int | None = None
    format_schema: str | dict[str, Any] | None = None

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._complete_blocking, prompt)

    def _complete_blocking(self, prompt: str) -> str:
        base = self.base_url.rstrip("/")
        endpoint = base if base.endswith("/api/chat") else f"{base}/api/chat"
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.think,
            "options": options,
        }
        if self.format_schema is not None:
            payload["format"] = self.format_schema
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        try:
            content = response_data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama response has no message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("Ollama response content is not text")
        if response_data.get("done_reason") == "length":
            raise RuntimeError("Ollama stopped at num_predict before completing the response")
        return content
