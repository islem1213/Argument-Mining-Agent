"""Thin, provider-agnostic structured-output client.

Works with any OpenAI-compatible endpoint (OpenAI, Mistral, Groq, Together, a
local Ollama or vLLM server). Responses are requested as JSON Schema structured
output derived from a Pydantic model; servers that reject ``json_schema`` fall
back to ``json_object``. Every response is validated, and a failed validation is
sent back to the model once with the error so it can repair its own JSON.

Configuration (environment variables):

* ``ARGMINE_LLM_MODEL``     model name (default ``gpt-4.1-mini``)
* ``ARGMINE_LLM_BASE_URL``  base URL of an OpenAI-compatible server
* ``ARGMINE_LLM_API_KEY``   API key (falls back to ``OPENAI_API_KEY``)
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ChatClient(Protocol):
    """The tiny subset of the OpenAI client this project relies on (easy to fake in tests)."""

    def complete(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("Install the LLM extra first: pip install -e '.[llm]'") from exc
        self.model = model or os.environ.get("ARGMINE_LLM_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key or os.environ.get("ARGMINE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=base_url or os.environ.get("ARGMINE_LLM_BASE_URL") or None,
        )
        self._schema_supported = True

    def complete(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self._schema_supported:
            try:
                response = self._client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}},
                )
                return response.choices[0].message.content or "{}"
            except Exception as exc:  # noqa: BLE001 - server-specific error classes
                if "response_format" not in str(exc) and "json_schema" not in str(exc):
                    raise
                self._schema_supported = False
        response = self._client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
        return response.choices[0].message.content or "{}"


_default: ChatClient | None = None


def default_client() -> ChatClient:
    global _default
    if _default is None:
        _default = OpenAICompatibleClient()
    return _default


def set_default_client(client: ChatClient | None) -> None:
    """Inject a client (e.g. a fake in tests, or a pre-configured provider)."""
    global _default
    _default = client


def structured(system: str, user: str, schema: type[T], client: ChatClient | None = None, retries: int = 1) -> T:
    """Ask for JSON matching ``schema``; on invalid output, feed the error back and retry."""
    client = client or default_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user + "\n\nReturn only JSON matching this schema:\n"
         + json.dumps(schema.model_json_schema())},
    ]
    last_error: Exception | None = None
    for _ in range(retries + 1):
        raw = client.complete(messages, schema)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That JSON was invalid: {exc.errors()[:3]}. Return corrected JSON only."}]
    raise RuntimeError(f"LLM output failed validation after {retries + 1} attempts") from last_error
