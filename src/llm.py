"""Model access with an automatic provider fallback. (AC-A11, REQ-N03)

Order: OpenRouter -> Groq -> LLMUnavailable. Callers treat LLMUnavailable as a
degradation signal and escalate; nothing in the pipeline crashes on a dead provider.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.config import Settings, get_settings

log = logging.getLogger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, httpx.TimeoutException)


class LLMUnavailable(RuntimeError):
    """Every provider failed, or the model is switched off."""


@dataclass
class Provider:
    name: str
    api_key: str
    base_url: str
    model: str


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    fallback_calls: int = 0
    provider_errors: list[str] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.stats = LLMStats()
        self._clients: dict[str, OpenAI] = {}
        self.providers = self._build_providers()

    def _build_providers(self) -> list[Provider]:
        s = self.settings
        providers: list[Provider] = []
        for name in s.configured_providers:
            if name == "openrouter":
                providers.append(Provider(name, s.openrouter_api_key, s.openrouter_base_url, s.model_name))
            elif name == "groq":
                providers.append(Provider(name, s.groq_api_key, s.groq_base_url, s.fallback_model_name))
        return providers

    @property
    def available(self) -> bool:
        return self.settings.enable_llm and bool(self.providers)

    def _client_for(self, provider: Provider) -> OpenAI:
        if provider.name not in self._clients:
            self._clients[provider.name] = OpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
            )
        return self._clients[provider.name]

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        json_mode: bool = False,
    ) -> str:
        if not self.settings.enable_llm:
            raise LLMUnavailable("ENABLE_LLM is false")
        if not self.providers:
            raise LLMUnavailable("no provider credentials configured")

        errors: list[str] = []
        for index, provider in enumerate(self.providers):
            for attempt in range(1, self.settings.llm_max_retries + 1):
                try:
                    self.stats.calls += 1
                    if index > 0:
                        self.stats.fallback_calls += 1
                    kwargs: dict[str, Any] = {
                        "model": provider.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = self._client_for(provider).chat.completions.create(**kwargs)
                    content = (response.choices[0].message.content or "").strip()
                    if not content:
                        raise ValueError("empty completion")
                    return content
                except _RETRYABLE as exc:
                    wait = min(2**attempt, 8)
                    log.warning("%s attempt %d failed (%s); retrying in %ss", provider.name, attempt, type(exc).__name__, wait)
                    if attempt < self.settings.llm_max_retries:
                        time.sleep(wait)
                    else:
                        errors.append(f"{provider.name}: {type(exc).__name__}")
                except APIStatusError as exc:
                    errors.append(f"{provider.name}: HTTP {exc.status_code}")
                    break  # 4xx will not improve on retry; move to the next provider
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                    break

        self.stats.failures += 1
        self.stats.provider_errors.extend(errors)
        raise LLMUnavailable("; ".join(errors) or "all providers failed")

    def complete_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
        raw = self.complete(system, user, json_mode=True, **kwargs)
        return parse_json_object(raw)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Small models fence their JSON or prepend prose; recover the object anyway."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ValueError(f"no JSON object in model output: {raw[:200]!r}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model returned JSON that is not an object")
    return parsed
