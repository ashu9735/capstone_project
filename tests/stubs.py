"""Test doubles. The suite must never need an API key or a network connection."""

from __future__ import annotations

import json

from src.llm import LLMClient, LLMUnavailable


class StubLLM(LLMClient):
    """Returns canned answers in place of a model provider."""

    def __init__(self, settings, intent="authentication_failure", confidence=0.94, fail=False):
        super().__init__(settings)
        self._intent, self._confidence, self._fail = intent, confidence, fail
        self.providers = [object()]  # type: ignore[list-item]
        self.settings.enable_llm = True

    @property
    def available(self) -> bool:
        return True

    def complete(self, system, user, temperature=0.0, max_tokens=700, json_mode=False):
        if self._fail:
            raise LLMUnavailable("stubbed provider outage")
        if json_mode:
            return json.dumps(
                {"intent": self._intent, "urgency": "medium",
                 "confidence": self._confidence, "rationale": "stub"}
            )
        return (
            "Thanks for getting in touch. A locked account unlocks automatically after "
            "thirty minutes, and the security page shows a red banner while it is locked "
            "[DOC-AUTH-001]. If the CLI is still failing, sign out and back in to discard "
            "the cached token."
        )
