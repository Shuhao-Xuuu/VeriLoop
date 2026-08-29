"""A deterministic, network-free ModelClient test double."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from veriloop.protocol import Message, ModelResponse


class ScriptedModel:
    def __init__(self, script: Iterable[ModelResponse | BaseException]) -> None:
        self._script = list(script)
        self.calls: list[tuple[list[Message], list[dict[str, Any]]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append((list(messages), deepcopy(tools)))
        index = self.call_count - 1
        if index >= len(self._script):
            raise AssertionError("ScriptedModel script exhausted")

        outcome = self._script[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
