"""Tool registration, schema exposure, validation, and safe execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Callable

from .protocol import ErrorKind, ToolCall, ToolResult


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    mutates_workspace: bool = False


class ToolRegistry:
    """The only component that validates and invokes registered handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": deepcopy(spec.input_schema),
                },
            }
            for spec in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return _tool_failure(
                call,
                ErrorKind.UNKNOWN_TOOL,
                f"unknown tool: {call.name}",
            )

        if not isinstance(call.arguments, dict):
            return _tool_failure(
                call,
                ErrorKind.INVALID_ARGUMENTS,
                "tool arguments must be an object",
            )

        validation_error = _validate_value(call.arguments, spec.input_schema, "arguments")
        if validation_error is not None:
            return _tool_failure(call, ErrorKind.INVALID_ARGUMENTS, validation_error)

        try:
            value = spec.handler(call.arguments)
        except Exception as exc:
            return _tool_failure(
                call,
                ErrorKind.TOOL_ERROR,
                f"tool raised {type(exc).__name__}: {exc}",
            )

        return ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=_stringify_result(value),
        )


def _tool_failure(call: ToolCall, kind: ErrorKind, message: str) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=message,
        error_kind=kind,
        retryable=False,
    )


def _stringify_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> str | None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        return f"{path} must be of type {expected_type}"

    if expected_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for name in required:
            if name not in value:
                return f"{path}.{name} is required"

        additional = schema.get("additionalProperties", False)
        for name, item in value.items():
            item_schema = properties.get(name)
            if item_schema is None:
                if additional is False:
                    return f"{path}.{name} is not allowed"
                if isinstance(additional, dict):
                    error = _validate_value(item, additional, f"{path}.{name}")
                    if error is not None:
                        return error
                continue

            error = _validate_value(item, item_schema, f"{path}.{name}")
            if error is not None:
                return error

    return None


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    return False
