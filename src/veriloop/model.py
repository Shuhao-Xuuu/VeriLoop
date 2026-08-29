"""Model boundary and a non-streaming OpenAI-compatible adapter."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Protocol

import openai

from .protocol import ErrorKind, FinishReason, Message, ModelResponse, Role, ToolCall


class ModelClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return one provider-independent assistant turn."""


class ModelClientError(RuntimeError):
    def __init__(self, message: str, kind: ErrorKind, retryable: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class ModelProtocolError(ModelClientError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorKind.MODEL_PROTOCOL_ERROR)


class ProviderRetryableError(ModelClientError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorKind.PROVIDER_RETRYABLE_ERROR, retryable=True)


class ProviderFatalError(ModelClientError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorKind.PROVIDER_FATAL_ERROR)


class ProviderRetryExhaustedError(ModelClientError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorKind.PROVIDER_RETRY_EXHAUSTED, retryable=True)


class OpenAICompatibleModel:
    """Translate between OpenAI Chat Completions and VeriLoop's protocol."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff_seconds: float = 0.25,
    ) -> None:
        if client is None:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self._model = model
        self._sleep = sleep
        self._backoff_seconds = backoff_seconds

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_provider_message(message) for message in messages],
            "stream": False,
        }
        if tools:
            request["tools"] = tools

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**request)
            except KeyboardInterrupt:
                raise
            except ProviderFatalError:
                raise
            except Exception as exc:
                if not _is_retryable_provider_error(exc):
                    raise ProviderFatalError(_describe_provider_error(exc)) from exc
                if attempt == 2:
                    raise ProviderRetryExhaustedError(
                        f"provider retry exhausted after 3 requests: "
                        f"{_describe_provider_error(exc)}"
                    ) from exc
                self._sleep(self._backoff_seconds * (2**attempt))
                continue

            return _parse_provider_response(response)

        raise AssertionError("unreachable retry loop")


def _to_provider_message(message: Message) -> dict[str, Any]:
    if message.role in (Role.SYSTEM, Role.USER):
        return {"role": message.role.value, "content": message.content}

    if message.role is Role.ASSISTANT:
        payload: dict[str, Any] = {
            "role": Role.ASSISTANT.value,
            "content": message.content,
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    if message.role is Role.TOOL:
        if message.tool_result is None:
            raise ModelProtocolError("tool message is missing its ToolResult")
        return {
            "role": Role.TOOL.value,
            "tool_call_id": message.tool_result.call_id,
            "content": message.content,
        }

    raise ModelProtocolError(f"unsupported message role: {message.role}")


def _parse_provider_response(response: Any) -> ModelResponse:
    try:
        choices = _field(response, "choices")
        if not choices:
            raise ModelProtocolError("provider response has no choices")
        choice = choices[0]
        provider_message = _field(choice, "message")
        if provider_message is None:
            raise ModelProtocolError("provider choice has no message")

        content = _field(provider_message, "content")
        text = "" if content is None else str(content)
        raw_calls = _field(provider_message, "tool_calls") or []
        tool_calls = tuple(_parse_tool_call(raw_call) for raw_call in raw_calls)
        finish_reason = _map_finish_reason(_field(choice, "finish_reason"))
        usage = _parse_usage(_field(response, "usage"))
    except ModelProtocolError:
        raise
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelProtocolError(f"malformed provider response: {exc}") from exc

    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


def _parse_tool_call(raw_call: Any) -> ToolCall:
    call_id = _field(raw_call, "id")
    function = _field(raw_call, "function")
    name = _field(function, "name") if function is not None else None
    raw_arguments = _field(function, "arguments") if function is not None else None

    if not isinstance(call_id, str) or not call_id:
        raise ModelProtocolError("provider tool call has no valid id")
    if not isinstance(name, str) or not name:
        raise ModelProtocolError(f"provider tool call {call_id} has no valid name")

    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ModelProtocolError(
                f"provider tool call {call_id} contains invalid JSON arguments"
            ) from exc
    elif isinstance(raw_arguments, dict):
        arguments = dict(raw_arguments)
    else:
        raise ModelProtocolError(
            f"provider tool call {call_id} arguments must be JSON"
        )

    if not isinstance(arguments, dict):
        raise ModelProtocolError(
            f"provider tool call {call_id} arguments must decode to an object"
        )

    return ToolCall(id=call_id, name=name, arguments=arguments)


def _map_finish_reason(value: Any) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "length": FinishReason.LENGTH,
        "cancelled": FinishReason.CANCELLED,
    }
    return mapping.get(value, FinishReason.ERROR)


def _parse_usage(raw_usage: Any) -> dict[str, int]:
    if raw_usage is None:
        return {}

    usage: dict[str, int] = {}
    input_tokens = _first_int(raw_usage, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(raw_usage, "completion_tokens", "output_tokens")
    total_tokens = _first_int(raw_usage, "total_tokens")
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage


def _first_int(value: Any, *names: str) -> int | None:
    for name in names:
        item = _field(value, name)
        if isinstance(item, int) and not isinstance(item, bool):
            return item
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _describe_provider_error(exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _is_retryable_provider_error(exc: Exception) -> bool:
    if isinstance(exc, ProviderRetryableError):
        return True

    retryable_types = tuple(
        error_type
        for error_type in (
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "InternalServerError", None),
        )
        if isinstance(error_type, type)
    )
    if retryable_types and isinstance(exc, retryable_types):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and (
        status_code in {408, 409, 429} or status_code >= 500
    )
