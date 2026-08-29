from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from veriloop.model import (
    ModelProtocolError,
    OpenAICompatibleModel,
    ProviderFatalError,
    ProviderRetryExhaustedError,
)
from veriloop.protocol import (
    FinishReason,
    Message,
    Role,
    ToolCall,
    ToolResult,
)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def provider_response(
    *,
    content: str | None = "done",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: Any = None,
) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def raw_tool_call(call_id: str, name: str, arguments: Any) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def model_for(
    outcomes: list[Any], *, sleeps: list[float] | None = None
) -> tuple[OpenAICompatibleModel, FakeClient]:
    client = FakeClient(outcomes)
    sleep_log = sleeps if sleeps is not None else []
    model = OpenAICompatibleModel(
        model="test-model",
        client=client,
        sleep=sleep_log.append,
        backoff_seconds=0.1,
    )
    return model, client


def test_plain_text_response_and_usage_are_converted() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    model, client = model_for([provider_response(content="answer", usage=usage)])

    response = model.complete([Message(Role.USER, "question")], [])

    assert response.text == "answer"
    assert response.tool_calls == ()
    assert response.finish_reason is FinishReason.STOP
    assert response.usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert client.completions.requests[0]["stream"] is False
    assert "tools" not in client.completions.requests[0]


def test_none_assistant_content_becomes_empty_text() -> None:
    model, _ = model_for([provider_response(content=None)])

    response = model.complete([], [])

    assert response.text == ""


def test_json_arguments_and_multiple_tool_calls_are_preserved() -> None:
    calls = [
        raw_tool_call("one", "first", '{"value": 1}'),
        raw_tool_call("two", "second", '{"enabled": true}'),
    ]
    model, _ = model_for(
        [provider_response(content=None, tool_calls=calls, finish_reason="tool_calls")]
    )

    response = model.complete([], [])

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls == (
        ToolCall(id="one", name="first", arguments={"value": 1}),
        ToolCall(id="two", name="second", arguments={"enabled": True}),
    )


@pytest.mark.parametrize("arguments", ["{bad json", "[1, 2, 3]", '"text"'])
def test_invalid_or_non_object_tool_arguments_are_protocol_errors(arguments: str) -> None:
    model, client = model_for(
        [provider_response(tool_calls=[raw_tool_call("bad", "tool", arguments)])]
    )

    with pytest.raises(ModelProtocolError):
        model.complete([], [])

    assert len(client.completions.requests) == 1


def test_retryable_error_then_success_uses_bounded_retry() -> None:
    sleeps: list[float] = []
    model, client = model_for(
        [TimeoutError("temporary"), provider_response(content="recovered")],
        sleeps=sleeps,
    )

    response = model.complete([], [])

    assert response.text == "recovered"
    assert len(client.completions.requests) == 2
    assert sleeps == [0.1]


def test_three_retryable_failures_raise_retry_exhausted() -> None:
    sleeps: list[float] = []
    model, client = model_for(
        [TimeoutError("one"), TimeoutError("two"), TimeoutError("three")],
        sleeps=sleeps,
    )

    with pytest.raises(ProviderRetryExhaustedError) as caught:
        model.complete([], [])

    assert caught.value.kind.value == "provider_retry_exhausted"
    assert len(client.completions.requests) == 3
    assert sleeps == [0.1, 0.2]


def test_default_sdk_client_limits_total_http_requests_to_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0
    created_clients: list[openai.OpenAI] = []
    real_openai_client = openai.OpenAI

    def fail_with_500(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            500,
            request=request,
            json={"error": {"message": "offline failure", "type": "server_error"}},
        )

    def build_offline_client(**kwargs: Any) -> openai.OpenAI:
        kwargs["base_url"] = "https://offline.invalid/v1"
        kwargs["http_client"] = httpx.Client(
            transport=httpx.MockTransport(fail_with_500)
        )
        client = real_openai_client(**kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", build_offline_client)
    model = OpenAICompatibleModel(
        model="offline-model",
        api_key="offline-placeholder",
        sleep=lambda _: None,
        backoff_seconds=0,
    )

    try:
        with pytest.raises(ProviderRetryExhaustedError):
            model.complete([], [])
    finally:
        created_clients[0].close()

    assert request_count == 3


def test_fatal_provider_error_is_not_retried() -> None:
    class BadRequest(Exception):
        status_code = 400

    sleeps: list[float] = []
    model, client = model_for([BadRequest("invalid model")], sleeps=sleeps)

    with pytest.raises(ProviderFatalError):
        model.complete([], [])

    assert len(client.completions.requests) == 1
    assert sleeps == []


def test_history_is_serialized_without_internal_or_provider_objects() -> None:
    model, client = model_for([provider_response()])
    call = ToolCall(id="call-1", name="lookup", arguments={"key": "x"})
    result = ToolResult(
        call_id="call-1",
        tool_name="lookup",
        ok=True,
        content="value",
    )
    messages = [
        Message(Role.SYSTEM, "system"),
        Message(Role.USER, "task"),
        Message(Role.ASSISTANT, "", tool_calls=(call,)),
        Message(Role.TOOL, "value", tool_result=result),
    ]

    model.complete(messages, [{"type": "function", "function": {"name": "lookup"}}])

    request = client.completions.requests[0]
    assert request["tools"] == [
        {"type": "function", "function": {"name": "lookup"}}
    ]
    assert request["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key": "x"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "value"},
    ]


@pytest.mark.parametrize(
    ("provider_reason", "expected"),
    [
        ("length", FinishReason.LENGTH),
        ("cancelled", FinishReason.CANCELLED),
        ("content_filter", FinishReason.ERROR),
    ],
)
def test_finish_reasons_are_normalized(
    provider_reason: str, expected: FinishReason
) -> None:
    model, _ = model_for([provider_response(finish_reason=provider_reason)])

    assert model.complete([], []).finish_reason is expected


def test_missing_choices_is_a_protocol_error() -> None:
    model, _ = model_for([SimpleNamespace(choices=[], usage=None)])

    with pytest.raises(ModelProtocolError, match="no choices"):
        model.complete([], [])
