from __future__ import annotations

import pytest

from tests.scripted_model import ScriptedModel
from veriloop.agent import AgentLoop
from veriloop.model import (
    ModelProtocolError,
    ProviderFatalError,
    ProviderRetryExhaustedError,
)
from veriloop.protocol import (
    AgentState,
    ErrorKind,
    FinishReason,
    ModelResponse,
    Role,
    ToolCall,
)
from veriloop.tools import ToolRegistry, ToolSpec


def response(
    text: str = "",
    *calls: ToolCall,
    finish_reason: FinishReason | None = None,
) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tuple(calls),
        finish_reason=finish_reason
        or (FinishReason.TOOL_CALLS if calls else FinishReason.STOP),
    )


def make_registry(events: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def read(arguments: dict[str, object]) -> str:
        if events is not None:
            events.append(f"read:{arguments['path']}")
        return "alpha"

    def transform(arguments: dict[str, object]) -> str:
        if events is not None:
            events.append(f"transform:{arguments['text']}")
        return str(arguments["text"]).upper()

    registry.register(
        ToolSpec(
            name="read",
            description="Test-only deterministic read",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read,
        )
    )
    registry.register(
        ToolSpec(
            name="transform",
            description="Test-only deterministic transform",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=transform,
        )
    )
    return registry


def tool_messages(messages: list | tuple) -> list:
    return [message for message in messages if message.role is Role.TOOL]


def test_direct_final_text_is_completed_unverified() -> None:
    model = ScriptedModel([response("final answer")])

    result = AgentLoop(model, ToolRegistry()).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "final answer"
    assert result.step_count == 1
    assert result.tool_call_count == 0
    assert [message.role for message in result.history] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
    ]
    assert model.call_count == 1


def test_single_tool_call_result_is_visible_to_next_model_turn() -> None:
    call = ToolCall(id="read-one", name="read", arguments={"path": "a.txt"})
    model = ScriptedModel([response("", call), response("got alpha")])

    result = AgentLoop(model, make_registry()).run("read it")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.step_count == 2
    assert result.tool_call_count == 1
    assert [message.role for message in result.history] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    second_messages = model.calls[1][0]
    seen = tool_messages(second_messages)
    assert len(seen) == 1
    assert seen[0].tool_result.call_id == "read-one"
    assert seen[0].content == "alpha"


def test_complete_three_turn_deterministic_trajectory() -> None:
    events: list[str] = []
    read_call = ToolCall(id="call-read", name="read", arguments={"path": "fixed"})
    transform_call = ToolCall(
        id="call-transform",
        name="transform",
        arguments={"text": "alpha"},
    )
    model = ScriptedModel(
        [
            response("", read_call),
            response("", transform_call),
            response("final: ALPHA"),
        ]
    )

    result = AgentLoop(model, make_registry(events)).run("three turns")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "final: ALPHA"
    assert result.step_count == 3
    assert result.tool_call_count == 2
    assert model.call_count == 3
    assert events == ["read:fixed", "transform:alpha"]
    assert [message.role for message in result.history] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    results = [message.tool_result for message in tool_messages(result.history)]
    assert [item.call_id for item in results] == ["call-read", "call-transform"]
    assert any(
        message.tool_result and message.tool_result.call_id == "call-read"
        for message in model.calls[1][0]
    )
    assert any(
        message.tool_result and message.tool_result.call_id == "call-transform"
        for message in model.calls[2][0]
    )


def test_multiple_calls_in_one_response_execute_serially_in_order() -> None:
    events: list[str] = []
    calls = (
        ToolCall(id="first", name="read", arguments={"path": "one"}),
        ToolCall(id="second", name="transform", arguments={"text": "two"}),
        ToolCall(id="third", name="read", arguments={"path": "three"}),
    )
    model = ScriptedModel([response("", *calls), response("done")])

    result = AgentLoop(model, make_registry(events)).run("ordered tools")

    assert events == ["read:one", "transform:two", "read:three"]
    assert result.tool_call_count == 3
    results = tool_messages(result.history)
    assert [item.tool_result.call_id for item in results] == [
        "first",
        "second",
        "third",
    ]
    assert len(results) == len(calls)
    assert [message.role for message in model.calls[1][0][-3:]] == [
        Role.TOOL,
        Role.TOOL,
        Role.TOOL,
    ]


def test_unknown_tool_error_is_seen_and_model_recovers() -> None:
    missing = ToolCall(id="bad-call", name="missing", arguments={})
    valid = ToolCall(id="good-call", name="read", arguments={"path": "fixed"})
    model = ScriptedModel(
        [response("", missing), response("", valid), response("recovered")]
    )

    result = AgentLoop(model, make_registry()).run("recover")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "recovered"
    assert result.tool_call_count == 2
    first_error = tool_messages(model.calls[1][0])[0].tool_result
    assert first_error.ok is False
    assert first_error.error_kind is ErrorKind.UNKNOWN_TOOL
    assert first_error.call_id == "bad-call"
    all_results = [message.tool_result for message in tool_messages(result.history)]
    assert [item.call_id for item in all_results] == ["bad-call", "good-call"]
    assert all_results[1].ok is True


@pytest.mark.parametrize(
    ("bad_call", "expected_kind"),
    [
        (
            ToolCall(id="invalid", name="read", arguments={}),
            ErrorKind.INVALID_ARGUMENTS,
        ),
        (
            ToolCall(id="explode", name="explode", arguments={}),
            ErrorKind.TOOL_ERROR,
        ),
    ],
)
def test_tool_errors_enter_next_request_without_ending_loop(
    bad_call: ToolCall, expected_kind: ErrorKind
) -> None:
    registry = make_registry()
    registry.register(
        ToolSpec(
            name="explode",
            description="Fail deterministically",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda arguments: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )
    model = ScriptedModel([response("", bad_call), response("corrected")])

    result = AgentLoop(model, registry).run("correct an error")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    seen = tool_messages(model.calls[1][0])
    assert len(seen) == 1
    assert seen[0].tool_result.error_kind is expected_kind
    assert seen[0].tool_result.call_id == bad_call.id


def test_max_steps_stops_exactly_without_extra_model_call() -> None:
    call = ToolCall(id="again", name="read", arguments={"path": "x"})
    model = ScriptedModel([response("", call), response("", call)])

    result = AgentLoop(model, make_registry(), max_steps=2).run("never finish")

    assert result.state is AgentState.MAX_STEPS
    assert result.error.kind is ErrorKind.MAX_STEPS
    assert result.step_count == 2
    assert result.tool_call_count == 2
    assert model.call_count == 2


def test_zero_max_steps_never_calls_model() -> None:
    model = ScriptedModel([])

    result = AgentLoop(model, ToolRegistry(), max_steps=0).run("no budget")

    assert result.state is AgentState.MAX_STEPS
    assert result.step_count == 0
    assert model.call_count == 0


@pytest.mark.parametrize(
    ("error", "expected_kind", "retryable"),
    [
        (
            ProviderFatalError("authentication failed"),
            ErrorKind.PROVIDER_FATAL_ERROR,
            False,
        ),
        (
            ProviderRetryExhaustedError("temporary failures exhausted"),
            ErrorKind.PROVIDER_RETRY_EXHAUSTED,
            True,
        ),
        (
            ModelProtocolError("bad arguments"),
            ErrorKind.MODEL_PROTOCOL_ERROR,
            False,
        ),
    ],
)
def test_model_errors_end_with_explicit_failure(
    error: Exception, expected_kind: ErrorKind, retryable: bool
) -> None:
    model = ScriptedModel([error])

    result = AgentLoop(model, ToolRegistry()).run("fail")

    assert result.state is AgentState.FAILED
    assert result.error.kind is expected_kind
    assert result.error.retryable is retryable
    assert result.step_count == 1
    assert [message.role for message in result.history] == [Role.SYSTEM, Role.USER]


def test_keyboard_interrupt_returns_cancelled() -> None:
    model = ScriptedModel([KeyboardInterrupt()])

    result = AgentLoop(model, ToolRegistry()).run("cancel")

    assert result.state is AgentState.CANCELLED
    assert result.error.kind is ErrorKind.CANCELLED
    assert result.step_count == 1


def test_each_model_request_gets_tools_and_an_isolated_history_list() -> None:
    call = ToolCall(id="one", name="read", arguments={"path": "x"})
    model = ScriptedModel([response("", call), response("done")])
    registry = make_registry()

    AgentLoop(model, registry).run("inspect")

    assert model.calls[0][1] == registry.schemas()
    assert model.calls[1][1] == registry.schemas()
    assert len(model.calls[0][0]) == 2
    assert len(model.calls[1][0]) == 4
