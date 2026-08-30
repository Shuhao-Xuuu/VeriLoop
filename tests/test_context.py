from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import pytest

from tests.scripted_model import ScriptedModel
from veriloop.agent import DEFAULT_SYSTEM_PROMPT, AgentLoop
from veriloop.context import CONTEXT_TRUNCATION_MARKER, ContextPolicy
from veriloop.model import _to_provider_message
from veriloop.protocol import (
    AgentState,
    ErrorKind,
    FinishReason,
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
)
from veriloop.tools import ToolRegistry, ToolSpec


def initial_messages(
    *,
    system: str = "system contract",
    user: str = "original user task",
) -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content=system),
        Message(role=Role.USER, content=user),
    ]


def tool_group(
    prefix: str,
    *,
    call_count: int = 1,
    result_content: str = "ok",
    verification_failure: bool = False,
    verification_success: bool = False,
    arguments: dict[str, object] | None = None,
) -> list[Message]:
    tool_name = (
        "complete_task"
        if verification_failure or verification_success
        else "read"
    )
    calls = tuple(
        ToolCall(
            id=f"{prefix}-{index}",
            name=tool_name,
            arguments=deepcopy(
                arguments
                if arguments is not None
                else {"path": f"{prefix}-{index}.txt"}
            ),
        )
        for index in range(call_count)
    )
    messages = [Message(role=Role.ASSISTANT, content="", tool_calls=calls)]
    for call in calls:
        result = ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=not verification_failure,
            content=result_content,
            error_kind=(
                ErrorKind.VERIFICATION_FAILED if verification_failure else None
            ),
            retryable=verification_failure,
            metadata={
                "verification": verification_failure or verification_success,
                "verified": (
                    False
                    if verification_failure
                    else (True if verification_success else None)
                ),
                "nested": {"seen": True},
            },
        )
        messages.append(
            Message(
                role=Role.TOOL,
                content=result.content,
                tool_result=result,
            )
        )
    return messages


def message_call_ids(messages: Sequence[Message]) -> list[str]:
    return [
        call.id
        for message in messages
        for call in message.tool_calls
    ]


def result_call_ids(messages: Sequence[Message]) -> list[str]:
    return [
        message.tool_result.call_id
        for message in messages
        if message.tool_result is not None
    ]


def assert_provider_legal(messages: Sequence[Message]) -> None:
    assert len(messages) >= 2
    assert messages[0].role is Role.SYSTEM
    assert messages[1].role is Role.USER
    index = 2
    while index < len(messages):
        message = messages[index]
        assert message.role is not Role.TOOL
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            index += 1
            continue
        expected = [call.id for call in message.tool_calls]
        assert len(expected) == len(set(expected))
        seen: list[str] = []
        index += 1
        while index < len(messages) and messages[index].role is Role.TOOL:
            result = messages[index].tool_result
            assert result is not None
            seen.append(result.call_id)
            index += 1
        assert len(seen) == len(set(seen))
        assert seen == expected


def response(*calls: ToolCall, text: str = "") -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tuple(calls),
        finish_reason=(
            FinishReason.TOOL_CALLS if calls else FinishReason.STOP
        ),
    )


def test_small_history_is_not_pruned_or_truncated() -> None:
    history = [*initial_messages(), *tool_group("only")]
    policy = ContextPolicy()

    projected = policy.project(history)

    assert projected == history
    assert projected is not history
    assert policy.estimate_chars(projected) == policy.estimate_chars(history)


def test_default_system_anchor_contains_host_verification_contract() -> None:
    projected = ContextPolicy().project(
        [
            Message(role=Role.SYSTEM, content=DEFAULT_SYSTEM_PROMPT),
            Message(role=Role.USER, content="task"),
        ]
    )

    assert "complete_task" in projected[0].content
    assert "Verification Gate" in projected[0].content
    assert "only the host" in projected[0].content
    assert "VERIFIED" in projected[0].content


def test_oldest_complete_group_is_removed_first() -> None:
    anchors = initial_messages()
    first = tool_group("first", result_content="a" * 80)
    second = tool_group("second", result_content="b" * 80)
    third = tool_group("third", result_content="c" * 80)
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=2)
    limit = sizing.estimate_chars([*anchors, *second, *third])
    policy = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=2,
    )

    projected = policy.project([*anchors, *first, *second, *third])

    assert message_call_ids(projected) == ["second-0", "third-0"]
    assert result_call_ids(projected) == ["second-0", "third-0"]
    assert projected[:2] == anchors
    assert_provider_legal(projected)


def test_multi_call_group_is_removed_as_one_atomic_unit() -> None:
    anchors = initial_messages()
    old_group = tool_group("old", call_count=3, result_content="old" * 40)
    recent = tool_group("recent", result_content="recent" * 20)
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=1)
    limit = sizing.estimate_chars([*anchors, *recent])

    projected = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=1,
    ).project([*anchors, *old_group, *recent])

    assert not any(call_id.startswith("old-") for call_id in message_call_ids(projected))
    assert not any(call_id.startswith("old-") for call_id in result_call_ids(projected))
    assert_provider_legal(projected)


def test_retained_multi_call_group_keeps_every_tool_result() -> None:
    anchors = initial_messages()
    old_group = tool_group("old", result_content="old" * 50)
    recent = tool_group("multi", call_count=3, result_content="new" * 20)
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=1)
    limit = sizing.estimate_chars([*anchors, *recent])

    projected = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=1,
    ).project([*anchors, *old_group, *recent])

    assert message_call_ids(projected) == ["multi-0", "multi-1", "multi-2"]
    assert result_call_ids(projected) == ["multi-0", "multi-1", "multi-2"]
    assert_provider_legal(projected)


def test_latest_verification_failure_group_survives_pruning() -> None:
    anchors = initial_messages()
    older_failure = tool_group(
        "failure-old",
        result_content="old failure" * 20,
        verification_failure=True,
    )
    latest_failure = tool_group(
        "failure-latest",
        result_content="latest failure" * 20,
        verification_failure=True,
    )
    recent = tool_group("recent", result_content="recent" * 20)
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=1)
    limit = sizing.estimate_chars([*anchors, *latest_failure, *recent])

    projected = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=1,
    ).project([*anchors, *older_failure, *latest_failure, *recent])

    assert message_call_ids(projected) == ["failure-latest-0", "recent-0"]
    assert result_call_ids(projected) == ["failure-latest-0", "recent-0"]
    assert_provider_legal(projected)


def test_verified_completion_is_not_pinned_as_a_failure() -> None:
    anchors = initial_messages()
    verified = tool_group(
        "verified",
        result_content="verified evidence" * 20,
        verification_success=True,
    )
    recent = tool_group("recent", result_content="recent" * 20)
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=1)
    limit = sizing.estimate_chars([*anchors, *recent])

    projected = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=1,
    ).project([*anchors, *verified, *recent])

    assert message_call_ids(projected) == ["recent-0"]
    assert_provider_legal(projected)


def test_recent_complete_groups_are_retained_in_order() -> None:
    anchors = initial_messages()
    groups = [
        tool_group(f"group-{index}", result_content=str(index) * 100)
        for index in range(4)
    ]
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=2)
    limit = sizing.estimate_chars([*anchors, *groups[2], *groups[3]])
    projected = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=2,
    ).project([*anchors, *groups[0], *groups[1], *groups[2], *groups[3]])

    assert message_call_ids(projected) == ["group-2-0", "group-3-0"]
    assert_provider_legal(projected)


def test_oversized_permanent_anchors_use_explicit_markers() -> None:
    history = initial_messages(system="s" * 1000, user="u" * 1000)
    policy = ContextPolicy(context_chars_soft_limit=80, recent_group_count=1)

    projected = policy.project(history)

    assert [message.role for message in projected] == [Role.SYSTEM, Role.USER]
    assert all(CONTEXT_TRUNCATION_MARKER in message.content for message in projected)
    assert policy.estimate_chars(projected) <= 80


def test_oversized_tool_result_is_bounded_without_breaking_pairing() -> None:
    history = [
        *initial_messages(system="s", user="u"),
        *tool_group("failure", result_content="failure evidence " * 200),
    ]
    policy = ContextPolicy(context_chars_soft_limit=180, recent_group_count=1)

    projected = policy.project(history)

    tool_message = next(message for message in projected if message.role is Role.TOOL)
    assert CONTEXT_TRUNCATION_MARKER in tool_message.content
    assert tool_message.tool_result is not None
    assert tool_message.tool_result.content == tool_message.content
    assert policy.estimate_chars(projected) <= 180
    assert_provider_legal(projected)


def test_oversized_nested_tool_arguments_are_cloned_and_bounded() -> None:
    arguments = {"outer": {"content": "x" * 2000, "mode": "overwrite"}}
    history = [
        *initial_messages(system="s", user="u"),
        *tool_group("write", arguments=arguments),
    ]
    policy = ContextPolicy(context_chars_soft_limit=220, recent_group_count=1)

    projected = policy.project(history)

    projected_arguments = projected[2].tool_calls[0].arguments
    assert projected_arguments["outer"]["mode"] == "overwrite"
    assert CONTEXT_TRUNCATION_MARKER in projected_arguments["outer"]["content"]
    assert policy.estimate_chars(projected) <= 220
    assert_provider_legal(projected)


def test_projection_is_deterministic_and_does_not_mutate_canonical_history() -> None:
    history = [
        *initial_messages(system="s", user="u"),
        *tool_group(
            "clone",
            arguments={"nested": {"content": "x" * 1000}},
            result_content="result" * 200,
        ),
    ]
    canonical_snapshot = deepcopy(history)
    policy = ContextPolicy(context_chars_soft_limit=180, recent_group_count=1)

    first = policy.project(history)
    second = policy.project(history)

    assert first == second
    assert history == canonical_snapshot
    first[2].tool_calls[0].arguments["nested"]["content"] = "changed"
    first[-1].tool_result.metadata["nested"]["seen"] = False
    assert history == canonical_snapshot
    assert second == policy.project(history)
    assert first != second


@pytest.mark.parametrize(
    "arguments",
    [
        {"values": list(range(1000))},
        {"content": ('\\"\n' * 1000)},
    ],
)
def test_non_string_or_escaped_arguments_have_a_bounded_json_projection(
    arguments: dict[str, object],
) -> None:
    history = [
        *initial_messages(system="s", user="u"),
        *tool_group("arguments", arguments=arguments),
    ]
    policy = ContextPolicy(context_chars_soft_limit=180, recent_group_count=1)

    projected = policy.project(history)

    projected_arguments = projected[2].tool_calls[0].arguments
    assert policy.estimate_chars(projected) <= 180
    assert CONTEXT_TRUNCATION_MARKER in str(projected_arguments)
    assert_provider_legal(projected)


@pytest.mark.parametrize(
    "malformed_tail",
    [
        [
            Message(
                role=Role.TOOL,
                content="orphan",
                tool_result=ToolResult(
                    call_id="orphan",
                    tool_name="read",
                    ok=True,
                    content="orphan",
                ),
            )
        ],
        [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(
                    ToolCall(id="missing", name="read", arguments={"path": "x"}),
                ),
            )
        ],
        [Message(role=Role.USER, content="unexpected follow-up")],
        [
            *tool_group("swapped", call_count=2)[0:1],
            *reversed(tool_group("swapped", call_count=2)[1:]),
        ],
    ],
)
def test_malformed_groups_are_never_projected(
    malformed_tail: list[Message],
) -> None:
    with pytest.raises(ValueError, match="orphan|complete group|assistant groups"):
        ContextPolicy().project([*initial_messages(), *malformed_tail])


def test_projected_history_has_valid_provider_message_order() -> None:
    anchors = initial_messages()
    history = [
        *anchors,
        *tool_group("drop", call_count=2, result_content="d" * 100),
        *tool_group("keep", call_count=2, result_content="k" * 100),
    ]
    sizing = ContextPolicy(context_chars_soft_limit=10_000, recent_group_count=1)
    limit = sizing.estimate_chars(
        [*anchors, *tool_group("keep", call_count=2, result_content="k" * 100)]
    )

    projected = ContextPolicy(
        context_chars_soft_limit=limit,
        recent_group_count=1,
    ).project(history)

    assert_provider_legal(projected)
    assert [message.role for message in projected] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.TOOL,
    ]
    provider_messages = [_to_provider_message(message) for message in projected]
    assert [message["role"] for message in provider_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [
        message["tool_call_id"]
        for message in provider_messages
        if message["role"] == "tool"
    ] == ["keep-0", "keep-1"]


def test_agent_loop_projects_model_requests_but_preserves_canonical_history() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="observe",
            description="Return bounded deterministic test evidence",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=lambda arguments: str(arguments["value"]) * 120,
        )
    )
    first_call = ToolCall(id="first", name="observe", arguments={"value": "a"})
    second_call = ToolCall(id="second", name="observe", arguments={"value": "b"})
    model = ScriptedModel(
        [
            response(first_call),
            response(second_call),
            response(text="done"),
        ]
    )
    policy = ContextPolicy(context_chars_soft_limit=200, recent_group_count=1)

    result = AgentLoop(
        model,
        registry,
        system_prompt="system",
        context_policy=policy,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "done"
    assert model.call_count == 3
    assert message_call_ids(model.calls[2][0]) == ["second"]
    assert result_call_ids(model.calls[2][0]) == ["second"]
    assert message_call_ids(result.history) == ["first", "second"]
    assert result_call_ids(result.history) == ["first", "second"]
    assert_provider_legal(model.calls[2][0])


def test_agent_loop_small_request_snapshots_remain_complete() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read",
            description="Return deterministic evidence",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda arguments: "alpha",
        )
    )
    call = ToolCall(id="read-one", name="read", arguments={"path": "a.txt"})
    model = ScriptedModel([response(call), response(text="done")])

    result = AgentLoop(model, registry, system_prompt="system").run("task")

    assert model.call_count == 2
    assert [message.role for message in model.calls[0][0]] == [
        Role.SYSTEM,
        Role.USER,
    ]
    assert [message.role for message in model.calls[1][0]] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
    ]
    assert result_call_ids(model.calls[1][0]) == ["read-one"]
    assert result.history[:4] == tuple(model.calls[1][0])


def test_structural_minimum_can_exceed_the_soft_limit_without_splitting_group() -> None:
    history = [
        *initial_messages(system="s", user="u"),
        *tool_group("wide", call_count=8, result_content="x" * 100),
    ]
    policy = ContextPolicy(
        context_chars_soft_limit=len(CONTEXT_TRUNCATION_MARKER),
        recent_group_count=1,
    )

    projected = policy.project(history)

    assert policy.estimate_chars(projected) > policy.context_chars_soft_limit
    assert message_call_ids(projected) == result_call_ids(projected)
    assert_provider_legal(projected)


def test_context_projection_failure_does_not_consume_a_model_step() -> None:
    class FailingContextPolicy:
        def project(self, history: Sequence[Message]) -> list[Message]:
            raise ValueError("broken projection")

    model = ScriptedModel([])

    result = AgentLoop(
        model,
        ToolRegistry(),
        context_policy=FailingContextPolicy(),
    ).run("task")

    assert result.state is AgentState.FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.INTERNAL_ERROR
    assert "context projection" in result.error.message
    assert result.step_count == 0
    assert model.call_count == 0


def test_duplicate_tool_call_ids_fail_before_tool_execution() -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="observe",
            description="Record execution",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=lambda arguments: executed.append(str(arguments["value"])),
        )
    )
    calls = (
        ToolCall(id="duplicate", name="observe", arguments={"value": "one"}),
        ToolCall(id="duplicate", name="observe", arguments={"value": "two"}),
    )
    model = ScriptedModel([response(*calls)])

    result = AgentLoop(model, registry).run("task")

    assert result.state is AgentState.FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.MODEL_PROTOCOL_ERROR
    assert result.step_count == 1
    assert result.tool_call_count == 0
    assert executed == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_chars_soft_limit": True}, "context_chars_soft_limit"),
        ({"context_chars_soft_limit": 1}, "context_chars_soft_limit"),
        ({"recent_group_count": 0}, "recent_group_count"),
    ],
)
def test_context_policy_rejects_invalid_limits(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ContextPolicy(**kwargs)
