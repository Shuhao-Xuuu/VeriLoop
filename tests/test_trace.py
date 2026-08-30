from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.scripted_model import ScriptedModel
from veriloop.agent import AgentLoop
from veriloop.filesystem import WorkspaceGuard
from veriloop.model import (
    OpenAICompatibleModel,
    ProviderFatalError,
    ProviderRetryableError,
)
from veriloop.process import CommandPolicy, CommandRunner
from veriloop.protocol import (
    AgentState,
    ErrorKind,
    FinishReason,
    Message,
    ModelResponse,
    ProtectedChangeKind,
    ProtectedFileChange,
    Role,
    ToolCall,
    ToolResult,
    VerificationCommandResult,
    VerificationPhase,
    VerificationResult,
)
from veriloop.tools import (
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
    register_workspace_tools,
)
from veriloop.trace import (
    REDACTION_MARKER,
    TRACE_TEXT_PREVIEW_CHARS,
    TRACE_TRUNCATION_MARKER,
    TraceError,
    TraceWriter,
    tool_call_payload,
    tool_result_payload,
    verification_result_payload,
)
from veriloop.verification import (
    VerificationGate,
    load_verification_spec,
    protected_guard_for_spec,
)


FIXED_TIME = datetime(2026, 8, 31, 12, 34, 56, tzinfo=timezone.utc)


def fixed_time() -> datetime:
    return FIXED_TIME


def response(text: str = "", *calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tuple(calls),
        finish_reason=(
            FinishReason.TOOL_CALLS if calls else FinishReason.STOP
        ),
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def event_types(path: Path) -> list[str]:
    return [event["event_type"] for event in read_events(path)]


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.calls.append(request)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = FakeCompletions(outcomes)
        self.chat = type("FakeChat", (), {"completions": self.completions})()


class FailingTraceStream:
    def __init__(self) -> None:
        self.closed = False

    def write(self, value: str) -> int:
        raise OSError("simulated trace write failure")

    def flush(self) -> None:
        raise OSError("simulated trace flush failure")

    def close(self) -> None:
        self.closed = True


def provider_response(text: str = "done") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"content": text, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def production_verification_stack(workspace: Path) -> tuple[ToolRegistry, VerificationGate]:
    base_guard = WorkspaceGuard(workspace)
    base_runner = CommandRunner(base_guard, CommandPolicy())
    spec = load_verification_spec(base_guard, base_runner)
    guard = protected_guard_for_spec(base_guard, spec)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    return registry, VerificationGate(spec, runner)


def test_trace_writer_creates_append_only_flushed_jsonl(tmp_path: Path) -> None:
    writer = TraceWriter(
        tmp_path,
        run_id="fixed-run",
        timestamp_factory=fixed_time,
    )
    assert writer.events_path == (
        tmp_path / ".veriloop" / "runs" / "fixed-run" / "events.jsonl"
    )

    first = writer.emit("run_started", AgentState.INITIALIZING, {"value": 1})
    first_bytes = writer.events_path.read_bytes()
    assert first["seq"] == 1
    assert len(read_events(writer.events_path)) == 1

    writer.emit("state_changed", AgentState.THINKING, {"value": 2})
    writer.emit("run_finished", AgentState.COMPLETED_UNVERIFIED)
    complete_bytes = writer.events_path.read_bytes()
    writer.close()

    assert complete_bytes.startswith(first_bytes)
    events = read_events(writer.events_path)
    assert [event["seq"] for event in events] == [1, 2, 3]
    assert {event["run_id"] for event in events} == {"fixed-run"}
    assert {event["timestamp"] for event in events} == {"2026-08-31T12:34:56Z"}
    assert all(
        {
            "schema_version",
            "seq",
            "timestamp",
            "run_id",
            "event_type",
            "state",
            "payload",
        }
        <= set(event)
        for event in events
    )


def test_trace_writer_never_overwrites_an_existing_run(tmp_path: Path) -> None:
    first = TraceWriter(tmp_path, run_id="same-run")
    first.emit("run_started", AgentState.INITIALIZING)
    original = first.events_path.read_bytes()

    with pytest.raises(TraceError, match="run directory"):
        TraceWriter(tmp_path, run_id="same-run")

    assert first.events_path.read_bytes() == original
    first.close()


def test_invalid_payload_or_closed_writer_fails_without_advancing_seq(
    tmp_path: Path,
) -> None:
    writer = TraceWriter(tmp_path, run_id="invalid-payload")
    writer.emit("run_started", AgentState.INITIALIZING)
    before = writer.events_path.read_bytes()

    with pytest.raises(TraceError, match="cannot be persisted"):
        writer.emit("bad", AgentState.THINKING, {"object": object()})

    assert writer.seq == 1
    assert writer.events_path.read_bytes() == before
    with pytest.raises(TraceError, match="cannot be persisted"):
        writer.emit("bad-number", AgentState.THINKING, {"value": float("nan")})
    assert writer.seq == 1
    assert writer.events_path.read_bytes() == before
    writer.close()
    with pytest.raises(TraceError, match="closed"):
        writer.emit("late", AgentState.FAILED)


def test_nested_known_secret_bearer_and_forbidden_fields_are_not_written(
    tmp_path: Path,
) -> None:
    secret = "veriloop-test-secret-value"
    writer = TraceWriter(
        tmp_path,
        run_id="redaction",
        known_secrets=(secret, ""),
    )
    writer.emit(
        "redaction_test",
        AgentState.THINKING,
        {
            f"key-{secret}": {
                "message": f"before {secret} after",
                "nested": [f"Authorization: Bearer token-{secret}"],
            },
            "environment": {"SECRET": secret},
            "headers": {"Authorization": f"Bearer {secret}"},
            "provider_client": object(),
            "reasoning": secret,
        },
    )
    writer.close()

    raw = writer.events_path.read_text(encoding="utf-8")
    assert secret not in raw
    assert "token-veriloop" not in raw
    assert REDACTION_MARKER in raw
    payload = read_events(writer.events_path)[0]["payload"]
    assert "environment" not in payload
    assert "headers" not in payload
    assert "provider_client" not in payload
    assert "reasoning" not in payload


def test_quoted_bearer_text_is_redacted_without_a_known_secret(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path, run_id="quoted-bearer")
    writer.emit(
        "redaction_test",
        AgentState.THINKING,
        {"message": '{"Authorization": "Bearer opaque-token-value"}'},
    )
    writer.close()

    raw = writer.events_path.read_text(encoding="utf-8")
    assert "opaque-token-value" not in raw
    assert REDACTION_MARKER in raw


def test_write_content_is_replaced_by_length_and_digest(tmp_path: Path) -> None:
    secret = "veriloop-test-secret-value"
    content = ("large-file-content-" + secret) * 500
    call = ToolCall(
        id="write-one",
        name="write_file",
        arguments={
            "path": "src/value.py",
            "content": content,
            "mode": "overwrite",
            "expected_sha256": "0" * 64,
        },
    )
    writer = TraceWriter(
        tmp_path,
        run_id="write-summary",
        known_secrets=(secret,),
    )
    writer.emit("tool_call_received", AgentState.EXECUTING, tool_call_payload(call))
    writer.close()

    raw = writer.events_path.read_text(encoding="utf-8")
    assert content not in raw
    assert secret not in raw
    arguments = read_events(writer.events_path)[0]["payload"]["arguments"]
    assert arguments["path"] == "src/value.py"
    assert arguments["content"] == {
        "length_chars": len(content),
        "recorded": False,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def test_tool_output_and_verification_streams_are_bounded(tmp_path: Path) -> None:
    secret = "veriloop-test-secret-value"
    stdout = "head-" + secret + ("x" * 8_000) + "-tail"
    result = ToolResult(
        call_id="command-one",
        tool_name="run_command",
        ok=False,
        content=json.dumps(
            {
                "argv": ["python", "verify.py"],
                "exit_code": 1,
                "stdout": stdout,
                "stderr": "failure",
                "stdout_truncated": False,
            }
        ),
        error_kind=ErrorKind.COMMAND_NONZERO_EXIT,
        metadata={"stdout": stdout, "stderr": "failure", "exit_code": 1},
        invalidates_verification=True,
    )
    command = VerificationCommandResult(
        argv=("python", "verify.py"),
        cwd=".",
        exit_code=1,
        timed_out=False,
        started=True,
        stdout=stdout,
        stderr="failure",
        stdout_truncated=False,
        stderr_truncated=False,
        duration_ms=12,
        error_kind=ErrorKind.VERIFICATION_FAILED,
    )
    verification = VerificationResult(
        phase=VerificationPhase.FINAL,
        passed=False,
        commands=(command,),
        protected_unchanged=False,
        protected_changes=(
            ProtectedFileChange("tests/test_value.py", ProtectedChangeKind.MODIFIED),
        ),
        mutation_seq=2,
        verified_seq=None,
        failure_kind=ErrorKind.PROTECTED_FILE_CHANGED,
        failure_signature="stable-signature",
    )
    writer = TraceWriter(
        tmp_path,
        run_id="bounded-output",
        known_secrets=(secret,),
    )
    writer.emit(
        "tool_execution_finished",
        AgentState.EXECUTING,
        tool_result_payload(result),
    )
    writer.emit(
        "verification_finished",
        AgentState.VERIFYING,
        verification_result_payload(verification),
    )
    writer.close()

    raw = writer.events_path.read_text(encoding="utf-8")
    assert secret not in raw
    assert len(raw) < len(stdout) * 2
    events = read_events(writer.events_path)
    tool_payload = events[0]["payload"]
    assert tool_payload["content_truncated"] is True
    assert TRACE_TRUNCATION_MARKER in tool_payload["content"]["stdout"]
    assert len(tool_payload["content"]["stdout"]) <= TRACE_TEXT_PREVIEW_CHARS
    verification_command = events[1]["payload"]["commands"][0]
    assert verification_command["stdout_truncated"] is True
    assert len(verification_command["stdout_preview"]) <= TRACE_TEXT_PREVIEW_CHARS
    assert events[1]["payload"]["protected_changes"] == [
        {"kind": "modified", "path": "tests/test_value.py"}
    ]


def test_plain_final_agent_lifecycle_is_redacted_and_finished(tmp_path: Path) -> None:
    secret = "veriloop-test-secret-value"
    writer = TraceWriter(
        tmp_path,
        run_id="plain-final",
        known_secrets=(secret,),
        timestamp_factory=fixed_time,
    )
    model = ScriptedModel([response(f"finished {secret}")])

    result = AgentLoop(model, ToolRegistry(), trace_writer=writer).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.run_id == "plain-final"
    assert result.trace_path == ".veriloop/runs/plain-final/events.jsonl"
    types = event_types(writer.events_path)
    assert types[0] == "run_started"
    assert "model_request_started" in types
    assert "model_response_received" in types
    assert types[-1] == "run_finished"
    assert "run_failed" not in types
    assert secret not in writer.events_path.read_text(encoding="utf-8")
    assert read_events(writer.events_path)[-1]["state"] == (
        AgentState.COMPLETED_UNVERIFIED.value
    )


def test_tool_events_and_workspace_revision_are_recorded_once(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="mutate",
            description="Test mutation fact",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda arguments: {"path": arguments["path"], "changed": True},
            mutates_workspace=True,
        )
    )
    call = ToolCall(id="mutate-one", name="mutate", arguments={"path": "a.py"})
    model = ScriptedModel([response("", call), response("done")])
    writer = TraceWriter(tmp_path, run_id="mutation")

    result = AgentLoop(model, registry, trace_writer=writer).run("task")

    assert result.mutation_seq == 1
    events = read_events(writer.events_path)
    types = [event["event_type"] for event in events]
    assert types.count("tool_call_received") == 1
    assert types.count("tool_execution_started") == 1
    assert types.count("tool_execution_finished") == 1
    assert types.count("workspace_revision_changed") == 1
    revision = next(
        event["payload"]
        for event in events
        if event["event_type"] == "workspace_revision_changed"
    )
    assert revision["previous_mutation_seq"] == 0
    assert revision["mutation_seq"] == 1
    assert revision["call_id"] == "mutate-one"


def test_non_json_tool_metadata_cannot_hide_a_completed_mutation(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    def mutate_then_fail(arguments: dict[str, Any]) -> None:
        (tmp_path / "changed.txt").write_text("changed\n", encoding="utf-8")
        raise ToolExecutionError(
            ErrorKind.TOOL_ERROR,
            "structured failure",
            metadata={"opaque": object()},
            invalidates_verification=True,
        )

    registry.register(
        ToolSpec(
            name="mutate_then_fail",
            description="Mutate and return a structured failure",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=mutate_then_fail,
            mutates_workspace=True,
        )
    )
    call = ToolCall(id="opaque-metadata", name="mutate_then_fail", arguments={})
    writer = TraceWriter(tmp_path, run_id="opaque-metadata")

    result = AgentLoop(
        ScriptedModel([response("", call), response("done")]),
        registry,
        trace_writer=writer,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.mutation_seq == 1
    finished = next(
        event
        for event in read_events(writer.events_path)
        if event["event_type"] == "tool_execution_finished"
    )
    assert finished["payload"]["metadata"]["opaque"] == {
        "recorded": False,
        "type": "object",
    }


def test_trace_write_failure_does_not_change_agent_or_freshness(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="mutate",
            description="Test mutation fact",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda arguments: {"changed": True},
            mutates_workspace=True,
        )
    )
    writer = TraceWriter(tmp_path, run_id="trace-write-failure")
    assert writer._stream is not None
    writer._stream.close()
    failing_stream = FailingTraceStream()
    writer._stream = failing_stream
    call = ToolCall(id="mutation", name="mutate", arguments={})

    result = AgentLoop(
        ScriptedModel([response("", call), response("done")]),
        registry,
        trace_writer=writer,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.mutation_seq == 1
    assert result.step_count == 2
    assert failing_stream.closed is True


def test_mixed_completion_records_deferred_results_without_execution(
    tmp_path: Path,
) -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="complete_task",
            description="Completion request",
            input_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            handler=lambda arguments: arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="dangerous",
            description="Must not run",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda arguments: executed.append("ran"),
            mutates_workspace=True,
        )
    )
    complete = ToolCall(
        id="complete-mixed",
        name="complete_task",
        arguments={"summary": "done"},
    )
    dangerous = ToolCall(id="dangerous-mixed", name="dangerous", arguments={})
    model = ScriptedModel(
        [response("", complete, dangerous), response("replanned")]
    )
    writer = TraceWriter(tmp_path, run_id="mixed")

    result = AgentLoop(model, registry, trace_writer=writer).run("task")

    assert result.mutation_seq == 0
    assert executed == []
    events = read_events(writer.events_path)
    assert sum(event["event_type"] == "tool_call_received" for event in events) == 2
    assert not any(event["event_type"] == "tool_execution_started" for event in events)
    deferred = [
        event
        for event in events
        if event["event_type"] == "tool_execution_finished"
    ]
    assert len(deferred) == 2
    assert all(event["payload"]["executed"] is False for event in deferred)


def test_completion_interrupt_closes_trace_with_cancelled_terminal(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    def interrupt(arguments: dict[str, Any]) -> None:
        raise KeyboardInterrupt

    registry.register(
        ToolSpec(
            name="complete_task",
            description="Completion request",
            input_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            handler=interrupt,
        )
    )
    completion = ToolCall(
        id="interrupted-completion",
        name="complete_task",
        arguments={"summary": "done"},
    )
    writer = TraceWriter(tmp_path, run_id="completion-interrupt")

    result = AgentLoop(
        ScriptedModel([response("", completion)]),
        registry,
        trace_writer=writer,
    ).run("task")

    assert result.state is AgentState.CANCELLED
    events = read_events(writer.events_path)
    assert events[-2]["event_type"] == "run_cancelled"
    assert events[-1]["event_type"] == "run_finished"
    finished = next(
        event
        for event in events
        if event["event_type"] == "tool_execution_finished"
    )
    assert finished["payload"]["cancelled"] is True


def test_provider_retry_events_are_real_retries_and_do_not_add_model_steps(
    tmp_path: Path,
) -> None:
    secret = "veriloop-test-secret-value"
    retry_error = ProviderRetryableError(f"Authorization: Bearer {secret}")
    client = FakeClient([retry_error, provider_response()])
    sleeps: list[float] = []
    writer = TraceWriter(
        tmp_path,
        run_id="retry-success",
        known_secrets=(secret,),
    )
    model = OpenAICompatibleModel(
        model="test-model",
        client=client,
        sleep=sleeps.append,
        backoff_seconds=0.5,
        retry_observer=writer.record_provider_retry,
    )

    result = AgentLoop(model, ToolRegistry(), trace_writer=writer).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.step_count == 1
    assert result.mutation_seq == 0
    assert len(client.completions.calls) == 2
    assert sleeps == [0.5]
    events = read_events(writer.events_path)
    retries = [event for event in events if event["event_type"] == "provider_retry"]
    assert len(retries) == 1
    assert retries[0]["payload"] == {
        "attempt": 1,
        "delay_seconds": 0.5,
        "error": "ProviderRetryableError",
        "will_retry": True,
    }
    assert secret not in writer.events_path.read_text(encoding="utf-8")


def test_retry_observer_failure_does_not_replace_provider_success(
    tmp_path: Path,
) -> None:
    client = FakeClient([ProviderRetryableError("retry"), provider_response()])
    sleeps: list[float] = []

    def broken_observer(attempt: int, error: str, delay_seconds: float) -> None:
        raise OSError("observer unavailable")

    model = OpenAICompatibleModel(
        model="test-model",
        client=client,
        sleep=sleeps.append,
        retry_observer=broken_observer,
    )

    result = AgentLoop(model, ToolRegistry()).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.step_count == 1
    assert len(client.completions.calls) == 2
    assert sleeps == [0.25]


def test_trace_enabled_surrogate_content_preserves_file_tool_error(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    register_workspace_tools(registry, guard, runner)
    call = ToolCall(
        id="invalid-text",
        name="write_file",
        arguments={
            "path": "invalid.txt",
            "content": "\ud800",
            "mode": "create",
        },
    )
    writer = TraceWriter(tmp_path, run_id="surrogate-content")

    result = AgentLoop(
        ScriptedModel([response("", call), response("done")]),
        registry,
        trace_writer=writer,
    ).run("task")

    tool_result = next(
        message.tool_result
        for message in result.history
        if message.tool_result is not None
    )
    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.mutation_seq == 0
    assert tool_result.error_kind is ErrorKind.FILE_NOT_TEXT
    assert not (tmp_path / "invalid.txt").exists()
    assert read_events(writer.events_path)[-1]["event_type"] == "run_finished"


def test_retry_exhaustion_emits_two_retries_then_run_failed(tmp_path: Path) -> None:
    secret = "veriloop-test-secret-value"
    failures = [ProviderRetryableError(secret) for _ in range(3)]
    client = FakeClient(failures)
    writer = TraceWriter(
        tmp_path,
        run_id="retry-exhausted",
        known_secrets=(secret,),
    )
    model = OpenAICompatibleModel(
        model="test-model",
        client=client,
        sleep=lambda delay: None,
        retry_observer=writer.record_provider_retry,
    )

    result = AgentLoop(model, ToolRegistry(), trace_writer=writer).run("task")

    assert result.state is AgentState.FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.PROVIDER_RETRY_EXHAUSTED
    assert result.step_count == 1
    types = event_types(writer.events_path)
    assert types.count("provider_retry") == 2
    assert types[-2:] == ["run_failed", "run_finished"]
    assert secret not in writer.events_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("outcome", "expected_state", "terminal_event"),
    [
        (KeyboardInterrupt(), AgentState.CANCELLED, "run_cancelled"),
        (ProviderFatalError("fatal"), AgentState.FAILED, "run_failed"),
    ],
)
def test_cancelled_and_failed_runs_have_explicit_terminal_events(
    tmp_path: Path,
    outcome: BaseException,
    expected_state: AgentState,
    terminal_event: str,
) -> None:
    writer = TraceWriter(tmp_path, run_id=f"terminal-{expected_state.value}")
    result = AgentLoop(
        ScriptedModel([outcome]),
        ToolRegistry(),
        trace_writer=writer,
    ).run("task")

    assert result.state is expected_state
    types = event_types(writer.events_path)
    assert types[-2:] == [terminal_event, "run_finished"]
    assert types.count("run_finished") == 1


def test_production_recovery_trace_finishes_verified(tmp_path: Path) -> None:
    original = b"bad\n"
    (tmp_path / "value.txt").write_bytes(original)
    (tmp_path / "verify.py").write_text(
        "from pathlib import Path\n"
        "raise SystemExit(0 if Path('value.txt').read_text().strip() == 'good' else 1)\n",
        encoding="utf-8",
    )
    (tmp_path / ".veriloop.toml").write_text(
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 2
max_same_failure = 3
[[verification.commands]]
argv = ["python", "verify.py"]
""",
        encoding="utf-8",
    )
    registry, gate = production_verification_stack(tmp_path)
    first_completion = ToolCall(
        id="complete-first",
        name="complete_task",
        arguments={"summary": "first attempt"},
    )
    repair = ToolCall(
        id="repair",
        name="write_file",
        arguments={
            "path": "value.txt",
            "content": "good\n",
            "mode": "overwrite",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    final_completion = ToolCall(
        id="complete-final",
        name="complete_task",
        arguments={"summary": "fixed"},
    )
    model = ScriptedModel(
        [
            response("", first_completion),
            response("", repair),
            response("", final_completion),
        ]
    )
    writer = TraceWriter(tmp_path, run_id="recovery")

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        trace_writer=writer,
    ).run("fix value")

    assert result.state is AgentState.VERIFIED
    assert result.mutation_seq == 1
    assert result.verified_seq == 1
    assert result.repair_rounds_used == 1
    events = read_events(writer.events_path)
    types = [event["event_type"] for event in events]
    assert "baseline_started" in types
    assert "baseline_finished" in types
    assert types.count("completion_requested") == 2
    assert types.count("verification_started") == 2
    assert types.count("verification_finished") == 2
    assert types.count("recovery_started") == 1
    assert types.count("workspace_revision_changed") == 1
    assert types[-1] == "run_finished"
    assert events[-1]["state"] == AgentState.VERIFIED.value
    completion_results = [
        event["payload"]
        for event in events
        if event["event_type"] == "tool_execution_finished"
        and event["payload"]["tool_name"] == "complete_task"
    ]
    assert [payload["ok"] for payload in completion_results] == [False, True]
    assert [
        payload["metadata"]["verified"] for payload in completion_results
    ] == [False, True]
    assert result.final_verification is not None
    assert result.final_verification.protected_unchanged is True


def test_trace_metadata_is_excluded_from_wildcard_protection(tmp_path: Path) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (tmp_path / ".veriloop.toml").write_text(
        """
[verification]
baseline_policy = "skip"
protected_globs = ["**/*"]
[[verification.commands]]
argv = ["python", "verify.py"]
""",
        encoding="utf-8",
    )
    registry, gate = production_verification_stack(tmp_path)
    writer = TraceWriter(tmp_path, run_id="wildcard")
    completion = ToolCall(
        id="complete",
        name="complete_task",
        arguments={"summary": "done"},
    )

    result = AgentLoop(
        ScriptedModel([response("", completion)]),
        registry,
        verification_gate=gate,
        trace_writer=writer,
    ).run("task")

    assert result.state is AgentState.VERIFIED
    assert result.mutation_seq == 0
    assert result.final_verification is not None
    assert result.final_verification.protected_unchanged is True
    assert not result.final_verification.protected_changes
