from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from veriloop.model import OpenAICompatibleModel
from veriloop.process import CommandRunner
from veriloop.protocol import AgentState
from veriloop.tools import ToolRegistry
from veriloop.trace import (
    ReplayError,
    TraceWriter,
    format_trace_replay,
    load_trace_events,
    replay_trace,
)


FIXED_TIME = datetime(2026, 8, 31, 12, 34, 56, tzinfo=timezone.utc)


def event(
    seq: int = 1,
    *,
    run_id: str = "saved-run",
    event_type: str = "run_started",
    state: str = "initializing",
    payload: object | None = None,
    schema_version: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "seq": seq,
        "timestamp": "2026-08-31T12:34:56Z",
        "run_id": run_id,
        "event_type": event_type,
        "state": state,
        "payload": {} if payload is None else payload,
    }


def write_events(path: Path, *events: object) -> None:
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=True, allow_nan=True) + "\n"
            for item in events
        ),
        encoding="utf-8",
    )


def build_saved_trace(workspace: Path) -> TraceWriter:
    writer = TraceWriter(
        workspace,
        run_id="saved-run",
        timestamp_factory=lambda: FIXED_TIME,
    )
    writer.emit("run_started", AgentState.INITIALIZING, {"task_length_chars": 4})
    writer.emit(
        "state_changed",
        AgentState.THINKING,
        {"from_state": "initializing", "to_state": "thinking"},
    )
    writer.emit(
        "tool_call_received",
        AgentState.EXECUTING,
        {"tool_name": "run_command", "call_id": "command-one"},
    )
    writer.emit(
        "tool_execution_finished",
        AgentState.EXECUTING,
        {
            "tool_name": "run_command",
            "call_id": "command-one",
            "ok": False,
            "content": {"exit_code": 1},
        },
    )
    writer.emit(
        "verification_finished",
        AgentState.VERIFYING,
        {
            "passed": False,
            "skipped": False,
            "failure_kind": "verification_failed",
            "commands": [
                {
                    "exit_code": 1,
                    "timed_out": False,
                    "started": True,
                }
            ],
        },
    )
    writer.emit(
        "run_finished",
        AgentState.VERIFICATION_FAILED,
        {"state": "verification_failed"},
    )
    writer.close()
    return writer


def workspace_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_replay_accepts_a_run_directory_or_events_file(tmp_path: Path) -> None:
    writer = build_saved_trace(tmp_path)

    from_directory = replay_trace(writer.run_dir)
    from_file = replay_trace(writer.events_path)

    assert from_directory == from_file
    assert "run_id: saved-run" in from_file
    assert "0002 state_changed state=thinking from=initializing to=thinking" in from_file
    assert "0003 tool_call_received state=executing tool=run_command" in from_file
    assert "0004 tool_execution_finished state=executing" in from_file
    assert "exit_code=1" in from_file
    assert "0005 verification_finished state=verifying passed=false" in from_file
    assert "command[1] exit_code=1 timed_out=false started=true" in from_file
    assert (
        "0006 run_finished state=verification_failed "
        "final_state=verification_failed"
    ) in from_file


def test_replay_never_calls_models_tools_commands_or_writes_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = build_saved_trace(tmp_path)
    before = workspace_snapshot(tmp_path)
    forbidden_calls: list[str] = []

    def forbidden(name: str) -> Callable[..., Any]:
        def fail(*args: object, **kwargs: object) -> Any:
            forbidden_calls.append(name)
            raise AssertionError(f"replay called {name}")

        return fail

    monkeypatch.setattr(OpenAICompatibleModel, "complete", forbidden("model"))
    monkeypatch.setattr(ToolRegistry, "execute", forbidden("tool"))
    monkeypatch.setattr(CommandRunner, "run", forbidden("command"))

    rendered = replay_trace(writer.run_dir)

    assert "run_finished" in rendered
    assert forbidden_calls == []
    assert workspace_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ((["not-an-object"],), "must be an object"),
        (({"seq": 1},), "is missing"),
        ((event(seq=2),), "expected seq 1"),
        ((event(), event(seq=3)), "expected seq 2"),
        ((event(), event(seq=2, run_id="other")), "changes run_id"),
        ((event(payload=[]),), "invalid payload"),
        ((event(schema_version=2),), "unsupported schema_version"),
        (({**event(), "event_type": []},), "invalid event_type"),
        (({**event(), "state": {}},), "invalid state"),
    ],
)
def test_replay_rejects_malformed_event_protocol(
    tmp_path: Path,
    items: tuple[object, ...],
    message: str,
) -> None:
    source = tmp_path / "events.jsonl"
    write_events(source, *items)

    with pytest.raises(ReplayError, match=message):
        load_trace_events(source)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "contains no events"),
        (b"\n", "line 1 is empty"),
        (b"not-json\n", "line 1 is not valid JSON"),
        (
            (
                '{"seq":1,"timestamp":"now","run_id":"run",'
                '"event_type":"event","state":"thinking",'
                '"payload":{"value":NaN}}\n'
            ).encode("utf-8"),
            "line 1 is not valid JSON",
        ),
        (b"\xff\n", "cannot be read as UTF-8"),
    ],
)
def test_replay_rejects_corrupt_jsonl(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    source = tmp_path / "events.jsonl"
    source.write_bytes(content)

    with pytest.raises(ReplayError, match=message):
        load_trace_events(source)


def test_replay_enforces_line_and_event_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "events.jsonl"
    write_events(source, event(payload={"value": "x" * 100}))
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_LINE_CHARS", 64)
    with pytest.raises(ReplayError, match="line 1 exceeds replay limit"):
        load_trace_events(source)

    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_LINE_CHARS", 1_000_000)
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_TOTAL_CHARS", 64)
    with pytest.raises(ReplayError, match="exceeds replay total size limit"):
        load_trace_events(source)

    write_events(source, event(), event(seq=2), event(seq=3))
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_TOTAL_CHARS", 16 * 1024 * 1024)
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_EVENTS", 2)
    with pytest.raises(ReplayError, match="exceeds replay event limit"):
        load_trace_events(source)


def test_formatter_enforces_event_command_and_output_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_EVENTS", 1)
    with pytest.raises(ReplayError, match="exceeds replay event limit"):
        format_trace_replay([event(), event(seq=2)])

    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_EVENTS", 10_000)
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_COMMANDS_PER_EVENT", 1)
    verification = event(
        event_type="verification_finished",
        state="verifying",
        payload={
            "passed": False,
            "skipped": False,
            "failure_kind": "verification_failed",
            "commands": [
                {"exit_code": 1, "timed_out": False, "started": True},
                {"exit_code": 2, "timed_out": False, "started": True},
            ],
        },
    )
    with pytest.raises(ReplayError, match="exceeds replay command limit"):
        format_trace_replay([verification])

    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_COMMANDS_PER_EVENT", 100)
    monkeypatch.setattr("veriloop.trace.REPLAY_MAX_OUTPUT_CHARS", 16)
    with pytest.raises(ReplayError, match="exceeds output limit"):
        format_trace_replay([event()])


def test_formatter_rejects_oversized_python_integers() -> None:
    verification = event(
        event_type="verification_finished",
        state="verifying",
        payload={
            "passed": False,
            "skipped": False,
            "failure_kind": "verification_failed",
            "commands": [
                {
                    "exit_code": 10**5_000,
                    "timed_out": False,
                    "started": True,
                }
            ],
        },
    )

    with pytest.raises(ReplayError, match="oversized integer"):
        format_trace_replay([verification])


def test_replay_missing_source_fails_without_creating_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing-run"

    with pytest.raises(ReplayError, match="run directory or JSONL file"):
        replay_trace(missing)

    assert not missing.exists()


def test_formatter_uses_an_allowlist_and_validates_direct_input() -> None:
    safe_event = event(
        event_type="model_response_received",
        state="thinking",
        payload={
            "reasoning": "hidden material must not render",
            "headers": "raw headers must not render",
            "text_preview": "unneeded model text must not render",
            "commands": [
                {"exit_code": 99, "timed_out": False, "started": True}
            ],
        },
    )

    rendered = format_trace_replay([safe_event])

    assert "model_response_received" in rendered
    assert "hidden material" not in rendered
    assert "raw headers" not in rendered
    assert "unneeded model text" not in rendered
    assert "command[" not in rendered
    with pytest.raises(ReplayError, match="expected seq 1"):
        format_trace_replay([event(seq=9)])


def test_replay_formats_cancelled_and_infrastructure_result_shapes() -> None:
    events = [
        event(
            event_type="baseline_finished",
            state="baseline_verifying",
            payload={"cancelled": True},
        ),
        event(
            seq=2,
            event_type="verification_finished",
            state="verifying",
            payload={"error_kind": "internal_error", "error_type": "RuntimeError"},
        ),
        event(
            seq=3,
            event_type="tool_execution_finished",
            state="executing",
            payload={
                "tool_name": "complete_task",
                "call_id": "completion-one",
                "executed": True,
                "cancelled": True,
            },
        ),
    ]

    rendered = format_trace_replay(events)

    assert "baseline_finished state=baseline_verifying cancelled=true" in rendered
    assert "verification_finished state=verifying error_kind=internal_error" in rendered
    assert "tool_execution_finished state=executing tool=complete_task cancelled=true" in rendered


def test_replay_escapes_terminal_controls_and_redacts_bearer_text() -> None:
    unsafe = event(
        event_type="tool_call_received",
        state="executing",
        payload={
            "tool_name": "\x1b[31m\x07 Authorization: Bearer replay-secret",
            "call_id": "hidden-call-id",
        },
    )

    rendered = format_trace_replay([unsafe])
    rendered.encode("utf-8")

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "replay-secret" not in rendered
    assert "hidden-call-id" not in rendered
    assert "\\x1b" in rendered
    assert "\\x07" in rendered
    assert "[REDACTED]" in rendered


@pytest.mark.parametrize("schema_version", [None, True, 1.0, 2])
def test_replay_rejects_explicit_invalid_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    item = event()
    item["schema_version"] = schema_version
    source = tmp_path / "events.jsonl"
    write_events(source, item)

    with pytest.raises(ReplayError, match="unsupported schema_version"):
        load_trace_events(source)


def test_replay_accepts_the_authoritative_envelope_without_schema_version(
    tmp_path: Path,
) -> None:
    item = event()
    del item["schema_version"]
    source = tmp_path / "events.jsonl"
    write_events(source, item)

    assert load_trace_events(source) == (item,)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            event(
                event_type="state_changed",
                state="thinking",
                payload={"from_state": "initializing"},
            ),
            "invalid to_state",
        ),
        (
            event(
                event_type="state_changed",
                state="executing",
                payload={"from_state": "initializing", "to_state": "thinking"},
            ),
            "inconsistent state transition",
        ),
        (
            event(
                event_type="verification_finished",
                state="verifying",
                payload={
                    "passed": False,
                    "skipped": False,
                    "failure_kind": "verification_failed",
                    "commands": "none",
                },
            ),
            "invalid commands",
        ),
        (
            event(
                event_type="run_finished",
                state="failed",
                payload={"state": "verified"},
            ),
            "inconsistent final state",
        ),
        (
            event(
                event_type="baseline_finished",
                state="baseline_verifying",
                payload={"cancelled": True, "commands": [1]},
            ),
            "inconsistent verification result",
        ),
        (
            event(
                event_type="verification_finished",
                state="verifying",
                payload={
                    "error_kind": "internal_error",
                    "commands": [{} for _ in range(101)],
                },
            ),
            "inconsistent verification result",
        ),
        (
            event(
                event_type="tool_execution_finished",
                state="executing",
                payload={
                    "tool_name": "run_command",
                    "call_id": "command-one",
                    "cancelled": True,
                    "ok": False,
                },
            ),
            "inconsistent tool result",
        ),
    ],
)
def test_replay_rejects_malformed_display_evidence(
    item: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ReplayError, match=message):
        format_trace_replay([item])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            (
                '{"schema_version":1,"seq":1,"timestamp":"now",'
                '"run_id":"run","event_type":"run_started",'
                '"state":"initializing","payload":{"value":1e10000}}\n'
            ).encode("utf-8"),
            "non-finite number",
        ),
        (
            (
                '{"schema_version":1,"seq":1,"timestamp":"now",'
                '"run_id":"run","event_type":"run_started",'
                '"state":"initializing","payload":{"value":"\\ud800"}}\n'
            ).encode("utf-8"),
            "invalid Unicode",
        ),
    ],
)
def test_replay_normalizes_pathological_json_failures(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    source = tmp_path / "events.jsonl"
    source.write_bytes(content)

    with pytest.raises(ReplayError, match=message):
        load_trace_events(source)


def test_replay_normalizes_json_parser_recursion_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "events.jsonl"
    write_events(source, event())

    def fail_parser(*args: object, **kwargs: object) -> object:
        raise RecursionError("pathological nesting")

    monkeypatch.setattr("veriloop.trace.json.loads", fail_parser)

    with pytest.raises(ReplayError, match="line 1 is not valid JSON"):
        load_trace_events(source)
