"""Append-only, redacted execution traces for one VeriLoop run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, TextIO
import uuid

from .filesystem import WorkspaceGuard, is_link_like
from .process import DEFAULT_OUTPUT_LIMIT_BYTES, CommandPolicy, CommandRunner
from .protocol import (
    AgentResult,
    AgentState,
    ErrorKind,
    Message,
    ModelResponse,
    ToolCall,
    ToolResult,
    VerificationResult,
)
from .tools import ToolExecutionError


TRACE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
TRACE_TEXT_PREVIEW_CHARS = 2_048
TRACE_GENERIC_STRING_CHARS = 4_096
TRACE_MAX_COLLECTION_ITEMS = 100
PATCH_DIFF_LIMIT_BYTES = DEFAULT_OUTPUT_LIMIT_BYTES
PATCH_GIT_TIMEOUT_SECONDS = 30
REPLAY_MAX_EVENTS = 10_000
REPLAY_MAX_LINE_CHARS = 1_000_000
REPLAY_MAX_TOTAL_CHARS = 16 * 1024 * 1024
REPLAY_MAX_COMMANDS_PER_EVENT = TRACE_MAX_COLLECTION_ITEMS
REPLAY_MAX_OUTPUT_CHARS = 8 * 1024 * 1024
REPLAY_MAX_INTEGER_BITS = 4_096
TRACE_TRUNCATION_MARKER = "...[veriloop trace truncated]..."
REDACTION_MARKER = "[REDACTED]"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUTHORIZATION_BEARER = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?\s*bearer\s+)"
    r"[^\s,;\"']+"
)
_CONTENT_ARGUMENT_NAMES = frozenset({"content", "old_text", "new_text"})
_REPLAY_EVENT_TYPES = frozenset(
    {
        "run_started",
        "baseline_started",
        "baseline_finished",
        "state_changed",
        "model_request_started",
        "model_response_received",
        "provider_retry",
        "tool_call_received",
        "tool_execution_started",
        "tool_execution_finished",
        "workspace_revision_changed",
        "completion_requested",
        "verification_started",
        "verification_finished",
        "recovery_started",
        "run_finished",
        "run_failed",
        "run_cancelled",
    }
)
_REPLAY_STATES = frozenset(state.value for state in AgentState)
_FORBIDDEN_TRACE_KEYS = frozenset(
    {
        "authorization",
        "chain_of_thought",
        "client",
        "environ",
        "environment",
        "headers",
        "provider_client",
        "reasoning",
    }
)


class TraceError(RuntimeError):
    """The host could not establish or append a trustworthy trace."""


class ReplayError(ValueError):
    """A saved trace cannot be replayed as trustworthy read-only evidence."""


@dataclass(frozen=True, slots=True)
class _PendingTextPreview:
    """Keep full text until TraceWriter has applied credential redaction."""

    text: str
    limit: int = TRACE_TEXT_PREVIEW_CHARS


class Redactor:
    """Apply small deterministic redactions without pretending to be DLP."""

    def __init__(self, known_secrets: Sequence[str] = ()) -> None:
        self._known_secrets = tuple(
            sorted(
                {secret for secret in known_secrets if secret},
                key=len,
                reverse=True,
            )
        )

    def text(self, value: str) -> str:
        redacted = value
        for secret in self._known_secrets:
            redacted = redacted.replace(secret, REDACTION_MARKER)
        return _AUTHORIZATION_BEARER.sub(
            lambda match: match.group(1) + REDACTION_MARKER,
            redacted,
        )

    def value(self, value: Any) -> Any:
        if isinstance(value, _PendingTextPreview):
            return _PendingTextPreview(self.text(value.text), value.limit)
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("trace mapping keys must be strings")
                if key.casefold() in _FORBIDDEN_TRACE_KEYS:
                    continue
                redacted[self.text(key)] = self.value(item)
            return redacted
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        if isinstance(value, Enum):
            return value.value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise TypeError(f"trace value has unsupported type: {type(value).__name__}")


class TraceWriter:
    """Write one flushed JSON object per event under a host-owned run directory."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        known_secrets: Sequence[str] = (),
        run_id: str | None = None,
        timestamp_factory: Callable[[], datetime] | None = None,
        artifact_runner: CommandRunner | None = None,
    ) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise TraceError("trace workspace must be an existing directory")
        actual_run_id = run_id or uuid.uuid4().hex
        if not _RUN_ID_PATTERN.fullmatch(actual_run_id):
            raise TraceError("run_id contains unsafe path characters")

        metadata_dir = _safe_directory(root, root / ".veriloop")
        runs_dir = _safe_directory(root, metadata_dir / "runs")
        run_dir = runs_dir / actual_run_id
        try:
            run_dir.mkdir()
        except OSError as exc:
            raise TraceError("trace run directory cannot be created") from exc
        if is_link_like(run_dir) or not run_dir.resolve().is_relative_to(root):
            raise TraceError("trace run directory escapes the workspace")

        events_path = run_dir / "events.jsonl"
        try:
            stream = events_path.open("x", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise TraceError("events.jsonl cannot be created") from exc

        self.workspace_root = root
        self.run_id = actual_run_id
        self.run_dir = run_dir
        self.events_path = events_path
        self.relative_events_path = events_path.relative_to(root).as_posix()
        self.result_path = run_dir / "result.json"
        self.relative_result_path = self.result_path.relative_to(root).as_posix()
        self.patch_path = run_dir / "patch.diff"
        self.relative_patch_path = self.patch_path.relative_to(root).as_posix()
        self._redactor = Redactor(known_secrets)
        self._timestamp_factory = timestamp_factory or (
            lambda: datetime.now(timezone.utc)
        )
        self._artifact_runner = artifact_runner
        self._stream: TextIO | None = stream
        self._seq = 0

    @property
    def seq(self) -> int:
        return self._seq

    def emit(
        self,
        event_type: str,
        state: AgentState | str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._stream is None:
            raise TraceError("trace writer is closed")
        if not isinstance(event_type, str) or not event_type:
            raise TraceError("event_type must be a non-empty string")
        state_value = state.value if isinstance(state, AgentState) else state
        if not isinstance(state_value, str) or not state_value:
            raise TraceError("event state must be a non-empty string")

        next_seq = self._seq + 1
        try:
            event = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "seq": next_seq,
                "timestamp": _timestamp_text(self._timestamp_factory()),
                "run_id": self.run_id,
                "event_type": event_type,
                "state": state_value,
                "payload": _bounded_value(
                    self._redactor.value(dict(payload or {})),
                ),
            }
            line = json.dumps(
                event,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._stream.write(line + "\n")
            self._stream.flush()
        except (OSError, TypeError, ValueError) as exc:
            raise TraceError("trace event cannot be persisted") from exc
        self._seq = next_seq
        return event

    def record_provider_retry(
        self,
        attempt: int,
        error: str,
        delay_seconds: float,
    ) -> None:
        self.emit(
            "provider_retry",
            AgentState.THINKING,
            {
                "attempt": attempt,
                "error": error,
                "delay_seconds": delay_seconds,
                "will_retry": True,
            },
        )

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.close()
            except OSError as exc:
                raise TraceError("trace writer cannot be closed") from exc

    def write_artifacts(self, result: AgentResult) -> AgentResult:
        """Persist the host result and an optional redacted working-tree diff."""

        self._assert_run_directory_safe()
        patch_available, patch_metadata = self._write_patch_if_available()
        completed = replace(
            result,
            run_id=self.run_id,
            trace_path=self.relative_events_path,
            result_path=self.relative_result_path,
            patch_path=(self.relative_patch_path if patch_available else None),
        )
        payload = _result_payload(
            completed,
            patch_metadata=patch_metadata,
            redactor=self._redactor,
        )
        encoded: bytes | None = None
        try:
            redacted_payload = _bounded_value(self._redactor.value(payload))
            encoded = (
                json.dumps(
                    redacted_payload,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            _atomic_write_new(self.run_dir, self.result_path, encoded)
        except KeyboardInterrupt:
            if encoded is not None and _installed_artifact_matches(
                self.result_path,
                encoded,
            ):
                return completed
            return replace(completed, result_path=None)
        except (TypeError, ValueError, UnicodeEncodeError, TraceError):
            return replace(completed, result_path=None)
        return completed

    def _write_patch_if_available(self) -> tuple[bool, dict[str, Any]]:
        runner = self._artifact_runner
        if runner is None:
            guard = _artifact_workspace_guard(self.workspace_root)
            runner = CommandRunner(guard, CommandPolicy())
        try:
            repository = runner.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=".",
                timeout_seconds=PATCH_GIT_TIMEOUT_SECONDS,
            )
        except ToolExecutionError as exc:
            if exc.kind is ErrorKind.COMMAND_NONZERO_EXIT:
                return False, _patch_metadata("not_git")
            return False, {
                "available": False,
                "status": "git_unavailable",
                "truncated": False,
                "error_kind": exc.kind.value,
            }
        if str(repository.get("stdout", "")).strip().casefold() != "true":
            return False, _patch_metadata("not_git")

        try:
            diff = runner.run(
                [
                    "git",
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--",
                    ".",
                ],
                cwd=".",
                timeout_seconds=PATCH_GIT_TIMEOUT_SECONDS,
            )
        except ToolExecutionError as exc:
            return False, {
                "available": False,
                "status": "git_diff_failed",
                "truncated": False,
                "error_kind": exc.kind.value,
            }
        total_bytes = diff.get("stdout_total_bytes")
        if (
            diff.get("stdout_truncated") is True
            or (
                isinstance(total_bytes, int)
                and not isinstance(total_bytes, bool)
                and total_bytes > PATCH_DIFF_LIMIT_BYTES
            )
        ):
            return False, {
                "available": False,
                "status": "git_diff_truncated",
                "truncated": True,
                "error_kind": None,
                "limit_bytes": PATCH_DIFF_LIMIT_BYTES,
            }
        text = diff.get("stdout")
        if not isinstance(text, str):
            return False, {
                "available": False,
                "status": "git_diff_invalid_output",
                "truncated": False,
                "error_kind": None,
            }
        if not text:
            return False, _patch_metadata("no_changes")
        raw_patch = text.encode("utf-8", errors="replace")
        if len(raw_patch) > PATCH_DIFF_LIMIT_BYTES:
            return False, {
                "available": False,
                "status": "git_diff_truncated",
                "truncated": True,
                "error_kind": None,
                "limit_bytes": PATCH_DIFF_LIMIT_BYTES,
            }
        redacted = self._redactor.text(text)
        redacted_patch = redacted.encode("utf-8", errors="replace")
        if len(redacted_patch) > PATCH_DIFF_LIMIT_BYTES:
            return False, {
                "available": False,
                "status": "git_diff_truncated",
                "truncated": True,
                "error_kind": None,
                "limit_bytes": PATCH_DIFF_LIMIT_BYTES,
            }
        try:
            _atomic_write_new(
                self.run_dir,
                self.patch_path,
                redacted_patch,
            )
        except TraceError:
            return False, _patch_metadata("patch_write_failed")
        except KeyboardInterrupt:
            if not _installed_artifact_matches(self.patch_path, redacted_patch):
                return False, _patch_metadata("patch_write_interrupted")
        return True, {
            "available": True,
            "status": "written",
            "redacted": redacted != text,
            "truncated": False,
            "error_kind": None,
        }

    def _assert_run_directory_safe(self) -> None:
        try:
            resolved = self.run_dir.resolve(strict=True)
        except OSError as exc:
            raise TraceError("trace run directory is unavailable") from exc
        if is_link_like(self.run_dir) or not resolved.is_relative_to(
            self.workspace_root
        ):
            raise TraceError("trace run directory is no longer safe")

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def tool_call_payload(call: ToolCall) -> dict[str, Any]:
    arguments, arguments_truncated = _summarize_arguments(call.arguments)
    return {
        "call_id": call.id,
        "tool_name": call.name,
        "arguments": arguments,
        "arguments_truncated": arguments_truncated,
    }


def tool_result_payload(
    result: ToolResult,
    *,
    executed: bool = True,
) -> dict[str, Any]:
    try:
        parsed_content = json.loads(result.content)
    except (json.JSONDecodeError, TypeError):
        content: Any = _PendingTextPreview(result.content)
        content_truncated = len(result.content) > TRACE_TEXT_PREVIEW_CHARS
    else:
        content, content_truncated = _summarize_output_value(parsed_content)
    metadata, metadata_truncated = _summarize_output_value(result.metadata)
    return {
        "call_id": result.call_id,
        "tool_name": result.tool_name,
        "executed": executed,
        "ok": result.ok,
        "error_kind": (
            result.error_kind.value if result.error_kind is not None else None
        ),
        "retryable": result.retryable,
        "invalidates_verification": result.invalidates_verification,
        "content": content,
        "content_truncated": content_truncated,
        "metadata": metadata,
        "metadata_truncated": metadata_truncated,
    }


def model_response_payload(response: ModelResponse) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "finish_reason": response.finish_reason.value,
        "tool_call_count": len(response.tool_calls),
        "usage": dict(response.usage),
        "text_length_chars": len(response.text),
    }
    if not response.tool_calls:
        payload["text_preview"] = _PendingTextPreview(response.text)
        payload["text_truncated"] = len(response.text) > TRACE_TEXT_PREVIEW_CHARS
    return payload


def verification_result_payload(result: VerificationResult) -> dict[str, Any]:
    return {
        "phase": result.phase.value,
        "passed": result.passed,
        "skipped": result.skipped,
        "mutation_seq": result.mutation_seq,
        "verified_seq": result.verified_seq,
        "protected_unchanged": result.protected_unchanged,
        "protected_changes": [
            {"path": change.relative_path, "kind": change.kind.value}
            for change in result.protected_changes
        ],
        "failure_kind": (
            result.failure_kind.value if result.failure_kind is not None else None
        ),
        "failure_signature": result.failure_signature,
        "commands": [
            {
                "argv": list(command.argv),
                "cwd": command.cwd,
                "exit_code": command.exit_code,
                "timed_out": command.timed_out,
                "started": command.started,
                "duration_ms": command.duration_ms,
                "error_kind": (
                    command.error_kind.value
                    if command.error_kind is not None
                    else None
                ),
                "stdout_preview": _PendingTextPreview(command.stdout),
                "stderr_preview": _PendingTextPreview(command.stderr),
                "stdout_truncated": (
                    command.stdout_truncated
                    or len(command.stdout) > TRACE_TEXT_PREVIEW_CHARS
                ),
                "stderr_truncated": (
                    command.stderr_truncated
                    or len(command.stderr) > TRACE_TEXT_PREVIEW_CHARS
                ),
            }
            for command in result.commands
        ],
    }


def history_summary(messages: Sequence[Message]) -> dict[str, Any]:
    return {
        "message_count": len(messages),
        "roles": [message.role.value for message in messages],
    }


def load_trace_events(source: str | Path) -> tuple[dict[str, Any], ...]:
    """Read and validate one JSONL trace without executing saved actions."""

    path = Path(source)
    try:
        if path.is_dir():
            path = path / "events.jsonl"
        if not path.is_file():
            raise ReplayError("replay source must be a run directory or JSONL file")
        stream = path.open("r", encoding="utf-8", newline="")
    except ReplayError:
        raise
    except (OSError, ValueError) as exc:
        raise ReplayError("replay source cannot be opened") from exc

    events: list[dict[str, Any]] = []
    expected_seq = 1
    expected_run_id: str | None = None
    total_chars = 0
    try:
        with stream:
            line_number = 0
            while True:
                line_number += 1
                line = stream.readline(REPLAY_MAX_LINE_CHARS + 1)
                if line == "":
                    break
                if len(line) > REPLAY_MAX_LINE_CHARS:
                    raise ReplayError(
                        f"events.jsonl line {line_number} exceeds replay limit"
                    )
                total_chars += len(line)
                if total_chars > REPLAY_MAX_TOTAL_CHARS:
                    raise ReplayError("events.jsonl exceeds replay total size limit")
                if len(events) >= REPLAY_MAX_EVENTS:
                    raise ReplayError("events.jsonl exceeds replay event limit")
                text = line.rstrip("\r\n")
                if not text:
                    raise ReplayError(
                        f"events.jsonl line {line_number} is empty"
                    )
                try:
                    event = json.loads(
                        text,
                        parse_constant=_reject_json_constant,
                    )
                except (
                    json.JSONDecodeError,
                    OverflowError,
                    RecursionError,
                    ValueError,
                ) as exc:
                    raise ReplayError(
                        f"events.jsonl line {line_number} is not valid JSON"
                    ) from exc
                _validate_replay_event(
                    event,
                    line_number=line_number,
                    expected_seq=expected_seq,
                    expected_run_id=expected_run_id,
                )
                if expected_run_id is None:
                    expected_run_id = event["run_id"]
                events.append(event)
                expected_seq += 1
    except ReplayError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ReplayError("events.jsonl cannot be read as UTF-8") from exc

    if not events:
        raise ReplayError("events.jsonl contains no events")
    return tuple(events)


def format_trace_replay(events: Sequence[Mapping[str, Any]]) -> str:
    """Format validated events using a small payload allowlist."""

    if not events:
        raise ReplayError("cannot format an empty trace")
    if len(events) > REPLAY_MAX_EVENTS:
        raise ReplayError("trace exceeds replay event limit")
    expected_run_id: str | None = None
    for index, event in enumerate(events, start=1):
        _validate_replay_event(
            event,
            line_number=index,
            expected_seq=index,
            expected_run_id=expected_run_id,
        )
        if expected_run_id is None:
            expected_run_id = event["run_id"]
    run_id = _replay_label(events[0].get("run_id"))
    lines: list[str] = []
    output_chars = 0

    def append_line(line: str) -> None:
        nonlocal output_chars
        output_chars += len(line) + (1 if lines else 0)
        if output_chars > REPLAY_MAX_OUTPUT_CHARS:
            raise ReplayError("formatted replay exceeds output limit")
        lines.append(line)

    append_line(f"run_id: {run_id}")
    append_line(f"events: {len(events)}")
    for event in events:
        seq = event.get("seq")
        event_type = _replay_label(event.get("event_type"))
        state = _replay_label(event.get("state"))
        payload_value = event.get("payload")
        payload = payload_value if isinstance(payload_value, Mapping) else {}
        details = _replay_event_details(event_type, payload)
        line = f"{seq:04d} {event_type} state={state}"
        if details:
            line += " " + " ".join(details)
        append_line(line)
        commands = (
            payload.get("commands")
            if event_type in {"baseline_finished", "verification_finished"}
            else None
        )
        if isinstance(commands, list):
            for index, command in enumerate(commands, start=1):
                append_line(
                    "  command["
                    f"{index}] exit_code={_replay_scalar(command.get('exit_code'))} "
                    f"timed_out={_replay_scalar(command.get('timed_out'))} "
                    f"started={_replay_scalar(command.get('started'))}"
                )
    return "\n".join(lines)


def replay_trace(source: str | Path) -> str:
    """Validate and format a trace without restoring or executing a session."""

    return format_trace_replay(load_trace_events(source))


def _validate_replay_event(
    event: Any,
    *,
    line_number: int,
    expected_seq: int,
    expected_run_id: str | None,
) -> None:
    if not isinstance(event, dict):
        raise ReplayError(f"events.jsonl line {line_number} must be an object")
    _validate_replay_json_value(event, line_number=line_number)
    required = {"seq", "timestamp", "run_id", "event_type", "state", "payload"}
    missing = sorted(required - set(event))
    if missing:
        raise ReplayError(
            f"events.jsonl line {line_number} is missing: {', '.join(missing)}"
        )
    seq = event["seq"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq != expected_seq:
        raise ReplayError(
            f"events.jsonl line {line_number} expected seq {expected_seq}"
        )
    run_id = event["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ReplayError(f"events.jsonl line {line_number} has invalid run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ReplayError(f"events.jsonl line {line_number} changes run_id")
    timestamp = event["timestamp"]
    if not isinstance(timestamp, str) or not timestamp:
        raise ReplayError(f"events.jsonl line {line_number} has invalid timestamp")
    event_type = event["event_type"]
    if not isinstance(event_type, str) or event_type not in _REPLAY_EVENT_TYPES:
        raise ReplayError(f"events.jsonl line {line_number} has invalid event_type")
    state = event["state"]
    if not isinstance(state, str) or state not in _REPLAY_STATES:
        raise ReplayError(f"events.jsonl line {line_number} has invalid state")
    if not isinstance(event["payload"], dict):
        raise ReplayError(f"events.jsonl line {line_number} has invalid payload")
    if "schema_version" in event:
        schema_version = event["schema_version"]
        if type(schema_version) is not int or schema_version != TRACE_SCHEMA_VERSION:
            raise ReplayError(
                f"events.jsonl line {line_number} has unsupported schema_version"
            )
    _validate_replay_payload(event, line_number=line_number)


def _validate_replay_json_value(value: Any, *, line_number: int) -> None:
    stack = [value]
    seen_containers: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            identity = id(item)
            if identity in seen_containers:
                raise ReplayError(
                    f"events.jsonl line {line_number} has a cyclic JSON value"
                )
            seen_containers.add(identity)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ReplayError(
                        f"events.jsonl line {line_number} has a non-string JSON key"
                    )
                _validate_replay_text(key, line_number=line_number)
                stack.append(child)
            continue
        if isinstance(item, list):
            identity = id(item)
            if identity in seen_containers:
                raise ReplayError(
                    f"events.jsonl line {line_number} has a cyclic JSON value"
                )
            seen_containers.add(identity)
            stack.extend(item)
            continue
        if isinstance(item, str):
            _validate_replay_text(item, line_number=line_number)
            continue
        if isinstance(item, float) and not math.isfinite(item):
            raise ReplayError(
                f"events.jsonl line {line_number} has a non-finite number"
            )
        if (
            isinstance(item, int)
            and not isinstance(item, bool)
            and item.bit_length() > REPLAY_MAX_INTEGER_BITS
        ):
            raise ReplayError(
                f"events.jsonl line {line_number} has an oversized integer"
            )
        if item is None or isinstance(item, (bool, int, float)):
            continue
        raise ReplayError(
            f"events.jsonl line {line_number} has an unsupported JSON value"
        )


def _validate_replay_text(value: str, *, line_number: int) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReplayError(
            f"events.jsonl line {line_number} contains invalid Unicode"
        ) from exc


def _validate_replay_payload(
    event: Mapping[str, Any],
    *,
    line_number: int,
) -> None:
    event_type = event["event_type"]
    payload = event["payload"]
    if event_type == "state_changed":
        from_state = _required_replay_text(payload, "from_state", line_number)
        to_state = _required_replay_text(payload, "to_state", line_number)
        if from_state not in _REPLAY_STATES or to_state not in _REPLAY_STATES:
            raise ReplayError(
                f"events.jsonl line {line_number} has invalid state transition"
            )
        if to_state != event["state"]:
            raise ReplayError(
                f"events.jsonl line {line_number} has inconsistent state transition"
            )
        return
    if event_type in {
        "tool_call_received",
        "tool_execution_started",
        "tool_execution_finished",
    }:
        _required_replay_text(payload, "tool_name", line_number)
        _required_replay_text(payload, "call_id", line_number)
        if event_type == "tool_execution_finished":
            if "cancelled" in payload:
                if payload["cancelled"] is not True:
                    raise ReplayError(
                        f"events.jsonl line {line_number} has invalid cancelled"
                    )
                if "ok" in payload or "content" in payload:
                    raise ReplayError(
                        f"events.jsonl line {line_number} has inconsistent tool result"
                    )
            elif not isinstance(payload.get("ok"), bool):
                raise ReplayError(
                    f"events.jsonl line {line_number} has invalid ok"
                )
            content = payload.get("content")
            if isinstance(content, Mapping) and "exit_code" in content:
                exit_code = content["exit_code"]
                if exit_code is not None and (
                    not isinstance(exit_code, int) or isinstance(exit_code, bool)
                ):
                    raise ReplayError(
                        f"events.jsonl line {line_number} has invalid exit_code"
                    )
        return
    if event_type in {"baseline_finished", "verification_finished"}:
        if "cancelled" in payload:
            if payload["cancelled"] is not True:
                raise ReplayError(
                    f"events.jsonl line {line_number} has invalid cancelled"
                )
            if any(
                field in payload
                for field in (
                    "passed",
                    "skipped",
                    "failure_kind",
                    "commands",
                    "error_kind",
                )
            ):
                raise ReplayError(
                    f"events.jsonl line {line_number} has inconsistent verification result"
                )
            return
        if "passed" not in payload:
            _required_replay_text(payload, "error_kind", line_number)
            if any(
                field in payload
                for field in ("skipped", "failure_kind", "commands")
            ):
                raise ReplayError(
                    f"events.jsonl line {line_number} has inconsistent verification result"
                )
            return
        if not isinstance(payload.get("passed"), bool):
            raise ReplayError(f"events.jsonl line {line_number} has invalid passed")
        if not isinstance(payload.get("skipped"), bool):
            raise ReplayError(f"events.jsonl line {line_number} has invalid skipped")
        if "failure_kind" not in payload:
            raise ReplayError(
                f"events.jsonl line {line_number} has invalid failure_kind"
            )
        failure_kind = payload.get("failure_kind")
        if failure_kind is not None and (
            not isinstance(failure_kind, str) or not failure_kind
        ):
            raise ReplayError(
                f"events.jsonl line {line_number} has invalid failure_kind"
            )
        commands = payload.get("commands")
        if not isinstance(commands, list):
            raise ReplayError(f"events.jsonl line {line_number} has invalid commands")
        if len(commands) > REPLAY_MAX_COMMANDS_PER_EVENT:
            raise ReplayError(
                f"events.jsonl line {line_number} exceeds replay command limit"
            )
        for command in commands:
            _validate_replay_command(command, line_number=line_number)
        return
    if event_type == "workspace_revision_changed":
        mutation_seq = payload.get("mutation_seq")
        if (
            not isinstance(mutation_seq, int)
            or isinstance(mutation_seq, bool)
            or mutation_seq < 1
        ):
            raise ReplayError(
                f"events.jsonl line {line_number} has invalid mutation_seq"
            )
        return
    if event_type == "provider_retry":
        attempt = payload.get("attempt")
        delay = payload.get("delay_seconds")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ReplayError(f"events.jsonl line {line_number} has invalid attempt")
        if (
            not isinstance(delay, (int, float))
            or isinstance(delay, bool)
            or (isinstance(delay, float) and not math.isfinite(delay))
            or delay < 0
        ):
            raise ReplayError(
                f"events.jsonl line {line_number} has invalid delay_seconds"
            )
        return
    if event_type in {"run_finished", "run_failed", "run_cancelled"}:
        final_state = _required_replay_text(payload, "state", line_number)
        if final_state not in _REPLAY_STATES or final_state != event["state"]:
            raise ReplayError(
                f"events.jsonl line {line_number} has inconsistent final state"
            )


def _required_replay_text(
    payload: Mapping[str, Any],
    field: str,
    line_number: int,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ReplayError(f"events.jsonl line {line_number} has invalid {field}")
    return value


def _validate_replay_command(command: Any, *, line_number: int) -> None:
    if not isinstance(command, dict):
        raise ReplayError(
            f"events.jsonl line {line_number} has an invalid command result"
        )
    if "exit_code" not in command:
        raise ReplayError(
            f"events.jsonl line {line_number} has an invalid command exit_code"
        )
    exit_code = command["exit_code"]
    if exit_code is not None and (
        not isinstance(exit_code, int) or isinstance(exit_code, bool)
    ):
        raise ReplayError(
            f"events.jsonl line {line_number} has an invalid command exit_code"
        )
    for field in ("timed_out", "started"):
        if not isinstance(command.get(field), bool):
            raise ReplayError(
                f"events.jsonl line {line_number} has an invalid command {field}"
            )


def _replay_event_details(
    event_type: str,
    payload: Mapping[str, Any],
) -> list[str]:
    details: list[str] = []
    if event_type == "state_changed":
        details.extend(
            [
                f"from={_replay_scalar(payload.get('from_state'))}",
                f"to={_replay_scalar(payload.get('to_state'))}",
            ]
        )
    if event_type in {
        "tool_call_received",
        "tool_execution_started",
        "tool_execution_finished",
    }:
        details.extend(
            [
                f"tool={_replay_scalar(payload.get('tool_name'))}",
            ]
        )
        if event_type == "tool_execution_finished":
            if payload.get("cancelled") is True:
                details.append("cancelled=true")
            else:
                details.append(f"ok={_replay_scalar(payload.get('ok'))}")
            content = payload.get("content")
            if isinstance(content, Mapping) and "exit_code" in content:
                details.append(
                    f"exit_code={_replay_scalar(content.get('exit_code'))}"
                )
    if event_type in {"baseline_finished", "verification_finished"}:
        if payload.get("cancelled") is True:
            details.append("cancelled=true")
        elif "passed" not in payload:
            details.append(
                f"error_kind={_replay_scalar(payload.get('error_kind'))}"
            )
        else:
            details.extend(
                [
                    f"passed={_replay_scalar(payload.get('passed'))}",
                    f"skipped={_replay_scalar(payload.get('skipped'))}",
                    f"failure_kind={_replay_scalar(payload.get('failure_kind'))}",
                ]
            )
    if event_type == "workspace_revision_changed":
        details.append(
            f"mutation_seq={_replay_scalar(payload.get('mutation_seq'))}"
        )
    if event_type == "provider_retry":
        details.extend(
            [
                f"attempt={_replay_scalar(payload.get('attempt'))}",
                f"delay_seconds={_replay_scalar(payload.get('delay_seconds'))}",
            ]
        )
    if event_type in {"run_finished", "run_failed", "run_cancelled"}:
        details.append(f"final_state={_replay_scalar(payload.get('state'))}")
    return details


def _replay_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return _preview_text(str(value), 160)[0]
    if isinstance(value, float):
        if not math.isfinite(value):
            return "<invalid>"
        return format(value, ".6g")
    return _replay_label(value)


def _replay_label(value: Any) -> str:
    if not isinstance(value, str):
        return "<invalid>"
    redacted = _AUTHORIZATION_BEARER.sub(
        lambda match: match.group(1) + REDACTION_MARKER,
        value,
    )
    normalized = " ".join(redacted.split())
    terminal_safe = "".join(
        character if character.isprintable() else _escaped_codepoint(character)
        for character in normalized
    )
    return _preview_text(terminal_safe, 160)[0]


def _escaped_codepoint(value: str) -> str:
    codepoint = ord(value)
    if codepoint <= 0xFF:
        return f"\\x{codepoint:02x}"
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _result_payload(
    result: AgentResult,
    *,
    patch_metadata: Mapping[str, Any],
    redactor: Redactor,
) -> dict[str, Any]:
    final_verification = result.final_verification
    final_message = redactor.text(result.final_message)
    error_message = (
        redactor.text(result.error.message) if result.error is not None else None
    )
    protected_unchanged = (
        final_verification.protected_unchanged
        if final_verification is not None
        else None
    )
    protected_changes = (
        [
            {"path": change.relative_path, "kind": change.kind.value}
            for change in final_verification.protected_changes
        ]
        if final_verification is not None
        else []
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": result.run_id,
        "state": result.state.value,
        "final_message": final_message,
        "final_message_length_chars": len(final_message),
        "final_message_truncated": (
            len(final_message) > TRACE_GENERIC_STRING_CHARS
        ),
        "step_count": result.step_count,
        "tool_call_count": result.tool_call_count,
        "mutation_seq": result.mutation_seq,
        "verified_seq": result.verified_seq,
        "repair_rounds_used": result.repair_rounds_used,
        "baseline": (
            verification_result_payload(result.baseline_verification)
            if result.baseline_verification is not None
            else None
        ),
        "final_verification": (
            verification_result_payload(final_verification)
            if final_verification is not None
            else None
        ),
        "protected_unchanged": protected_unchanged,
        "protected_changes": protected_changes,
        "changed_files": list(result.changed_files),
        "changed_files_truncated": (
            len(result.changed_files) > TRACE_MAX_COLLECTION_ITEMS
        ),
        "trace_path": result.trace_path,
        "result_path": result.result_path,
        "patch_path": result.patch_path,
        "patch": dict(patch_metadata),
        "duration_ms": result.duration_ms,
        "model_usage": dict(result.model_usage),
        "error_kind": (
            result.error.kind.value if result.error is not None else None
        ),
        "error_message": error_message,
        "error_message_truncated": (
            error_message is not None
            and len(error_message) > TRACE_GENERIC_STRING_CHARS
        ),
    }


def _patch_metadata(status: str) -> dict[str, Any]:
    return {
        "available": False,
        "status": status,
        "truncated": False,
        "error_kind": None,
    }


def _safe_directory(root: Path, path: Path) -> Path:
    try:
        if path.exists():
            if is_link_like(path) or not path.is_dir():
                raise TraceError("trace metadata path is not a safe directory")
        else:
            path.mkdir()
        resolved = path.resolve(strict=True)
    except TraceError:
        raise
    except OSError as exc:
        raise TraceError("trace metadata directory cannot be prepared") from exc
    if not resolved.is_relative_to(root):
        raise TraceError("trace metadata directory escapes the workspace")
    return resolved


def _artifact_workspace_guard(root: Path) -> WorkspaceGuard:
    try:
        return WorkspaceGuard(root)
    except (OSError, ValueError) as exc:
        raise TraceError("artifact workspace cannot be prepared") from exc


def _atomic_write_new(directory: Path, target: Path, content: bytes) -> None:
    if target.exists() or is_link_like(target):
        raise TraceError(f"artifact already exists: {target.name}")
    temp_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            dir=directory,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() or is_link_like(target):
            raise TraceError(f"artifact already exists: {target.name}")
        try:
            if os.name == "nt":
                os.rename(temp_path, target)
                temp_path = None
            else:
                os.link(temp_path, target)
        except FileExistsError as exc:
            raise TraceError(f"artifact already exists: {target.name}") from exc
    except TraceError:
        raise
    except OSError as exc:
        raise TraceError(f"artifact cannot be written: {target.name}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _installed_artifact_matches(path: Path, expected: bytes) -> bool:
    try:
        return (
            not is_link_like(path)
            and path.is_file()
            and path.read_bytes() == expected
        )
    except OSError:
        return False


def _timestamp_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _summarize_arguments(
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    summarized: dict[str, Any] = {}
    truncated = False
    for key in sorted(arguments):
        if key.casefold() in _FORBIDDEN_TRACE_KEYS:
            truncated = True
            continue
        value = arguments[key]
        if key in _CONTENT_ARGUMENT_NAMES and isinstance(value, str):
            summarized[key] = {
                "recorded": False,
                "length_chars": len(value),
                "sha256": _text_sha256(value),
            }
            truncated = True
        else:
            summarized[key], item_truncated = _summarize_output_value(value)
            truncated = truncated or item_truncated
    return summarized, truncated


def _summarize_output_value(
    value: Any,
    *,
    depth: int = 0,
) -> tuple[Any, bool]:
    if depth >= 8:
        return TRACE_TRUNCATION_MARKER, True
    if isinstance(value, str):
        return (
            _PendingTextPreview(value),
            len(value) > TRACE_TEXT_PREVIEW_CHARS,
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if isinstance(value, Enum):
        return value.value, False
    if isinstance(value, Mapping):
        summarized: dict[str, Any] = {}
        truncated = len(value) > TRACE_MAX_COLLECTION_ITEMS
        for key in sorted(value, key=str)[:TRACE_MAX_COLLECTION_ITEMS]:
            item = value[key]
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_TRACE_KEYS:
                continue
            if key_text in _CONTENT_ARGUMENT_NAMES and isinstance(item, str):
                summarized[key_text] = {
                    "recorded": False,
                    "length_chars": len(item),
                    "sha256": _text_sha256(item),
                }
                truncated = True
                continue
            summarized_item, item_truncated = _summarize_output_value(
                item,
                depth=depth + 1,
            )
            summarized[key_text] = summarized_item
            truncated = truncated or item_truncated
        return summarized, truncated
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        truncated = len(value) > TRACE_MAX_COLLECTION_ITEMS
        for item in value[:TRACE_MAX_COLLECTION_ITEMS]:
            summarized_item, item_truncated = _summarize_output_value(
                item,
                depth=depth + 1,
            )
            items.append(summarized_item)
            truncated = truncated or item_truncated
        return items, truncated
    return {
        "recorded": False,
        "type": type(value).__name__,
    }, True


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return TRACE_TRUNCATION_MARKER
    if isinstance(value, _PendingTextPreview):
        return _preview_text(value.text, value.limit)[0]
    if isinstance(value, str):
        return _preview_text(value, TRACE_GENERIC_STRING_CHARS)[0]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[
                :TRACE_MAX_COLLECTION_ITEMS
            ]
            if str(key).casefold() not in _FORBIDDEN_TRACE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_value(item, depth=depth + 1)
            for item in value[:TRACE_MAX_COLLECTION_ITEMS]
        ]
    raise TypeError(f"trace value has unsupported type: {type(value).__name__}")


def _preview_text(
    text: str,
    limit: int = TRACE_TEXT_PREVIEW_CHARS,
) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    available = limit - len(TRACE_TRUNCATION_MARKER)
    head = max(0, (available + 1) // 2)
    tail = max(0, available - head)
    suffix = text[-tail:] if tail else ""
    return text[:head] + TRACE_TRUNCATION_MARKER + suffix, True


def _text_sha256(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
