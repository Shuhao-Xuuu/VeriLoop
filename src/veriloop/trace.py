"""Append-only, redacted execution traces for one VeriLoop run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, TextIO
import uuid

from .filesystem import is_link_like
from .protocol import (
    AgentState,
    Message,
    ModelResponse,
    ToolCall,
    ToolResult,
    VerificationResult,
)


TRACE_SCHEMA_VERSION = 1
TRACE_TEXT_PREVIEW_CHARS = 2_048
TRACE_GENERIC_STRING_CHARS = 4_096
TRACE_MAX_COLLECTION_ITEMS = 100
TRACE_TRUNCATION_MARKER = "...[veriloop trace truncated]..."
REDACTION_MARKER = "[REDACTED]"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUTHORIZATION_BEARER = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?\s*bearer\s+)"
    r"[^\s,;\"']+"
)
_CONTENT_ARGUMENT_NAMES = frozenset({"content", "old_text", "new_text"})
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
        self._redactor = Redactor(known_secrets)
        self._timestamp_factory = timestamp_factory or (
            lambda: datetime.now(timezone.utc)
        )
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
        content_preview, content_truncated = _preview_text(result.content)
        content: Any = content_preview
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
        text_preview, truncated = _preview_text(response.text)
        payload["text_preview"] = text_preview
        payload["text_truncated"] = truncated
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
                "stdout_preview": _preview_text(command.stdout)[0],
                "stderr_preview": _preview_text(command.stderr)[0],
                "stdout_truncated": (
                    command.stdout_truncated or _preview_text(command.stdout)[1]
                ),
                "stderr_truncated": (
                    command.stderr_truncated or _preview_text(command.stderr)[1]
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
        return _preview_text(value)
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
