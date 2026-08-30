"""Provider-independent protocol objects shared by the VeriLoop harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


class ErrorKind(str, Enum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_ERROR = "tool_error"
    MODEL_PROTOCOL_ERROR = "model_protocol_error"
    PROVIDER_RETRYABLE_ERROR = "provider_retryable_error"
    PROVIDER_RETRY_EXHAUSTED = "provider_retry_exhausted"
    PROVIDER_FATAL_ERROR = "provider_fatal_error"
    MAX_STEPS = "max_steps"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"
    PATH_OUTSIDE_WORKSPACE = "path_outside_workspace"
    PATH_NOT_FOUND = "path_not_found"
    PATH_READ_DENIED = "path_read_denied"
    PATH_WRITE_DENIED = "path_write_denied"
    PATH_IS_DIRECTORY = "path_is_directory"
    PATH_IS_SYMLINK = "path_is_symlink"
    FILE_TOO_LARGE = "file_too_large"
    FILE_NOT_TEXT = "file_not_text"
    STALE_FILE = "stale_file"
    EDIT_TEXT_NOT_FOUND = "edit_text_not_found"
    EDIT_TEXT_AMBIGUOUS = "edit_text_ambiguous"
    NO_CHANGE = "no_change"
    FILE_ALREADY_EXISTS = "file_already_exists"
    COMMAND_INVALID = "command_invalid"
    COMMAND_DENIED = "command_denied"
    COMMAND_TIMEOUT = "command_timeout"
    COMMAND_NONZERO_EXIT = "command_nonzero_exit"
    COMMAND_START_ERROR = "command_start_error"
    INVALID_VERIFICATION_CONFIG = "invalid_verification_config"
    BASELINE_UNEXPECTED_PASS = "baseline_unexpected_pass"
    BASELINE_INFRASTRUCTURE_ERROR = "baseline_infrastructure_error"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_TIMEOUT = "verification_timeout"
    VERIFICATION_START_ERROR = "verification_start_error"
    PROTECTED_FILE_CHANGED = "protected_file_changed"
    COMPLETION_MUST_BE_SINGLE_CALL = "completion_must_be_single_call"
    DEFERRED_REPLAN_REQUIRED = "deferred_replan_required"
    STALLED = "stalled"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    ok: bool
    content: str
    error_kind: ErrorKind | None = None
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_result: ToolResult | None = None


class AgentState(str, Enum):
    INITIALIZING = "initializing"
    BASELINE_VERIFYING = "baseline_verifying"
    THINKING = "thinking"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    VERIFIED = "verified"
    COMPLETED_UNVERIFIED = "completed_unverified"
    VERIFICATION_FAILED = "verification_failed"
    STALLED = "stalled"
    FAILED = "failed"
    MAX_STEPS = "max_steps"
    CANCELLED = "cancelled"


class VerificationPhase(str, Enum):
    BASELINE = "baseline"
    FINAL = "final"


class ProtectedChangeKind(str, Enum):
    CREATED = "created"
    DELETED = "deleted"
    MODIFIED = "modified"
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class ProtectedFileRecord:
    relative_path: str
    existed: bool
    size: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ProtectedFileChange:
    relative_path: str
    kind: ProtectedChangeKind


@dataclass(frozen=True, slots=True)
class VerificationCommandResult:
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    timed_out: bool
    started: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: int
    error_kind: ErrorKind | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    phase: VerificationPhase
    passed: bool
    commands: tuple[VerificationCommandResult, ...]
    protected_unchanged: bool
    protected_changes: tuple[ProtectedFileChange, ...]
    mutation_seq: int
    verified_seq: int | None
    failure_kind: ErrorKind | None = None
    failure_signature: str | None = None
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class AgentError:
    kind: ErrorKind
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class AgentResult:
    state: AgentState
    final_message: str
    step_count: int
    tool_call_count: int
    history: tuple[Message, ...]
    error: AgentError | None = None
    mutation_seq: int = 0
    verified_seq: int | None = None
    repair_rounds_used: int = 0
    baseline_verification: VerificationResult | None = None
    final_verification: VerificationResult | None = None
    changed_files: tuple[str, ...] = ()
    run_id: str | None = None
    trace_path: str | None = None
    result_path: str | None = None
    patch_path: str | None = None
    duration_ms: int = 0
    model_usage: dict[str, int] = field(default_factory=dict)
    state_history: tuple[AgentState, ...] = ()
