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
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"
    MAX_STEPS = "max_steps"
    CANCELLED = "cancelled"


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
