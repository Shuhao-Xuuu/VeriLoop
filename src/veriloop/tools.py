"""Tool registration, schema exposure, validation, and safe execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .protocol import ErrorKind, ToolCall, ToolResult

if TYPE_CHECKING:
    from .filesystem import WorkspaceGuard
    from .process import CommandRunner


ToolHandler = Callable[[dict[str, Any]], Any]
COMPLETE_TASK_TOOL_NAME = "complete_task"


class ToolExecutionError(RuntimeError):
    """A safe, expected handler failure that becomes a structured ToolResult."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        retryable: bool = False,
        metadata: dict[str, Any] | None = None,
        invalidates_verification: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.metadata = dict(metadata or {})
        self.invalidates_verification = invalidates_verification


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    mutates_workspace: bool = False


class ToolRegistry:
    """The only component that validates and invokes registered handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": deepcopy(spec.input_schema),
                },
            }
            for spec in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return _tool_failure(
                call,
                ErrorKind.UNKNOWN_TOOL,
                f"unknown tool: {call.name}",
            )

        if not isinstance(call.arguments, dict):
            return _tool_failure(
                call,
                ErrorKind.INVALID_ARGUMENTS,
                "tool arguments must be an object",
            )

        validation_error = _validate_value(call.arguments, spec.input_schema, "arguments")
        if validation_error is not None:
            return _tool_failure(call, ErrorKind.INVALID_ARGUMENTS, validation_error)

        try:
            value = spec.handler(call.arguments)
        except ToolExecutionError as exc:
            return _tool_failure(
                call,
                exc.kind,
                str(exc),
                retryable=exc.retryable,
                metadata=exc.metadata,
                invalidates_verification=(
                    spec.mutates_workspace and exc.invalidates_verification
                ),
            )
        except Exception as exc:
            return _tool_failure(
                call,
                ErrorKind.TOOL_ERROR,
                f"tool raised {type(exc).__name__}: {exc}",
            )

        return ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=_stringify_result(value),
            invalidates_verification=spec.mutates_workspace,
        )


def contains_known_secret(value: Any, known_secrets: Iterable[str]) -> bool:
    """Detect an exact host credential before a value reaches tool execution."""

    secrets = tuple(
        secret
        for secret in known_secrets
        if isinstance(secret, str) and secret
    )
    if not secrets:
        return False
    pending = [value]
    seen_containers: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if any(secret in item for secret in secrets):
                return True
            continue
        if isinstance(item, dict):
            identity = id(item)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            pending.extend(item.keys())
            pending.extend(item.values())
            continue
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            pending.extend(item)
    return False


def register_filesystem_tools(
    registry: ToolRegistry,
    guard: WorkspaceGuard,
) -> None:
    """Register the five production file tools bound to one WorkspaceGuard."""

    from .filesystem import edit_file, list_files, read_file, search_text, write_file

    registry.register(
        ToolSpec(
            name="list_files",
            description="List bounded workspace contents without following symlink directories.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 300,
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda arguments: list_files(guard, **arguments),
        )
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a bounded line range from one UTF-8 workspace file and return its SHA-256.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "start_line": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "default": 400,
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda arguments: read_file(guard, **arguments),
        )
    )
    registry.register(
        ToolSpec(
            name="search_text",
            description="Search workspace UTF-8 files for a bounded set of literal text matches.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "max_results": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda arguments: search_text(guard, **arguments),
        )
    )
    registry.register(
        ToolSpec(
            name="edit_file",
            description="Replace exactly one text occurrence after matching the file's SHA-256.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                },
                "required": [
                    "path",
                    "old_text",
                    "new_text",
                    "expected_sha256",
                ],
                "additionalProperties": False,
            },
            handler=lambda arguments: edit_file(guard, **arguments),
            mutates_workspace=True,
        )
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Atomically create or SHA-guarded overwrite one UTF-8 workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["create", "overwrite"],
                    },
                    "expected_sha256": {"type": ["string", "null"]},
                },
                "required": ["path", "content", "mode"],
                "additionalProperties": False,
            },
            handler=lambda arguments: write_file(guard, **arguments),
            mutates_workspace=True,
        )
    )


def register_process_tool(registry: ToolRegistry, runner: CommandRunner) -> None:
    """Register the production run_command handler."""

    registry.register(
        ToolSpec(
            name="run_command",
            description="Run one allowlisted argv command in a workspace directory with bounded output.",
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {
                        "type": "integer",
                        "default": 60,
                        "minimum": 1,
                        "maximum": 120,
                    },
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
            handler=runner,
            mutates_workspace=True,
        )
    )


def register_completion_tool(registry: ToolRegistry) -> None:
    """Register the model's side-effect-free request for host acceptance."""

    registry.register(
        ToolSpec(
            name=COMPLETE_TASK_TOOL_NAME,
            description=(
                "Request host verification after finishing the task. This tool does "
                "not itself grant VERIFIED and must be called alone."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "remaining_risks": {"type": "string"},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            handler=lambda arguments: {
                "summary": arguments["summary"],
                "remaining_risks": arguments.get("remaining_risks", ""),
            },
        )
    )


def register_workspace_tools(
    registry: ToolRegistry,
    guard: WorkspaceGuard,
    runner: CommandRunner,
) -> None:
    """Register every production workspace and completion tool."""

    register_filesystem_tools(registry, guard)
    register_process_tool(registry, runner)
    register_completion_tool(registry)


def build_workspace_tools(
    workspace: str | Path,
    *,
    additional_allowed_programs: Iterable[str] = (),
    max_file_bytes: int = 1024 * 1024,
) -> ToolRegistry:
    """Build the complete production registry bound to one workspace."""

    from .filesystem import WorkspaceGuard
    from .process import CommandPolicy, CommandRunner

    guard = WorkspaceGuard(workspace, max_file_bytes=max_file_bytes)
    policy = CommandPolicy(additional_allowed_programs)
    runner = CommandRunner(guard, policy)
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    return registry


def _tool_failure(
    call: ToolCall,
    kind: ErrorKind,
    message: str,
    *,
    retryable: bool = False,
    metadata: dict[str, Any] | None = None,
    invalidates_verification: bool = False,
) -> ToolResult:
    details = dict(metadata or {})
    return ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=_stringify_result(
            {
                "error_kind": kind.value,
                "message": message,
                "retryable": retryable,
                "details": details,
            }
        ),
        error_kind=kind,
        retryable=retryable,
        metadata=details,
        invalidates_verification=invalidates_verification,
    )


def make_tool_failure(
    call: ToolCall,
    kind: ErrorKind,
    message: str,
    *,
    retryable: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    """Create a paired host protocol failure without invoking a handler."""

    return _tool_failure(
        call,
        kind,
        message,
        retryable=retryable,
        metadata=metadata,
    )


def _stringify_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> str | None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_any_type(value, expected_type):
        if isinstance(expected_type, list):
            label = " or ".join(str(item) for item in expected_type)
        else:
            label = str(expected_type)
        return f"{path} must be of type {label}"

    allowed_values = schema.get("enum")
    if allowed_values is not None and value not in allowed_values:
        return f"{path} must be one of {allowed_values}"

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            return f"{path} must contain at least {min_length} characters"

    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int) and value < minimum:
            return f"{path} must be greater than or equal to {minimum}"
        if isinstance(maximum, int) and value > maximum:
            return f"{path} must be less than or equal to {maximum}"

    if expected_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for name in required:
            if name not in value:
                return f"{path}.{name} is required"

        additional = schema.get("additionalProperties", False)
        for name, item in value.items():
            item_schema = properties.get(name)
            if item_schema is None:
                if additional is False:
                    return f"{path}.{name} is not allowed"
                if isinstance(additional, dict):
                    error = _validate_value(item, additional, f"{path}.{name}")
                    if error is not None:
                        return error
                continue

            error = _validate_value(item, item_schema, f"{path}.{name}")
            if error is not None:
                return error

    if isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return f"{path} must contain at least {min_items} items"
        if isinstance(max_items, int) and len(value) > max_items:
            return f"{path} must contain at most {max_items} items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_value(item, item_schema, f"{path}[{index}]")
                if error is not None:
                    return error

    return None


def _matches_any_type(value: Any, expected_type: str | Iterable[str]) -> bool:
    if isinstance(expected_type, str):
        return _matches_type(value, expected_type)
    return any(_matches_type(value, item) for item in expected_type)


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return False
