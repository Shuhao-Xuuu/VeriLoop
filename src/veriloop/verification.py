"""Frozen verification configuration and host-side verification primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath
import tomllib
from typing import Any

from .filesystem import WorkspaceGuard
from .process import CommandRunner
from .protocol import ErrorKind
from .tools import ToolExecutionError


DEFAULT_CONFIG_PATH = ".veriloop.toml"
DEFAULT_MAX_REPAIR_ROUNDS = 2
DEFAULT_MAX_SAME_FAILURE = 2


class BaselinePolicy(str, Enum):
    MUST_FAIL = "must_fail"
    RECORD_ONLY = "record_only"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class VerificationCommandSpec:
    argv: tuple[str, ...]
    cwd: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class VerificationSpec:
    baseline_policy: BaselinePolicy
    commands: tuple[VerificationCommandSpec, ...]
    max_repair_rounds: int
    max_same_failure: int
    protected_globs: tuple[str, ...]
    config_path: str | None


class VerificationConfigError(ValueError):
    """A deterministic, safe-to-display verification configuration error."""

    kind = ErrorKind.INVALID_VERIFICATION_CONFIG


def load_verification_spec(
    guard: WorkspaceGuard,
    runner: CommandRunner,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> VerificationSpec:
    """Load one workspace-relative TOML file into an immutable specification."""

    relative_config, resolved_config = _resolve_config_path(guard, config_path)
    if not resolved_config.exists():
        return _empty_spec(relative_config)
    if not resolved_config.is_file():
        raise VerificationConfigError(
            f"verification config is not a file: {relative_config}"
        )
    if guard.path_uses_link(relative_config):
        raise VerificationConfigError(
            f"verification config must not use a symlink: {relative_config}"
        )

    try:
        raw_bytes = resolved_config.read_bytes()
    except OSError as exc:
        raise VerificationConfigError(
            f"verification config cannot be read: {relative_config}"
        ) from exc
    if len(raw_bytes) > guard.max_file_bytes:
        raise VerificationConfigError(
            f"verification config exceeds {guard.max_file_bytes} bytes"
        )
    try:
        document = tomllib.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise VerificationConfigError("verification config must be UTF-8") from exc
    except tomllib.TOMLDecodeError as exc:
        raise VerificationConfigError(
            f"verification config contains invalid TOML: {exc}"
        ) from exc

    if set(document) - {"verification"}:
        name = sorted(set(document) - {"verification"})[0]
        raise VerificationConfigError(f"unknown top-level config field: {name}")
    raw_verification = document.get("verification", {})
    if not isinstance(raw_verification, dict):
        raise VerificationConfigError("verification must be a TOML table")
    allowed_fields = {
        "baseline_policy",
        "commands",
        "max_repair_rounds",
        "max_same_failure",
        "protected_globs",
    }
    if set(raw_verification) - allowed_fields:
        name = sorted(set(raw_verification) - allowed_fields)[0]
        raise VerificationConfigError(f"unknown verification field: {name}")

    baseline_policy = _baseline_policy(
        raw_verification.get("baseline_policy", BaselinePolicy.RECORD_ONLY.value)
    )
    max_repair_rounds = _bounded_integer(
        "max_repair_rounds",
        raw_verification.get("max_repair_rounds", DEFAULT_MAX_REPAIR_ROUNDS),
        minimum=0,
    )
    max_same_failure = _bounded_integer(
        "max_same_failure",
        raw_verification.get("max_same_failure", DEFAULT_MAX_SAME_FAILURE),
        minimum=1,
    )
    protected_globs = _protected_globs(
        raw_verification.get("protected_globs", [])
    )
    commands = _commands(
        raw_verification.get("commands", []),
        guard=guard,
        runner=runner,
    )

    return VerificationSpec(
        baseline_policy=baseline_policy,
        commands=commands,
        max_repair_rounds=max_repair_rounds,
        max_same_failure=max_same_failure,
        protected_globs=protected_globs,
        config_path=relative_config,
    )


def _empty_spec(config_path: str | None) -> VerificationSpec:
    return VerificationSpec(
        baseline_policy=BaselinePolicy.RECORD_ONLY,
        commands=(),
        max_repair_rounds=DEFAULT_MAX_REPAIR_ROUNDS,
        max_same_failure=DEFAULT_MAX_SAME_FAILURE,
        protected_globs=(),
        config_path=config_path,
    )


def _resolve_config_path(
    guard: WorkspaceGuard,
    config_path: str | Path,
) -> tuple[str, Path]:
    if not isinstance(config_path, (str, Path)):
        raise VerificationConfigError("verification config path must be a string or Path")
    value = str(config_path)
    if not value:
        raise VerificationConfigError("verification config path must not be empty")
    try:
        resolved = guard.resolve_for_read(value)
    except ToolExecutionError as exc:
        raise VerificationConfigError(
            f"invalid verification config path: {value}"
        ) from exc
    return guard.relative(resolved), resolved


def _baseline_policy(value: Any) -> BaselinePolicy:
    if not isinstance(value, str):
        raise VerificationConfigError("baseline_policy must be a string")
    try:
        return BaselinePolicy(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in BaselinePolicy)
        raise VerificationConfigError(
            f"baseline_policy must be one of: {allowed}"
        ) from exc


def _bounded_integer(name: str, value: Any, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VerificationConfigError(f"{name} must be an integer")
    if value < minimum:
        raise VerificationConfigError(f"{name} must be at least {minimum}")
    return value


def _protected_globs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise VerificationConfigError("protected_globs must be a list of strings")
    normalized: list[str] = []
    for pattern in value:
        if (
            not pattern
            or "\x00" in pattern
            or "\\" in pattern
            or pattern.startswith("/")
            or PureWindowsPath(pattern).drive
            or any(part in {"", ".", ".."} for part in pattern.split("/"))
        ):
            raise VerificationConfigError(
                f"protected glob must be a non-empty workspace-relative pattern: {pattern}"
            )
        normalized.append(pattern)
    return tuple(normalized)


def _commands(
    value: Any,
    *,
    guard: WorkspaceGuard,
    runner: CommandRunner,
) -> tuple[VerificationCommandSpec, ...]:
    if not isinstance(value, list):
        raise VerificationConfigError("verification.commands must be an array of tables")
    commands: list[VerificationCommandSpec] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise VerificationConfigError(
                f"verification command {index} must be a table"
            )
        if set(item) - {"argv", "cwd", "timeout_seconds"}:
            name = sorted(set(item) - {"argv", "cwd", "timeout_seconds"})[0]
            raise VerificationConfigError(
                f"unknown field in verification command {index}: {name}"
            )
        argv = item.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(argument, str) for argument in argv)
        ):
            raise VerificationConfigError(
                f"verification command {index} argv must be a non-empty list of strings"
            )
        cwd = item.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd:
            raise VerificationConfigError(
                f"verification command {index} cwd must be a non-empty string"
            )
        timeout = item.get("timeout_seconds", 60)
        if not isinstance(timeout, int) or isinstance(timeout, bool):
            raise VerificationConfigError(
                f"verification command {index} timeout_seconds must be an integer"
            )
        if timeout < 1 or timeout > runner.max_timeout_seconds:
            raise VerificationConfigError(
                f"verification command {index} timeout_seconds must be between 1 and "
                f"{runner.max_timeout_seconds}"
            )
        try:
            cwd_path = guard.resolve(cwd, allow_root=True)
            if not cwd_path.exists() or not cwd_path.is_dir():
                raise VerificationConfigError(
                    f"verification command {index} cwd must be an existing directory"
                )
            runner.policy.validate(
                list(argv),
                cwd=cwd_path,
                workspace_root=guard.root,
            )
        except ToolExecutionError as exc:
            raise VerificationConfigError(
                f"verification command {index} is rejected: {exc}"
            ) from exc
        commands.append(
            VerificationCommandSpec(
                argv=tuple(argv),
                cwd=guard.relative(cwd_path),
                timeout_seconds=timeout,
            )
        )
    return tuple(commands)
