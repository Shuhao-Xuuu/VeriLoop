"""Frozen verification configuration and host-side verification primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import site
import stat
import sys
import tempfile
import tomllib
from typing import Any, Sequence

from .filesystem import (
    WorkspaceGuard,
    is_link_like,
    matches_workspace_glob,
    normalize_workspace_glob,
)
from .process import CommandRunner
from .protocol import (
    ErrorKind,
    ProtectedChangeKind,
    ProtectedFileChange,
    ProtectedFileRecord,
    VerificationCommandResult,
    VerificationPhase,
    VerificationResult,
)
from .tools import ToolExecutionError, contains_known_secret


DEFAULT_CONFIG_PATH = ".veriloop.toml"
DEFAULT_MAX_REPAIR_ROUNDS = 2
DEFAULT_MAX_SAME_FAILURE = 2
_FAILURE_SIGNATURE_SAMPLE_CHARS = 2048
_FAILURE_SIGNATURE_EDGE_CHARS = _FAILURE_SIGNATURE_SAMPLE_CHARS // 2

_PYTHON_STARTUP_CONTROL_MODULES = (
    "sitecustomize",
    "usercustomize",
)
_PYTEST_CONTROL_FILES = (
    "conftest.py",
    "pytest.toml",
    ".pytest.toml",
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)
_PYTEST_PLUGIN_METADATA_FILES = (
    "*.dist-info/entry_points.txt",
    "*.egg-info/entry_points.txt",
)
_MANIFEST_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".veriloop",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
)
_PYTHON_MODULE_NAME = re.compile(
    r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\Z"
)
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
    verifier_control_paths: tuple[str, ...]
    config_path: str | None


class VerificationConfigError(ValueError):
    """A deterministic, safe-to-display verification configuration error."""

    kind = ErrorKind.INVALID_VERIFICATION_CONFIG


class ProtectedManifestError(RuntimeError):
    """Protected inputs could not be recorded without ambiguity."""


class VerificationGate:
    """Run frozen verification commands without owning the agent loop."""

    def __init__(self, spec: VerificationSpec, runner: CommandRunner) -> None:
        self.spec = spec
        self._runner = runner
        self._initial_manifest = build_protected_manifest(runner.guard, spec)

    @property
    def protected_manifest(self) -> tuple[ProtectedFileRecord, ...]:
        return self._initial_manifest

    def protected_changes(self) -> tuple[ProtectedFileChange, ...]:
        current = build_protected_manifest(self._runner.guard, self.spec)
        return compare_protected_manifests(self._initial_manifest, current)

    def run_baseline(self, *, mutation_seq: int = 0) -> VerificationResult:
        if not self.spec.commands or self.spec.baseline_policy is BaselinePolicy.SKIP:
            return VerificationResult(
                phase=VerificationPhase.BASELINE,
                passed=True,
                commands=(),
                protected_unchanged=True,
                protected_changes=(),
                mutation_seq=mutation_seq,
                verified_seq=None,
                skipped=True,
            )

        commands = tuple(
            _run_verification_command(self._runner, command)
            for command in self.spec.commands
        )
        infrastructure_failed = any(
            not command.started or command.timed_out for command in commands
        )
        if infrastructure_failed:
            return VerificationResult(
                phase=VerificationPhase.BASELINE,
                passed=False,
                commands=commands,
                protected_unchanged=True,
                protected_changes=(),
                mutation_seq=mutation_seq,
                verified_seq=None,
                failure_kind=ErrorKind.BASELINE_INFRASTRUCTURE_ERROR,
            )

        if self.spec.baseline_policy is BaselinePolicy.MUST_FAIL and all(
            command.exit_code == 0 for command in commands
        ):
            return VerificationResult(
                phase=VerificationPhase.BASELINE,
                passed=False,
                commands=commands,
                protected_unchanged=True,
                protected_changes=(),
                mutation_seq=mutation_seq,
                verified_seq=None,
                failure_kind=ErrorKind.BASELINE_UNEXPECTED_PASS,
            )

        return VerificationResult(
            phase=VerificationPhase.BASELINE,
            passed=True,
            commands=commands,
            protected_unchanged=True,
            protected_changes=(),
            mutation_seq=mutation_seq,
            verified_seq=None,
        )

    def run_final(self, *, mutation_seq: int) -> VerificationResult:
        before_changes = self.protected_changes()
        if not self.spec.commands:
            failure_kind = (
                ErrorKind.PROTECTED_FILE_CHANGED
                if before_changes
                else ErrorKind.VERIFICATION_FAILED
            )
            return VerificationResult(
                phase=VerificationPhase.FINAL,
                passed=False,
                commands=(),
                protected_unchanged=not before_changes,
                protected_changes=before_changes,
                mutation_seq=mutation_seq,
                verified_seq=None,
                failure_kind=failure_kind,
                failure_signature=_failure_signature(
                    commands=(),
                    protected_changes=before_changes,
                    failure_kind=failure_kind,
                    workspace_root=self._runner.guard.root,
                ),
            )

        commands = tuple(
            _run_verification_command(self._runner, command)
            for command in self.spec.commands
        )
        after_changes = self.protected_changes()
        protected_changes = _merge_protected_changes(before_changes, after_changes)

        if protected_changes:
            failure_kind = ErrorKind.PROTECTED_FILE_CHANGED
        elif any(not command.started for command in commands):
            failure_kind = ErrorKind.VERIFICATION_START_ERROR
        elif any(command.timed_out for command in commands):
            failure_kind = ErrorKind.VERIFICATION_TIMEOUT
        elif any(command.exit_code != 0 for command in commands):
            failure_kind = ErrorKind.VERIFICATION_FAILED
        else:
            failure_kind = None

        passed = failure_kind is None
        return VerificationResult(
            phase=VerificationPhase.FINAL,
            passed=passed,
            commands=commands,
            protected_unchanged=not protected_changes,
            protected_changes=protected_changes,
            mutation_seq=mutation_seq,
            verified_seq=mutation_seq if passed else None,
            failure_kind=failure_kind,
            failure_signature=(
                None
                if failure_kind is None
                else _failure_signature(
                    commands=commands,
                    protected_changes=protected_changes,
                    failure_kind=failure_kind,
                    workspace_root=self._runner.guard.root,
                )
            ),
        )

    def grants_verified(
        self,
        result: VerificationResult,
        *,
        mutation_seq: int,
    ) -> bool:
        """Apply every host-owned invariant required for VERIFIED."""

        return (
            bool(self.spec.commands)
            and result.phase is VerificationPhase.FINAL
            and result.passed
            and bool(result.commands)
            and len(result.commands) == len(self.spec.commands)
            and all(command.started for command in result.commands)
            and all(not command.timed_out for command in result.commands)
            and all(command.exit_code == 0 for command in result.commands)
            and result.protected_unchanged
            and not result.protected_changes
            and result.verified_seq == mutation_seq
            and result.mutation_seq == mutation_seq
        )


def load_verification_spec(
    guard: WorkspaceGuard,
    runner: CommandRunner,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    known_secrets: Sequence[str] = (),
) -> VerificationSpec:
    """Load one workspace-relative TOML file into an immutable specification."""

    if isinstance(config_path, (str, Path)) and contains_known_secret(
        str(config_path), known_secrets
    ):
        raise VerificationConfigError(
            "verification config path contains a host credential"
        )
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
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationConfigError("verification config must be UTF-8") from exc
    if contains_known_secret(decoded, known_secrets):
        raise VerificationConfigError(
            "verification config contains a host credential"
        )
    try:
        document = tomllib.loads(decoded)
    except tomllib.TOMLDecodeError as exc:
        raise VerificationConfigError(
            f"verification config contains invalid TOML: {exc}"
        ) from exc
    if contains_known_secret(document, known_secrets):
        raise VerificationConfigError(
            "verification config contains a host credential"
        )

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
    configured_protected_globs = _protected_globs(
        raw_verification.get("protected_globs", [])
    )
    commands, verifier_control_globs, verifier_control_paths = _commands(
        raw_verification.get("commands", []),
        guard=guard,
        runner=runner,
        known_secrets=known_secrets,
    )
    protected_globs = tuple(
        dict.fromkeys((*configured_protected_globs, *verifier_control_globs))
    )

    return VerificationSpec(
        baseline_policy=baseline_policy,
        commands=commands,
        max_repair_rounds=max_repair_rounds,
        max_same_failure=max_same_failure,
        protected_globs=protected_globs,
        verifier_control_paths=verifier_control_paths,
        config_path=relative_config,
    )


def build_protected_manifest(
    guard: WorkspaceGuard,
    spec: VerificationSpec,
) -> tuple[ProtectedFileRecord, ...]:
    """Hash the frozen protected path set without following links."""

    records: dict[str, ProtectedFileRecord] = {}
    for path in _workspace_entries(guard.root):
        relative = path.relative_to(guard.root).as_posix()
        if any(
            matches_workspace_glob(relative, pattern)
            for pattern in spec.protected_globs
        ):
            records[relative] = _protected_record(path, relative)

    exact_paths = list(spec.verifier_control_paths)
    if spec.config_path is not None:
        exact_paths.append(spec.config_path)
    for relative in exact_paths:
        records[relative] = _protected_record(
            guard.lexical_path(relative),
            relative,
        )

    return tuple(records[path] for path in sorted(records))


def compare_protected_manifests(
    initial: tuple[ProtectedFileRecord, ...],
    current: tuple[ProtectedFileRecord, ...],
) -> tuple[ProtectedFileChange, ...]:
    """Return stable path-only integrity changes between two manifests."""

    before = {record.relative_path: record for record in initial}
    after = {record.relative_path: record for record in current}
    changes: list[ProtectedFileChange] = []
    for relative in sorted(set(before) | set(after)):
        old = before.get(relative)
        new = after.get(relative)
        if (old is None or not old.existed) and new is not None and new.existed:
            kind = ProtectedChangeKind.CREATED
        elif old is not None and old.existed and (new is None or not new.existed):
            kind = ProtectedChangeKind.DELETED
        elif old is None or new is None or not old.existed or not new.existed:
            continue
        elif old.file_kind != new.file_kind:
            kind = ProtectedChangeKind.REPLACED
        elif old.size != new.size or old.sha256 != new.sha256:
            kind = ProtectedChangeKind.MODIFIED
        else:
            continue
        changes.append(ProtectedFileChange(relative_path=relative, kind=kind))
    return tuple(changes)


def _merge_protected_changes(
    *groups: tuple[ProtectedFileChange, ...],
) -> tuple[ProtectedFileChange, ...]:
    unique = {
        (change.relative_path, change.kind.value): change
        for group in groups
        for change in group
    }
    return tuple(unique[key] for key in sorted(unique))


def _failure_signature(
    *,
    commands: tuple[VerificationCommandResult, ...],
    protected_changes: tuple[ProtectedFileChange, ...],
    failure_kind: ErrorKind,
    workspace_root: Path,
) -> str:
    payload = {
        "failure_kind": failure_kind.value,
        "commands": [
            {
                "argv": [
                    _normalized_failure_text(argument, workspace_root)
                    for argument in command.argv
                ],
                "cwd": _normalized_failure_text(command.cwd, workspace_root),
                "exit_code": command.exit_code,
                "timed_out": command.timed_out,
                "started": command.started,
                "error_kind": (
                    command.error_kind.value
                    if command.error_kind is not None
                    else None
                ),
                "stdout_sample": _normalized_failure_sample(
                    command.stdout,
                    workspace_root,
                ),
                "stderr_sample": _normalized_failure_sample(
                    command.stderr,
                    workspace_root,
                ),
            }
            for command in commands
        ],
        "protected_changes": [
            {"path": change.relative_path, "kind": change.kind.value}
            for change in sorted(
                protected_changes,
                key=lambda item: (item.relative_path, item.kind.value),
            )
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_failure_sample(
    text: str,
    workspace_root: Path,
) -> dict[str, str | bool]:
    normalized = _normalized_failure_text(text, workspace_root)
    if len(normalized) <= _FAILURE_SIGNATURE_SAMPLE_CHARS:
        return {
            "head": normalized,
            "tail": "",
            "middle_omitted": False,
        }
    return {
        "head": normalized[:_FAILURE_SIGNATURE_EDGE_CHARS],
        "tail": normalized[-_FAILURE_SIGNATURE_EDGE_CHARS:],
        "middle_omitted": True,
    }


def _normalized_failure_text(text: str, workspace_root: Path) -> str:
    normalized = text
    root_values = {
        str(workspace_root),
        workspace_root.as_posix(),
        str(workspace_root).replace("\\", "/"),
        str(workspace_root).replace("/", "\\"),
    }
    flags = re.IGNORECASE if os.name == "nt" else 0
    for root_value in sorted(root_values, key=len, reverse=True):
        if root_value:
            normalized = re.sub(
                re.escape(root_value),
                "<workspace>",
                normalized,
                flags=flags,
            )

    temp_root = Path(tempfile.gettempdir())
    temp_values = {
        str(temp_root),
        temp_root.as_posix(),
        str(temp_root).replace("\\", "/"),
        str(temp_root).replace("/", "\\"),
    }
    for temp_value in sorted(temp_values, key=len, reverse=True):
        if temp_value:
            normalized = re.sub(
                re.escape(temp_value),
                "<temp>",
                normalized,
                flags=flags,
            )

    substitutions = (
        (
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",
            "<timestamp>",
        ),
        (
            r"(?i)\b(timestamp|date|time)\s*[:=]\s*"
            r"(?:\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?"
            r"|\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
            r"\1=<timestamp>",
        ),
        (
            r"(?i)\b(pid|process_id|seq(?:uence)?)\s*[:=]\s*\d+\b",
            r"\1=<volatile>",
        ),
        (
            r"(?i)\brun_id\s*[:=]\s*[a-z0-9][a-z0-9._-]*",
            "run_id=<volatile>",
        ),
        (
            r"(?i)\b(in|duration|elapsed|latency)\s*[:=]?\s*"
            r"\d+(?:\.\d+)?\s*(?:ns|us|ms|s|sec(?:ond)?s?|minutes?)\b",
            r"\1 <duration>",
        ),
        (r"(?i)<\d+ bytes omitted>", "<bytes omitted>"),
        (
            r"(?i)([\\/])(?:pytest-\d+|popen-gw\d+|tmp[a-z0-9_.-]{6,})"
            r"(?=[\\/\s\"']|$)",
            r"\1<random-temp>",
        ),
    )
    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def protected_guard_for_spec(
    guard: WorkspaceGuard,
    spec: VerificationSpec,
) -> WorkspaceGuard:
    """Create the model-facing guard with frozen verification write denies."""

    exact_paths = list(spec.verifier_control_paths)
    if spec.config_path is not None:
        exact_paths.append(spec.config_path)
    return WorkspaceGuard(
        guard.root,
        max_file_bytes=guard.max_file_bytes,
        protected_write_globs=spec.protected_globs,
        protected_write_paths=exact_paths,
    )


def _empty_spec(config_path: str | None) -> VerificationSpec:
    return VerificationSpec(
        baseline_policy=BaselinePolicy.RECORD_ONLY,
        commands=(),
        max_repair_rounds=DEFAULT_MAX_REPAIR_ROUNDS,
        max_same_failure=DEFAULT_MAX_SAME_FAILURE,
        protected_globs=(),
        verifier_control_paths=(),
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
        try:
            normalized_pattern = normalize_workspace_glob(pattern)
        except ValueError as exc:
            raise VerificationConfigError(
                f"protected glob must be a non-empty workspace-relative pattern: {pattern}"
            ) from exc
        normalized.append(normalized_pattern)
    return tuple(normalized)


def _commands(
    value: Any,
    *,
    guard: WorkspaceGuard,
    runner: CommandRunner,
    known_secrets: Sequence[str],
) -> tuple[
    tuple[VerificationCommandSpec, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(value, list):
        raise VerificationConfigError("verification.commands must be an array of tables")
    commands: list[VerificationCommandSpec] = []
    verifier_control_globs: set[str] = set()
    verifier_control_paths: set[str] = set()
    pytest_plugin_names: set[str] = set()
    pytest_config_paths: set[str] = set()
    has_pytest_command = False
    python_path = next(
        (
            value
            for name, value in runner.child_environment.items()
            if name.casefold() == "pythonpath"
        ),
        None,
    )
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
        if contains_known_secret({"argv": argv, "cwd": cwd}, known_secrets):
            raise VerificationConfigError(
                f"verification command {index} contains a host credential"
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
            relative_cwd = guard.relative(cwd_path)
            if any(
                part.casefold() in _MANIFEST_SKIPPED_DIRECTORIES
                for part in Path(relative_cwd).parts
            ):
                raise VerificationConfigError(
                    f"verification command {index} cwd enters runtime metadata or cache"
                )
            validated_argv = runner.policy.validate(
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
                cwd=relative_cwd,
                timeout_seconds=timeout,
            )
        )
        control_globs, control_paths, plugin_names, config_paths, is_pytest = (
            _python_verifier_controls(
                validated_argv,
                cwd=cwd_path,
                guard=guard,
                python_path=python_path,
            )
        )
        verifier_control_globs.update(control_globs)
        verifier_control_paths.update(control_paths)
        pytest_plugin_names.update(plugin_names)
        pytest_config_paths.update(config_paths)
        has_pytest_command = has_pytest_command or is_pytest

    if has_pytest_command:
        pytest_plugin_names.update(_installed_pytest_plugin_names())
        pytest_plugin_names.update(
            _workspace_pytest_plugin_names(
                guard,
                explicit_config_paths=pytest_config_paths,
            )
        )
        for plugin_name in pytest_plugin_names:
            verifier_control_globs.update(
                _python_module_control_globs(plugin_name)
            )

    return (
        tuple(commands),
        tuple(sorted(verifier_control_globs)),
        tuple(sorted(verifier_control_paths)),
    )


def _python_verifier_controls(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    guard: WorkspaceGuard,
    python_path: str | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool,
]:
    if not argv or argv[0] != sys.executable:
        return (), (), (), (), False

    startup_globs, startup_paths = _workspace_python_startup_controls(
        guard,
        cwd=cwd,
        python_path=python_path,
    )
    controls = set(startup_globs)
    for module_name in _PYTHON_STARTUP_CONTROL_MODULES:
        controls.update(_python_module_control_globs(module_name))
    controls.update(_workspace_python_startup_hook_controls(guard))

    if len(argv) < 3 or argv[1] != "-m":
        return tuple(sorted(controls)), startup_paths, (), (), False

    module = argv[2]
    controls.update(_python_module_control_globs(module))

    if module.casefold() != "pytest":
        return tuple(sorted(controls)), startup_paths, (), (), False

    for name in _PYTEST_CONTROL_FILES:
        controls.update(_workspace_candidate_globs(name))
    for name in _PYTEST_PLUGIN_METADATA_FILES:
        controls.update(_workspace_candidate_globs(name))
    config_paths, plugin_names = _pytest_argv_controls(
        argv[3:],
        cwd=cwd,
        guard=guard,
    )
    control_paths = set(config_paths)
    control_paths.update(startup_paths)
    return (
        tuple(sorted(controls)),
        tuple(sorted(control_paths)),
        tuple(sorted(plugin_names)),
        tuple(sorted(config_paths)),
        True,
    )


def _workspace_python_startup_controls(
    guard: WorkspaceGuard,
    *,
    cwd: Path,
    python_path: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        site_directories = list(site.getsitepackages())
        user_site = site.getusersitepackages()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise VerificationConfigError(
            "Python startup paths cannot be enumerated"
        ) from exc
    if isinstance(user_site, str):
        site_directories.append(user_site)
    else:
        site_directories.extend(user_site)

    controls: set[str] = set()
    paths = _workspace_pythonpath_control_paths(
        python_path,
        cwd=cwd,
        guard=guard,
    )
    for directory in site_directories:
        relative = _workspace_runtime_control_path(
            directory,
            guard=guard,
            description="Python site directory",
        )
        if relative is None:
            continue
        if relative != ".":
            paths.add(relative)
            pth_pattern = f"{relative}/*.pth"
        else:
            pth_pattern = "*.pth"
        controls.add(pth_pattern)
        controls.add(_case_insensitive_glob(pth_pattern))
        pth_controls, pth_paths = _workspace_pth_controls(
            guard.lexical_path(relative),
            relative=relative,
            guard=guard,
        )
        controls.update(pth_controls)
        paths.update(pth_paths)

    executable = Path(sys.executable)
    startup_candidates = (
        executable,
        executable.with_suffix("._pth"),
        executable.parent
        / f"python{sys.version_info.major}{sys.version_info.minor}._pth",
        executable.parent / "pyvenv.cfg",
        executable.parent.parent / "pyvenv.cfg",
    )
    for candidate in startup_candidates:
        relative = _workspace_runtime_control_path(
            candidate,
            guard=guard,
            description="Python interpreter startup control",
        )
        if relative not in (None, "."):
            paths.add(relative)
            if candidate.name.casefold().endswith("._pth"):
                paths.update(
                    _workspace_isolated_python_path_controls(
                        candidate,
                        relative=relative,
                        guard=guard,
                    )
                )
    return tuple(sorted(controls)), tuple(sorted(paths))


def _workspace_pythonpath_control_paths(
    python_path: str | None,
    *,
    cwd: Path,
    guard: WorkspaceGuard,
) -> set[str]:
    paths: set[str] = set()
    if python_path is None:
        return paths
    for raw_entry in python_path.split(os.pathsep):
        candidate = cwd if not raw_entry else Path(raw_entry)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        relative = _workspace_runtime_control_path(
            candidate,
            guard=guard,
            description="PYTHONPATH entry",
        )
        if relative not in (None, "."):
            paths.add(relative)
    return paths


def _workspace_isolated_python_path_controls(
    path: Path,
    *,
    relative: str,
    guard: WorkspaceGuard,
) -> set[str]:
    text = _control_text(
        path,
        relative,
        guard.max_file_bytes,
        description="Python ._pth file",
    )
    if text is None:
        return set()
    paths: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "import site":
            continue
        if line.startswith(("import ", "import\t")):
            raise VerificationConfigError(
                f"Python ._pth executable line is not supported: {relative}"
            )
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        target = _workspace_runtime_control_path(
            candidate,
            guard=guard,
            description="Python ._pth path entry",
        )
        if target not in (None, "."):
            paths.add(target)
    return paths


def _workspace_python_startup_hook_controls(
    guard: WorkspaceGuard,
) -> set[str]:
    modules = {name.casefold() for name in _PYTHON_STARTUP_CONTROL_MODULES}
    source_names = {f"{name}.py" for name in modules}
    compiled_names = {
        f"{name}{suffix}".casefold()
        for name in modules
        for suffix in importlib.machinery.all_suffixes()
        if suffix != ".py"
    }
    package_compiled_names = {
        f"__init__{suffix}".casefold()
        for suffix in importlib.machinery.all_suffixes()
        if suffix != ".py"
    }

    controls: set[str] = set()
    for path in _workspace_entries(guard.root):
        name = path.name.casefold()
        parent_name = path.parent.name.casefold()
        is_source = name in source_names or (
            name == "__init__.py" and parent_name in modules
        )
        is_compiled = name in compiled_names or (
            name in package_compiled_names and parent_name in modules
        )
        if not is_source and not is_compiled:
            continue
        relative = path.relative_to(guard.root).as_posix()
        if is_compiled:
            raise VerificationConfigError(
                f"compiled Python startup hook cannot be inspected safely: {relative}"
            )
        text = _control_text(
            path,
            relative,
            guard.max_file_bytes,
            description="Python startup hook",
        )
        if text is None:
            continue
        for module_name in _static_python_control_modules(
            text,
            relative=relative,
            description="Python startup hook",
            allow_docstring=True,
        ):
            controls.update(_python_module_control_globs(module_name))
    return controls


def _workspace_pth_controls(
    site_directory: Path,
    *,
    relative: str,
    guard: WorkspaceGuard,
) -> tuple[set[str], set[str]]:
    if not site_directory.exists():
        return set(), set()
    if not site_directory.is_dir():
        raise VerificationConfigError(
            f"Python site path is not a directory: {relative}"
        )
    try:
        pth_files = sorted(
            path
            for path in site_directory.iterdir()
            if path.name.casefold().endswith(".pth")
        )
    except OSError as exc:
        raise VerificationConfigError(
            f"Python site directory cannot be read: {relative}"
        ) from exc

    controls: set[str] = set()
    paths: set[str] = set()
    for path in pth_files:
        pth_relative = guard.relative_lexical(path)
        text = _control_text(
            path,
            pth_relative,
            guard.max_file_bytes,
            description="Python .pth file",
        )
        if text is None:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                for module_name in _python_pth_import_modules(line, pth_relative):
                    controls.update(_python_module_control_globs(module_name))
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = site_directory / candidate
            target = _workspace_runtime_control_path(
                candidate,
                guard=guard,
                description="Python .pth path entry",
            )
            if target not in (None, "."):
                paths.add(target)
    return controls, paths


def _python_pth_import_modules(line: str, relative: str) -> set[str]:
    return _static_python_control_modules(
        line,
        relative=relative,
        description="Python executable .pth line",
        allow_docstring=False,
    )


def _static_python_control_modules(
    text: str,
    *,
    relative: str,
    description: str,
    allow_docstring: bool,
) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise VerificationConfigError(
            f"{description} cannot be parsed safely: {relative}"
        ) from exc
    modules: set[str] = set()
    bound_names: set[str] = set()
    for statement in tree.body:
        if (
            allow_docstring
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if _PYTHON_MODULE_NAME.fullmatch(alias.name) is None:
                    raise VerificationConfigError(
                        f"{description} import cannot be protected: {relative}"
                    )
                modules.add(alias.name)
                bound_names.add(alias.asname or alias.name.split(".", 1)[0])
            continue
        if isinstance(statement, ast.ImportFrom):
            if (
                statement.level
                or statement.module is None
                or _PYTHON_MODULE_NAME.fullmatch(statement.module) is None
            ):
                raise VerificationConfigError(
                    f"{description} import cannot be protected: {relative}"
                )
            modules.add(statement.module)
            bound_names.update(
                alias.asname or alias.name for alias in statement.names
            )
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            root = call.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if (
                call.args
                or call.keywords
                or not isinstance(root, ast.Name)
                or root.id not in bound_names
            ):
                raise VerificationConfigError(
                    f"{description} is not statically safe: {relative}"
                )
            continue
        raise VerificationConfigError(
            f"{description} is not statically safe: {relative}"
        )
    if not modules and not allow_docstring:
        raise VerificationConfigError(
            f"{description} has no protected import: {relative}"
        )
    return modules


def _workspace_runtime_control_path(
    value: str | os.PathLike[str],
    *,
    guard: WorkspaceGuard,
    description: str,
) -> str | None:
    try:
        candidate = Path(os.path.abspath(value))
        relative = guard.relative_lexical(candidate)
    except (OSError, TypeError, ValueError, ToolExecutionError):
        return None
    try:
        uses_link = guard.path_uses_link(relative)
    except ToolExecutionError as exc:
        raise VerificationConfigError(
            f"{description} must stay inside the workspace: {relative}"
        ) from exc
    if uses_link:
        raise VerificationConfigError(
            f"{description} must not use a symlink: {relative}"
        )
    return relative


def _python_module_control_globs(module: str) -> tuple[str, ...]:
    if _PYTHON_MODULE_NAME.fullmatch(module) is None:
        return ()
    top_level = module.split(".", 1)[0]
    controls: set[str] = set()
    for suffix in importlib.machinery.all_suffixes():
        controls.update(_workspace_candidate_globs(f"{top_level}{suffix}"))
    controls.update(_workspace_candidate_globs(top_level))
    controls.update(_workspace_candidate_globs(f"{top_level}/**"))
    return tuple(sorted(controls))


def _pytest_argv_controls(
    arguments: tuple[str, ...],
    *,
    cwd: Path,
    guard: WorkspaceGuard,
) -> tuple[set[str], set[str]]:
    control_paths: set[str] = set()
    plugin_names: set[str] = set()
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        config_value: str | None = None
        plugin_value: str | None = None
        if argument in {"-c", "--config-file"}:
            if index + 1 >= len(arguments):
                raise VerificationConfigError(
                    f"pytest option {argument} requires a config path"
                )
            index += 1
            config_value = arguments[index]
        elif argument.startswith("--config-file="):
            config_value = argument.split("=", 1)[1]
        elif argument.startswith("-c") and not argument.startswith("--"):
            config_value = argument[2:]
        elif argument == "-p":
            if index + 1 >= len(arguments):
                raise VerificationConfigError("pytest option -p requires a plugin name")
            index += 1
            plugin_value = arguments[index]
        elif argument.startswith("-p") and not argument.startswith("--"):
            plugin_value = argument[2:]
        else:
            plugin_names.update(_pytest_plugin_names_from_text(argument))

        if config_value is not None:
            relative = _workspace_pytest_config_path(
                config_value,
                cwd=cwd,
                guard=guard,
            )
            if relative is not None:
                control_paths.add(relative)
        if plugin_value is not None:
            plugin_name = plugin_value.strip()
            if not plugin_name:
                raise VerificationConfigError(
                    "pytest option -p requires a plugin name"
                )
            if not plugin_name.startswith("no:"):
                if _PYTHON_MODULE_NAME.fullmatch(plugin_name) is None:
                    raise VerificationConfigError(
                        "pytest plugin module name cannot be protected safely"
                    )
                plugin_names.add(plugin_name)
        index += 1
    return control_paths, plugin_names


def _workspace_pytest_config_path(
    value: str,
    *,
    cwd: Path,
    guard: WorkspaceGuard,
) -> str:
    if not value:
        raise VerificationConfigError("pytest config path must not be empty")
    try:
        candidate = Path(value)
        lexical = Path(
            os.path.abspath(candidate if candidate.is_absolute() else cwd / candidate)
        )
        lexical_relative = guard.relative_lexical(lexical)
        if guard.path_uses_link(lexical_relative):
            raise VerificationConfigError(
                f"pytest config path must not use a symlink: {lexical_relative}"
            )
        resolved = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (cwd / candidate).resolve(strict=False)
        )
        relative = resolved.relative_to(guard.root).as_posix()
    except VerificationConfigError:
        raise
    except (OSError, RuntimeError, ValueError, ToolExecutionError) as exc:
        raise VerificationConfigError(
            "pytest config path must stay inside the workspace"
        ) from exc
    if not relative or relative == ".":
        raise VerificationConfigError("pytest config path must name a file")
    return relative


def _workspace_pytest_plugin_names(
    guard: WorkspaceGuard,
    *,
    explicit_config_paths: set[str],
) -> set[str]:
    candidates: dict[str, Path] = {}
    config_names = {name.casefold() for name in _PYTEST_CONTROL_FILES}
    for path in _workspace_entries(guard.root):
        relative = path.relative_to(guard.root).as_posix()
        name = path.name.casefold()
        if name in config_names or (
            name == "entry_points.txt"
            and path.parent.name.casefold().endswith((".dist-info", ".egg-info"))
        ):
            candidates[relative] = path
    for relative in explicit_config_paths:
        candidates[relative] = guard.lexical_path(relative)

    plugin_names: set[str] = set()
    for relative, path in sorted(candidates.items()):
        text = _pytest_control_text(path, relative, guard.max_file_bytes)
        if text is None:
            continue
        plugin_names.update(_pytest_plugin_names_from_text(text))
        if path.name.casefold() == "conftest.py":
            plugin_names.update(
                _pytest_plugins_from_python_source(text, relative)
            )
        if "pytest11" in text.casefold():
            plugin_names.update(_pytest_entrypoint_modules(text))
    return plugin_names


def _installed_pytest_plugin_names() -> set[str]:
    try:
        entrypoints = importlib.metadata.entry_points(group="pytest11")
    except Exception as exc:
        raise VerificationConfigError(
            "installed pytest plugin entry points cannot be enumerated"
        ) from exc

    names: set[str] = set()
    for entrypoint in entrypoints:
        value = getattr(entrypoint, "value", None)
        module = _pytest_entrypoint_module(value) if isinstance(value, str) else None
        if module is None:
            raise VerificationConfigError(
                "installed pytest plugin entry point has an unsafe target"
            )
        names.add(module)
    return names


def _pytest_control_text(
    path: Path,
    relative: str,
    max_bytes: int,
) -> str | None:
    return _control_text(
        path,
        relative,
        max_bytes,
        description="pytest control file",
    )


def _control_text(
    path: Path,
    relative: str,
    max_bytes: int,
    *,
    description: str,
) -> str | None:
    if not path.exists():
        return None
    if is_link_like(path):
        raise VerificationConfigError(
            f"{description} must not use a symlink: {relative}"
        )
    if not path.is_file():
        raise VerificationConfigError(f"{description} is not a file: {relative}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VerificationConfigError(
            f"{description} cannot be read: {relative}"
        ) from exc
    if len(raw) > max_bytes:
        raise VerificationConfigError(
            f"{description} exceeds {max_bytes} bytes: {relative}"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationConfigError(
            f"{description} must be UTF-8: {relative}"
        ) from exc


def _pytest_plugin_names_from_text(text: str) -> set[str]:
    if "-p" not in text:
        return set()
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars="[]=,")
        lexer.whitespace_split = True
        lexer.commenters = "#;"
        raw_tokens = list(lexer)
    except ValueError as exc:
        raise VerificationConfigError(
            "pytest plugin options cannot be parsed safely"
        ) from exc
    tokens = [
        token
        for token in raw_tokens
        if token and not all(character in "[]=," for character in token)
    ]

    names: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name: str | None = None
        if token == "-p":
            if index + 1 >= len(tokens):
                raise VerificationConfigError(
                    "pytest option -p requires a plugin name"
                )
            index += 1
            name = tokens[index].strip()
        elif token.startswith("-p") and not token.startswith("--"):
            name = token[2:].strip()
        if name is None:
            index += 1
            continue
        if not name:
            raise VerificationConfigError("pytest option -p requires a plugin name")
        if name.startswith("no:"):
            index += 1
            continue
        if _PYTHON_MODULE_NAME.fullmatch(name) is None:
            raise VerificationConfigError(
                "pytest plugin module name cannot be protected safely"
            )
        names.add(name)
        index += 1
    return names


def _pytest_plugins_from_python_source(text: str, relative: str) -> set[str]:
    if "pytest_plugins" not in text:
        return set()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise VerificationConfigError(
            f"conftest.py cannot be parsed safely: {relative}"
        ) from exc

    names: set[str] = set()
    allowed_targets: set[int] = set()
    for node in ast.walk(tree):
        value: ast.expr | None = None
        targets: tuple[ast.expr, ...] = ()
        if isinstance(node, ast.Assign):
            value = node.value
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = (node.target,)
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == "pytest_plugins"
            for target in targets
        ):
            continue
        allowed_targets.update(
            id(target)
            for target in targets
            if isinstance(target, ast.Name) and target.id == "pytest_plugins"
        )
        try:
            declared = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise VerificationConfigError(
                f"pytest_plugins must be a literal string or sequence: {relative}"
            ) from exc
        values = (declared,) if isinstance(declared, str) else declared
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(item, str) for item in values
        ):
            raise VerificationConfigError(
                f"pytest_plugins must be a literal string or sequence: {relative}"
            )
        for item in values:
            for name in item.split(","):
                normalized = name.strip()
                if _PYTHON_MODULE_NAME.fullmatch(normalized) is None:
                    raise VerificationConfigError(
                        f"pytest plugin module name cannot be protected: {relative}"
                    )
                names.add(normalized)
    if any(
        (
            isinstance(node, ast.Name)
            and node.id == "pytest_plugins"
            and id(node) not in allowed_targets
        )
        or (
            isinstance(node, ast.Constant)
            and node.value == "pytest_plugins"
        )
        for node in ast.walk(tree)
    ):
        raise VerificationConfigError(
            f"pytest_plugins must use a literal assignment: {relative}"
        )
    return names


def _pytest_entrypoint_modules(text: str) -> set[str]:
    names: set[str] = set()
    in_pytest_group = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_pytest_group = "pytest11" in line.casefold()
            continue
        if not in_pytest_group or not line or line.startswith(("#", ";")):
            continue
        if "=" not in line:
            continue
        value = line.split("=", 1)[1]
        module = _pytest_entrypoint_module(value)
        if module is None:
            raise VerificationConfigError(
                "workspace pytest plugin entry point has an unsafe target"
            )
        names.add(module)
    return names


def _pytest_entrypoint_module(value: str) -> str | None:
    normalized = value.strip().strip("\"'")
    target = normalized.split("[", 1)[0].strip()
    module = target.split(":", 1)[0].strip()
    if _PYTHON_MODULE_NAME.fullmatch(module) is None:
        return None
    return module


def _workspace_candidate_globs(path: str) -> tuple[str, ...]:
    patterns = [path, f"**/{path}"]
    case_insensitive = _case_insensitive_glob(path)
    patterns.extend((case_insensitive, f"**/{case_insensitive}"))
    return tuple(patterns)


def _case_insensitive_glob(path: str) -> str:
    return "".join(
        f"[{character.casefold()}{character.upper()}]"
        if "a" <= character.casefold() <= "z"
        else character
        for character in path
    )


def _workspace_entries(root: Path) -> tuple[Path, ...]:
    entries: list[Path] = []
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            children = sorted(os.scandir(directory), key=lambda item: item.name)
            for child in children:
                path = Path(child.path)
                link_like = is_link_like(path)
                if child.name.casefold() in _MANIFEST_SKIPPED_DIRECTORIES and (
                    link_like or child.is_dir(follow_symlinks=False)
                ):
                    continue
                if child.is_dir(follow_symlinks=False) and not link_like:
                    pending.append(path)
                else:
                    entries.append(path)
    except OSError as exc:
        raise ProtectedManifestError(
            f"protected paths cannot be enumerated: {type(exc).__name__}"
        ) from exc
    return tuple(sorted(entries, key=lambda path: path.relative_to(root).as_posix()))


def _protected_record(path: Path, relative: str) -> ProtectedFileRecord:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return ProtectedFileRecord(
            relative_path=relative,
            existed=False,
            size=None,
            sha256=None,
            file_kind="missing",
        )
    except OSError as exc:
        raise ProtectedManifestError(
            f"protected path metadata cannot be read: {relative}"
        ) from exc

    mode = details.st_mode
    if is_link_like(path):
        try:
            link_value = os.readlink(path).encode("utf-8", errors="surrogatepass")
        except OSError:
            link_value = f"reparse:{details.st_size}".encode("ascii")
        return ProtectedFileRecord(
            relative_path=relative,
            existed=True,
            size=details.st_size,
            sha256=hashlib.sha256(link_value).hexdigest(),
            file_kind="link",
        )
    if stat.S_ISREG(mode):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(128 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ProtectedManifestError(
                f"protected file cannot be hashed: {relative}"
            ) from exc
        return ProtectedFileRecord(
            relative_path=relative,
            existed=True,
            size=details.st_size,
            sha256=digest.hexdigest(),
            file_kind="file",
        )
    if stat.S_ISDIR(mode):
        file_kind = "directory"
    else:
        file_kind = "other"
    return ProtectedFileRecord(
        relative_path=relative,
        existed=True,
        size=details.st_size,
        sha256=None,
        file_kind=file_kind,
    )


def _run_verification_command(
    runner: CommandRunner,
    command: VerificationCommandSpec,
) -> VerificationCommandResult:
    try:
        metadata = runner.run(
            list(command.argv),
            cwd=command.cwd,
            timeout_seconds=command.timeout_seconds,
        )
    except ToolExecutionError as exc:
        metadata = exc.metadata
        if exc.kind is ErrorKind.COMMAND_TIMEOUT:
            error_kind = ErrorKind.VERIFICATION_TIMEOUT
            started = True
            timed_out = True
        elif exc.kind is ErrorKind.COMMAND_NONZERO_EXIT:
            error_kind = ErrorKind.VERIFICATION_FAILED
            started = True
            timed_out = False
        else:
            error_kind = ErrorKind.VERIFICATION_START_ERROR
            started = False
            timed_out = False
        return _command_result(
            command,
            metadata,
            started=started,
            timed_out=timed_out,
            error_kind=error_kind,
        )

    return _command_result(
        command,
        metadata,
        started=True,
        timed_out=False,
        error_kind=None,
    )


def _command_result(
    command: VerificationCommandSpec,
    metadata: dict[str, Any],
    *,
    started: bool,
    timed_out: bool,
    error_kind: ErrorKind | None,
) -> VerificationCommandResult:
    raw_argv = metadata.get("argv", command.argv)
    if isinstance(raw_argv, (list, tuple)) and all(
        isinstance(item, str) for item in raw_argv
    ):
        argv = tuple(raw_argv)
    else:
        argv = command.argv
    exit_code = metadata.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        exit_code = None
    duration_ms = metadata.get("duration_ms", 0)
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool):
        duration_ms = 0
    return VerificationCommandResult(
        argv=argv,
        cwd=str(metadata.get("cwd", command.cwd)),
        exit_code=exit_code,
        timed_out=timed_out,
        started=started,
        stdout=str(metadata.get("stdout", "")),
        stderr=str(metadata.get("stderr", "")),
        stdout_truncated=metadata.get("stdout_truncated") is True,
        stderr_truncated=metadata.get("stderr_truncated") is True,
        duration_ms=max(0, duration_ms),
        error_kind=error_kind,
    )
