"""Frozen verification configuration and host-side verification primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import stat
import tomllib
from typing import Any

from .filesystem import (
    SKIPPED_DIRECTORIES,
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
            return VerificationResult(
                phase=VerificationPhase.FINAL,
                passed=False,
                commands=(),
                protected_unchanged=not before_changes,
                protected_changes=before_changes,
                mutation_seq=mutation_seq,
                verified_seq=None,
                failure_kind=(
                    ErrorKind.PROTECTED_FILE_CHANGED
                    if before_changes
                    else ErrorKind.VERIFICATION_FAILED
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

    if spec.config_path is not None:
        config_path = guard.lexical_path(spec.config_path)
        records[spec.config_path] = _protected_record(
            config_path,
            spec.config_path,
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


def protected_guard_for_spec(
    guard: WorkspaceGuard,
    spec: VerificationSpec,
) -> WorkspaceGuard:
    """Create the model-facing guard with frozen verification write denies."""

    exact_paths = (spec.config_path,) if spec.config_path is not None else ()
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
                if child.name.casefold() in SKIPPED_DIRECTORIES and (
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
