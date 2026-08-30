"""Allowlisted local command execution with bounded model-visible output."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
import signal
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO, Iterable, Mapping

from .filesystem import WorkspaceGuard
from .protocol import ErrorKind
from .tools import ToolExecutionError


DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 120
DEFAULT_OUTPUT_LIMIT_BYTES = 16 * 1024

_PYTHON_NAMES = frozenset({"python", "python3", "py"})
_PYTHON_MODULES = frozenset({"pytest", "unittest", "compileall"})
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {"status", "diff", "log", "show", "ls-files", "rev-parse"}
)
_HARD_DENIED_PROGRAMS = frozenset(
    {
        "sudo",
        "su",
        "rm",
        "rmdir",
        "del",
        "erase",
        "chmod",
        "chown",
        "shutdown",
        "reboot",
        "mkfs",
        "dd",
        "curl",
        "wget",
        "ssh",
        "scp",
        "powershell",
        "pwsh",
        "cmd",
        "bash",
        "sh",
        "zsh",
    }
)
_PACKAGE_PROGRAMS = frozenset({"pip", "pip3", "npm", "pnpm", "yarn"})
_DENIED_GIT_OPTIONS = (
    "--output",
    "--ext-diff",
    "--textconv",
    "--config-env",
    "--exec-path",
)
_ALLOWED_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "SYSTEMROOT",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "PATHEXT",
        "CI",
        "TERM",
    }
)
_SENSITIVE_ENVIRONMENT_WORDS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
)


class CommandPolicy:
    """Reject shell hosts and allow only explicit program shapes."""

    def __init__(self, additional_allowed_programs: Iterable[str] = ()) -> None:
        allowed_names: set[str] = set()
        allowed_paths: set[str] = set()
        for item in additional_allowed_programs:
            if not isinstance(item, str) or not item or "\x00" in item:
                raise ValueError("additional allowed programs must be non-empty strings")
            if not _is_bare_program(item):
                if not Path(item).is_absolute():
                    raise ValueError(
                        "additional allowed program paths must be absolute"
                    )
                allowed_paths.add(_normalized_program_path(item))
            else:
                allowed_names.add(_program_name(item))
        self._allowed_names = frozenset(allowed_names)
        self._allowed_paths = frozenset(allowed_paths)
        self._current_python = _normalized_program_path(sys.executable)

    def validate(
        self,
        argv: object,
        *,
        cwd: Path | None = None,
        workspace_root: Path | None = None,
    ) -> tuple[str, ...]:
        if not isinstance(argv, list):
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                "argv must be a list of strings",
            )
        if not argv:
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                "argv must not be empty",
            )
        if any(not isinstance(item, str) for item in argv):
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                "every argv element must be a string",
            )
        if any(item == "" or "\x00" in item for item in argv):
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                "argv elements must be non-empty and contain no NUL characters",
            )

        copied = tuple(argv)
        program = copied[0]
        name = _program_name(program)
        normalized_path = _normalized_program_path(program)

        if name in _HARD_DENIED_PROGRAMS or _is_batch_file(program):
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                f"program is denied by command policy: {name}",
            )

        if name == "git":
            if not _is_bare_program(program):
                raise ToolExecutionError(
                    ErrorKind.COMMAND_DENIED,
                    "git must be invoked by its bare program name",
                )
            self._validate_git(copied)
            return copied

        is_python = name in _PYTHON_NAMES or normalized_path == self._current_python
        if is_python:
            self._validate_python(copied, cwd=cwd, workspace_root=workspace_root)
            return (sys.executable, *copied[1:])

        if name in _PACKAGE_PROGRAMS:
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                "package-manager execution is denied",
            )

        if _is_bare_program(program) and name in self._allowed_names:
            return copied
        if normalized_path in self._allowed_paths:
            return (normalized_path, *copied[1:])

        raise ToolExecutionError(
            ErrorKind.COMMAND_DENIED,
            f"program is not allowlisted: {name}",
        )

    @staticmethod
    def _validate_git(argv: tuple[str, ...]) -> None:
        if len(argv) < 2 or argv[1].casefold() not in _READ_ONLY_GIT_SUBCOMMANDS:
            subcommand = argv[1] if len(argv) > 1 else "<missing>"
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                f"git subcommand is not read-only allowlisted: {subcommand}",
            )
        for argument in argv[2:]:
            lowered = argument.casefold()
            if any(
                lowered == option or lowered.startswith(f"{option}=")
                for option in _DENIED_GIT_OPTIONS
            ):
                raise ToolExecutionError(
                    ErrorKind.COMMAND_DENIED,
                    f"git option is denied: {argument}",
                )

    @staticmethod
    def _validate_python(
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        workspace_root: Path | None,
    ) -> None:
        arguments = argv[1:]
        if not arguments:
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                "interactive Python is denied",
            )
        if arguments[0] == "-c" or arguments[0].startswith("-c"):
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                "python -c is denied",
            )
        if arguments[0] == "-m":
            if len(arguments) < 2 or arguments[1].casefold() not in _PYTHON_MODULES:
                module = arguments[1] if len(arguments) > 1 else "<missing>"
                raise ToolExecutionError(
                    ErrorKind.COMMAND_DENIED,
                    f"Python module is not allowlisted: {module}",
                )
            return
        if arguments[0].startswith("-"):
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                f"Python interpreter option is denied: {arguments[0]}",
            )

        script_argument = arguments[0]
        script_path = Path(script_argument)
        windows_path = PureWindowsPath(script_argument)
        if (
            script_path.is_absolute()
            or script_path.anchor
            or script_path.drive
            or windows_path.anchor
            or windows_path.drive
            or script_path.suffix.casefold() != ".py"
        ):
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                "Python scripts must be relative .py files inside the workspace",
            )
        if cwd is None or workspace_root is None:
            return
        try:
            resolved_script = (cwd / script_path).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                "Python script path cannot be resolved safely",
            ) from exc
        if not resolved_script.is_relative_to(workspace_root):
            raise ToolExecutionError(
                ErrorKind.COMMAND_DENIED,
                "Python script path escapes the workspace",
            )
        if not resolved_script.exists() or not resolved_script.is_file():
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                f"Python script does not exist: {script_argument}",
            )


@dataclass(slots=True)
class CommandRunner:
    guard: WorkspaceGuard
    policy: CommandPolicy
    max_timeout_seconds: int = MAX_TIMEOUT_SECONDS
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES
    termination_grace_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.max_timeout_seconds < 1:
            raise ValueError("max_timeout_seconds must be positive")
        if self.output_limit_bytes < 256:
            raise ValueError("output_limit_bytes must be at least 256")
        if self.termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive")

    def __call__(self, arguments: dict[str, object]) -> dict[str, object]:
        return self.run(
            arguments.get("argv"),
            cwd=arguments.get("cwd", "."),
            timeout_seconds=arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        )

    def run(
        self,
        argv: object,
        *,
        cwd: object = ".",
        timeout_seconds: object = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, object]:
        syntactic_argv = self.policy.validate(argv)
        timeout = self._validate_timeout(timeout_seconds)
        cwd_path = self._resolve_cwd(cwd)
        command = self.policy.validate(
            list(syntactic_argv),
            cwd=cwd_path,
            workspace_root=self.guard.root,
        )
        environment = filtered_subprocess_environment(os.environ)
        started = time.monotonic()

        with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
            mode="w+b"
        ) as stderr_file:
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd_path,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    shell=False,
                    **_process_group_options(),
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                duration_ms = _duration_ms(started)
                raise ToolExecutionError(
                    ErrorKind.COMMAND_START_ERROR,
                    f"command could not be started: {type(exc).__name__}",
                    metadata={
                        "argv": list(command),
                        "cwd": self.guard.relative(cwd_path),
                        "duration_ms": duration_ms,
                        "started": False,
                    },
                ) from exc

            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process(process, self.termination_grace_seconds)
            except BaseException:
                _terminate_process(process, self.termination_grace_seconds)
                raise

            result = self._result(
                command,
                cwd_path,
                process,
                stdout_file,
                stderr_file,
                started,
            )

        if timed_out:
            raise ToolExecutionError(
                ErrorKind.COMMAND_TIMEOUT,
                f"command exceeded the {timeout}-second timeout",
                metadata=result,
                invalidates_verification=True,
            )
        if process.returncode != 0:
            raise ToolExecutionError(
                ErrorKind.COMMAND_NONZERO_EXIT,
                f"command exited with code {process.returncode}",
                metadata=result,
                invalidates_verification=True,
            )
        return result

    def _validate_timeout(self, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                "timeout_seconds must be an integer",
            )
        if value < 1 or value > self.max_timeout_seconds:
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                f"timeout_seconds must be between 1 and {self.max_timeout_seconds}",
            )
        return value

    def _resolve_cwd(self, value: object) -> Path:
        if not isinstance(value, str) or value == "":
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                "cwd must be a non-empty relative path",
            )
        cwd_path = self.guard.resolve(value, allow_root=True)
        if not cwd_path.exists():
            raise ToolExecutionError(
                ErrorKind.PATH_NOT_FOUND,
                f"cwd does not exist: {value}",
            )
        if not cwd_path.is_dir():
            raise ToolExecutionError(
                ErrorKind.COMMAND_INVALID,
                f"cwd is not a directory: {value}",
            )
        return cwd_path

    def _result(
        self,
        command: tuple[str, ...],
        cwd: Path,
        process: subprocess.Popen[bytes],
        stdout_file: BinaryIO,
        stderr_file: BinaryIO,
        started: float,
    ) -> dict[str, object]:
        stdout, stdout_total, stdout_truncated = _output_preview(
            stdout_file,
            self.output_limit_bytes,
        )
        stderr, stderr_total, stderr_truncated = _output_preview(
            stderr_file,
            self.output_limit_bytes,
        )
        return {
            "argv": list(command),
            "cwd": self.guard.relative(cwd),
            "exit_code": process.returncode,
            "duration_ms": _duration_ms(started),
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_total_bytes": stdout_total,
            "stderr_total_bytes": stderr_total,
            "started": True,
        }


def filtered_subprocess_environment(
    source: Mapping[str, str],
) -> dict[str, str]:
    """Build a minimal child environment without provider or common secret names."""

    filtered: dict[str, str] = {}
    for name in source:
        normalized = name.upper()
        if normalized not in _ALLOWED_ENVIRONMENT_NAMES:
            continue
        if any(word in normalized for word in _SENSITIVE_ENVIRONMENT_WORDS):
            continue
        filtered[name] = source[name]
    return filtered


def _process_group_options() -> dict[str, object]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creation_flag}
    return {}


def _terminate_process(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    if os.name == "posix":
        _terminate_posix_group(process, grace_seconds)
        return

    if process.poll() is None:
        with suppress(OSError):
            process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            process.kill()
    finally:
        try:
            process.wait(timeout=max(grace_seconds, 0.1))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _terminate_posix_group(
    process: subprocess.Popen[bytes],
    grace_seconds: float,
) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _posix_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _posix_group_exists(process.pid):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=max(grace_seconds, 0.1))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _posix_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _output_preview(stream: BinaryIO, limit: int) -> tuple[str, int, bool]:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    total = stream.tell()
    if total <= limit:
        stream.seek(0)
        data = stream.read()
        text = data.decode("utf-8", errors="replace")
        if len(text.encode("utf-8")) <= limit:
            return text, total, False

    selected = min(total - 1, max(limit - 64, 2))
    while True:
        head_size = selected // 2
        tail_size = selected - head_size
        omitted = total - selected
        marker = f"\n... <{omitted} bytes omitted> ...\n"
        stream.seek(0)
        head = stream.read(head_size).decode("utf-8", errors="replace")
        stream.seek(-tail_size, os.SEEK_END)
        tail = stream.read(tail_size).decode("utf-8", errors="replace")
        preview = head + marker + tail
        encoded_size = len(preview.encode("utf-8"))
        if encoded_size <= limit:
            return preview, total, True

        marker_size = len(marker.encode("utf-8"))
        body_size = max(encoded_size - marker_size, 1)
        body_budget = max(limit - marker_size, 2)
        reduced = max(2, selected * body_budget // body_size)
        selected = min(selected - 1, reduced)


def _duration_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _program_name(program: str) -> str:
    name = program.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return name[:-4] if name.endswith(".exe") else name


def _normalized_program_path(program: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(program))
    except (OSError, ValueError):
        return program.casefold()


def _has_path_separator(value: str) -> bool:
    return "/" in value or "\\" in value


def _is_bare_program(value: str) -> bool:
    windows_path = PureWindowsPath(value)
    return not (
        _has_path_separator(value)
        or Path(value).is_absolute()
        or Path(value).anchor
        or Path(value).drive
        or windows_path.anchor
        or windows_path.drive
    )


def _is_batch_file(value: str) -> bool:
    lowered = value.casefold()
    return lowered.endswith(".bat") or lowered.endswith(".cmd")
