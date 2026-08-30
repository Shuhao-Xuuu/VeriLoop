from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import pytest

import veriloop.process as process_module
from veriloop.filesystem import WorkspaceGuard
from veriloop.process import CommandPolicy, CommandRunner
from veriloop.protocol import ErrorKind, ToolCall
from veriloop.tools import ToolExecutionError, ToolRegistry, register_process_tool


def write_script(workspace: Path, name: str, source: str) -> Path:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def make_runner(
    workspace: Path,
    *,
    output_limit_bytes: int = 16 * 1024,
) -> tuple[CommandRunner, ToolRegistry]:
    guard = WorkspaceGuard(workspace)
    runner = CommandRunner(
        guard,
        CommandPolicy(),
        output_limit_bytes=output_limit_bytes,
    )
    registry = ToolRegistry()
    register_process_tool(registry, runner)
    return runner, registry


def execute(
    registry: ToolRegistry,
    arguments: dict[str, object],
    *,
    call_id: str = "command-1",
):
    return registry.execute(
        ToolCall(id=call_id, name="run_command", arguments=arguments)
    )


def payload(result) -> dict[str, object]:
    return json.loads(result.content)


def test_run_command_exit_zero_stdout_and_stderr(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(
        workspace,
        "emit.py",
        "import sys\nprint('stdout-value')\nprint('stderr-value', file=sys.stderr)\n",
    )
    _, registry = make_runner(workspace)

    result = execute(registry, {"argv": [sys.executable, "emit.py"]})
    data = payload(result)

    assert result.ok
    assert result.call_id == "command-1"
    assert data["exit_code"] == 0
    assert data["stdout"].splitlines() == ["stdout-value"]
    assert data["stderr"].splitlines() == ["stderr-value"]
    assert data["stdout_total_bytes"] == len(data["stdout"].encode())
    assert data["stderr_total_bytes"] == len(data["stderr"].encode())
    assert data["stdout_truncated"] is False
    assert data["stderr_truncated"] is False
    assert data["started"] is True
    assert result.invalidates_verification is True
    assert isinstance(data["duration_ms"], int) and data["duration_ms"] >= 0


def test_nonzero_exit_is_structured_tool_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(
        workspace,
        "fail.py",
        "import sys\nprint('failure detail', file=sys.stderr)\nraise SystemExit(7)\n",
    )
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {"argv": [sys.executable, "fail.py"]},
        call_id="nonzero-7",
    )
    data = payload(result)

    assert result.ok is False
    assert result.error_kind is ErrorKind.COMMAND_NONZERO_EXIT
    assert result.call_id == "nonzero-7"
    assert result.metadata["exit_code"] == 7
    assert result.metadata["stderr"].splitlines() == ["failure detail"]
    assert result.metadata["started"] is True
    assert result.invalidates_verification is True
    assert data["error_kind"] == "command_nonzero_exit"
    assert data["details"]["exit_code"] == 7


def test_timeout_ends_direct_child_and_returns_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    heartbeat = workspace / "heartbeat.txt"
    write_script(
        workspace,
        "slow.py",
        """from pathlib import Path
import os
import time
Path('pid.txt').write_text(str(os.getpid()), encoding='utf-8')
while True:
    with Path('heartbeat.txt').open('a', encoding='utf-8') as stream:
        stream.write('beat\\n')
        stream.flush()
    time.sleep(0.02)
""",
    )
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {
            "argv": [sys.executable, "slow.py"],
            "timeout_seconds": 1,
        },
    )
    size_after_return = heartbeat.stat().st_size
    time.sleep(0.2)

    assert result.error_kind is ErrorKind.COMMAND_TIMEOUT
    assert result.metadata["exit_code"] is not None
    assert result.metadata["duration_ms"] >= 1000
    assert result.metadata["started"] is True
    assert result.invalidates_verification is True
    assert heartbeat.stat().st_size == size_after_return
    assert (workspace / "pid.txt").read_text(encoding="utf-8").isdigit()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup only")
def test_timeout_ends_posix_process_group(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    heartbeat = workspace / "grandchild-heartbeat.txt"
    write_script(
        workspace,
        "grandchild.py",
        """from pathlib import Path
import time
while True:
    with Path('grandchild-heartbeat.txt').open('a', encoding='utf-8') as stream:
        stream.write('beat\\n')
        stream.flush()
    time.sleep(0.02)
""",
    )
    write_script(
        workspace,
        "parent.py",
        """import subprocess
import sys
import time
subprocess.Popen([sys.executable, 'grandchild.py'])
while True:
    time.sleep(1)
""",
    )
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {"argv": [sys.executable, "parent.py"], "timeout_seconds": 1},
    )
    size_after_return = heartbeat.stat().st_size
    time.sleep(0.2)

    assert result.error_kind is ErrorKind.COMMAND_TIMEOUT
    assert heartbeat.stat().st_size == size_after_return


def test_large_stdout_and_stderr_keep_head_tail_and_byte_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stdout_bytes = b"OUT_HEAD" + (b"x" * 50_000) + b"OUT_TAIL"
    stderr_bytes = b"ERR_HEAD" + (b"y" * 60_000) + b"ERR_TAIL"
    write_script(
        workspace,
        "large_output.py",
        """import sys
sys.stdout.buffer.write(b'OUT_HEAD' + (b'x' * 50000) + b'OUT_TAIL')
sys.stderr.buffer.write(b'ERR_HEAD' + (b'y' * 60000) + b'ERR_TAIL')
""",
    )
    _, registry = make_runner(workspace, output_limit_bytes=1024)

    result = execute(registry, {"argv": [sys.executable, "large_output.py"]})
    data = payload(result)

    assert result.ok
    assert data["stdout_total_bytes"] == len(stdout_bytes)
    assert data["stderr_total_bytes"] == len(stderr_bytes)
    assert data["stdout_truncated"] is True
    assert data["stderr_truncated"] is True
    assert data["stdout"].startswith("OUT_HEAD")
    assert data["stdout"].endswith("OUT_TAIL")
    assert data["stderr"].startswith("ERR_HEAD")
    assert data["stderr"].endswith("ERR_TAIL")
    assert "bytes omitted" in data["stdout"]
    assert "bytes omitted" in data["stderr"]
    assert len(data["stdout"].encode()) <= 1024
    assert len(data["stderr"].encode()) <= 1024


def test_non_utf8_output_preview_remains_bounded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(
        workspace,
        "invalid_output.py",
        "import sys\nsys.stdout.buffer.write(b'HEAD' + (b'\\xff' * 50000) + b'TAIL')\n",
    )
    _, registry = make_runner(workspace, output_limit_bytes=1024)

    result = execute(registry, {"argv": [sys.executable, "invalid_output.py"]})
    data = payload(result)

    assert result.ok
    assert data["stdout_truncated"] is True
    assert data["stdout_total_bytes"] == 50_008
    assert data["stdout"].startswith("HEAD")
    assert data["stdout"].endswith("TAIL")
    assert "bytes omitted" in data["stdout"]
    assert len(data["stdout"].encode("utf-8")) <= 1024


def test_cwd_is_relative_workspace_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(
        workspace,
        "sub/check_cwd.py",
        "from pathlib import Path\nprint(Path.cwd().name)\n",
    )
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {"argv": [sys.executable, "check_cwd.py"], "cwd": "sub"},
    )
    data = payload(result)

    assert result.ok
    assert data["cwd"] == "sub"
    assert data["stdout"].splitlines() == ["sub"]


@pytest.mark.parametrize("cwd", ["../", "../../", "C:\\outside"])
def test_cwd_escape_is_rejected(tmp_path: Path, cwd: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(workspace, "ok.py", "print('ok')\n")
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {"argv": [sys.executable, "ok.py"], "cwd": cwd},
    )

    assert result.error_kind is ErrorKind.PATH_OUTSIDE_WORKSPACE


def test_absolute_missing_and_file_cwd_are_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(workspace, "ok.py", "print('ok')\n")
    (workspace / "not-a-directory").write_text("x", encoding="utf-8")
    _, registry = make_runner(workspace)

    absolute = execute(
        registry,
        {"argv": [sys.executable, "ok.py"], "cwd": str(workspace)},
    )
    missing = execute(
        registry,
        {"argv": [sys.executable, "ok.py"], "cwd": "missing"},
    )
    file_cwd = execute(
        registry,
        {"argv": [sys.executable, "ok.py"], "cwd": "not-a-directory"},
    )

    assert absolute.error_kind is ErrorKind.PATH_OUTSIDE_WORKSPACE
    assert missing.error_kind is ErrorKind.PATH_NOT_FOUND
    assert file_cwd.error_kind is ErrorKind.COMMAND_INVALID


@pytest.mark.parametrize("argv", [[], "python", [sys.executable, 3]])
def test_registry_rejects_invalid_argv_schema(tmp_path: Path, argv: object) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry = make_runner(workspace)

    result = execute(registry, {"argv": argv})

    assert result.error_kind is ErrorKind.INVALID_ARGUMENTS


@pytest.mark.parametrize("argv", [[], ("python",), ["python", 3]])
def test_policy_rejects_invalid_argv(argv: object) -> None:
    with pytest.raises(ToolExecutionError) as captured:
        CommandPolicy().validate(argv)

    assert captured.value.kind is ErrorKind.COMMAND_INVALID


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-c", "echo no"],
        ["sh", "-c", "echo no"],
        ["zsh", "-c", "echo no"],
        ["cmd", "/c", "echo no"],
        ["powershell", "-Command", "echo no"],
        ["powershell.exe", "-Command", "echo no"],
        ["pwsh", "-Command", "echo no"],
        ["sudo", "git", "status"],
        ["rm", "file"],
        ["rmdir", "folder"],
        ["curl", "https://example.invalid"],
        ["wget", "https://example.invalid"],
        ["python", "-c", "print('no')"],
        ["python", "-cprint('no')"],
        ["python", "-m", "pip", "install", "thing"],
        ["pip", "install", "thing"],
        ["npm", "install"],
        ["pnpm", "install"],
        ["yarn", "install"],
    ],
)
def test_command_policy_denies_shells_downloaders_and_installers(
    argv: list[str],
) -> None:
    with pytest.raises(ToolExecutionError) as captured:
        CommandPolicy().validate(argv)

    assert captured.value.kind is ErrorKind.COMMAND_DENIED


@pytest.mark.parametrize("subcommand", ["status", "diff", "log", "show", "ls-files", "rev-parse"])
def test_command_policy_allows_read_only_git(subcommand: str) -> None:
    assert CommandPolicy().validate(["git", subcommand]) == ("git", subcommand)


@pytest.mark.parametrize("program", ["./git", "tools/git", ".\\git.exe", "C:\\tools\\git.exe"])
def test_command_policy_rejects_git_path_spoof(program: str) -> None:
    with pytest.raises(ToolExecutionError) as captured:
        CommandPolicy().validate([program, "status"])

    assert captured.value.kind is ErrorKind.COMMAND_DENIED


@pytest.mark.parametrize(
    "argv",
    [
        ["pip", "--quiet", "install", "thing"],
        ["npm", "i", "thing"],
        ["yarn", "add", "thing"],
    ],
)
def test_additional_allowlist_cannot_enable_package_managers(
    argv: list[str],
) -> None:
    with pytest.raises(ToolExecutionError) as captured:
        CommandPolicy([argv[0]]).validate(argv)

    assert captured.value.kind is ErrorKind.COMMAND_DENIED


def test_additional_program_paths_are_absolute_and_bound_to_that_path(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "tools" / "helper"
    executable.parent.mkdir()
    executable.write_text("placeholder", encoding="utf-8")

    for relative_path in ["tools/helper", "C:helper.exe"]:
        with pytest.raises(ValueError, match="must be absolute"):
            CommandPolicy([relative_path])

    policy = CommandPolicy([str(executable)])
    validated = policy.validate([str(executable), "argument"])
    assert Path(validated[0]) == executable.resolve()


def test_allowed_bare_name_does_not_allow_a_path_with_same_basename() -> None:
    policy = CommandPolicy(["helper"])

    with pytest.raises(ToolExecutionError) as captured:
        policy.validate(["tools/helper", "argument"])

    assert captured.value.kind is ErrorKind.COMMAND_DENIED


@pytest.mark.parametrize(
    "subcommand",
    [
        "commit",
        "push",
        "reset",
        "clean",
        "checkout",
        "restore",
        "rebase",
        "merge",
        "cherry-pick",
    ],
)
def test_command_policy_denies_mutating_git(subcommand: str) -> None:
    with pytest.raises(ToolExecutionError) as captured:
        CommandPolicy().validate(["git", subcommand])

    assert captured.value.kind is ErrorKind.COMMAND_DENIED


@pytest.mark.parametrize("option", ["--output=leak.txt", "--ext-diff", "--textconv"])
def test_command_policy_denies_git_side_effect_options(option: str) -> None:
    with pytest.raises(ToolExecutionError) as captured:
        CommandPolicy().validate(["git", "diff", option])

    assert captured.value.kind is ErrorKind.COMMAND_DENIED


def test_python_modules_and_workspace_script_are_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = write_script(workspace, "ok.py", "print('ok')\n")
    policy = CommandPolicy()

    assert policy.validate([sys.executable, "-m", "pytest"])[1:] == ("-m", "pytest")
    assert policy.validate(["python3", "-m", "unittest"])[1:] == ("-m", "unittest")
    assert policy.validate(
        [sys.executable, "ok.py"],
        cwd=workspace,
        workspace_root=workspace,
    )[1] == script.name


def test_python_script_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(tmp_path, "outside.py", "print('outside')\n")
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {"argv": [sys.executable, "../outside.py"]},
    )

    assert result.error_kind is ErrorKind.COMMAND_DENIED


def test_timeout_above_maximum_is_rejected_by_schema_and_runner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(workspace, "ok.py", "print('ok')\n")
    runner, registry = make_runner(workspace)

    schema_result = execute(
        registry,
        {"argv": [sys.executable, "ok.py"], "timeout_seconds": 121},
    )
    with pytest.raises(ToolExecutionError) as captured:
        runner.run([sys.executable, "ok.py"], timeout_seconds=121)

    assert schema_result.error_kind is ErrorKind.INVALID_ARGUMENTS
    assert captured.value.kind is ErrorKind.COMMAND_INVALID


def test_shell_argument_is_not_part_of_run_command_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(workspace, "ok.py", "print('ok')\n")
    _, registry = make_runner(workspace)

    result = execute(
        registry,
        {"argv": [sys.executable, "ok.py"], "shell": True},
    )

    assert result.error_kind is ErrorKind.INVALID_ARGUMENTS


def test_start_error_becomes_command_start_error(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(workspace, "ok.py", "print('ok')\n")
    _, registry = make_runner(workspace)

    def fail_start(*args, **kwargs):
        raise OSError("fictional start failure")

    monkeypatch.setattr(process_module.subprocess, "Popen", fail_start)
    result = execute(registry, {"argv": [sys.executable, "ok.py"]})

    assert result.error_kind is ErrorKind.COMMAND_START_ERROR
    assert result.metadata["started"] is False
    assert result.invalidates_verification is False
    assert "Traceback" not in result.content
    assert "fictional start failure" not in result.content


def test_runner_always_starts_with_shell_false(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_script(workspace, "ok.py", "print('ok')\n")
    _, registry = make_runner(workspace)
    original_popen = process_module.subprocess.Popen
    seen_shell_values: list[object] = []

    def recording_popen(*args, **kwargs):
        seen_shell_values.append(kwargs.get("shell"))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(process_module.subprocess, "Popen", recording_popen)

    result = execute(registry, {"argv": [sys.executable, "ok.py"]})

    assert result.ok
    assert seen_shell_values == [False]


def test_child_environment_filters_fictional_secrets_and_keeps_path(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    names = [
        "OPENAI_API_KEY",
        "FICTIONAL_TOKEN",
        "FICTIONAL_SECRET",
        "FICTIONAL_PASSWORD",
        "FICTIONAL_CREDENTIAL",
    ]
    write_script(
        workspace,
        "environment.py",
        """import json
import os
names = [
    'OPENAI_API_KEY',
    'FICTIONAL_TOKEN',
    'FICTIONAL_SECRET',
    'FICTIONAL_PASSWORD',
    'FICTIONAL_CREDENTIAL',
]
print(json.dumps({name: name in os.environ for name in names} | {'PATH_PRESENT': bool(os.environ.get('PATH'))}))
""",
    )
    for name in names:
        monkeypatch.setenv(name, "fictional-test-value")
    _, registry = make_runner(workspace)

    result = execute(registry, {"argv": [sys.executable, "environment.py"]})
    child_environment = json.loads(payload(result)["stdout"])

    assert result.ok
    assert all(child_environment[name] is False for name in names)
    assert child_environment["PATH_PRESENT"] is True
    assert "fictional-test-value" not in result.content


def test_environment_filter_never_reads_disallowed_secret_values() -> None:
    class GuardedEnvironment(dict[str, str]):
        def __getitem__(self, name: str) -> str:
            if name == "OPENAI_API_KEY":
                raise AssertionError("secret value was accessed")
            return super().__getitem__(name)

    source = GuardedEnvironment(
        {
            "PATH": "fictional-path",
            "OPENAI_API_KEY": "must-not-be-read",
        }
    )

    assert process_module.filtered_subprocess_environment(source) == {
        "PATH": "fictional-path"
    }
