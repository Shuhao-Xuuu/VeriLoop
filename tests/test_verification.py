from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import time

import pytest

from tests.scripted_model import ScriptedModel
import veriloop.process as process_module
from veriloop.agent import AgentLoop
from veriloop.protocol import (
    AgentResult,
    AgentState,
    ErrorKind,
    FinishReason,
    ModelResponse,
    ProtectedChangeKind,
    ProtectedFileChange,
    Role,
    VerificationCommandResult,
    VerificationPhase,
    VerificationResult,
    ToolCall,
)
from veriloop.filesystem import WorkspaceGuard
from veriloop.process import CommandPolicy, CommandRunner
from veriloop.tools import (
    ToolRegistry,
    register_filesystem_tools,
    register_workspace_tools,
)
from veriloop.verification import (
    BaselinePolicy,
    VerificationConfigError,
    VerificationGate,
    build_protected_manifest,
    compare_protected_manifests,
    load_verification_spec,
    protected_guard_for_spec,
)


def write_config(workspace: Path, text: str) -> Path:
    path = workspace / ".veriloop.toml"
    path.write_text(text, encoding="utf-8")
    return path


def load(workspace: Path):
    guard = WorkspaceGuard(workspace)
    runner = CommandRunner(guard, CommandPolicy())
    return load_verification_spec(guard, runner)


def gate(workspace: Path, text: str) -> VerificationGate:
    write_config(workspace, text)
    guard = WorkspaceGuard(workspace)
    runner = CommandRunner(guard, CommandPolicy())
    return VerificationGate(load_verification_spec(guard, runner), runner)


def final_response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=(),
        finish_reason=FinishReason.STOP,
    )


def tool_response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=tuple(calls),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def loaded_components(workspace: Path):
    guard = WorkspaceGuard(workspace)
    runner = CommandRunner(guard, CommandPolicy())
    spec = load_verification_spec(guard, runner)
    return guard, runner, spec


def execute_file_tool(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
):
    return registry.execute(ToolCall(id=f"{name}-call", name=name, arguments=arguments))


def verification_stack(workspace: Path, config_text: str):
    write_config(workspace, config_text)
    base_guard, _, spec = loaded_components(workspace)
    guarded = protected_guard_for_spec(base_guard, spec)
    runner = CommandRunner(guarded, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guarded, runner)
    return registry, VerificationGate(spec, runner)


def interrupt_first_process_wait_after_start(
    monkeypatch: pytest.MonkeyPatch,
    started_marker: Path,
) -> None:
    real_popen = process_module.subprocess.Popen

    def popen_then_interrupt(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        real_wait = process.wait
        first_wait = True

        def wait(*, timeout=None):
            nonlocal first_wait
            if first_wait:
                first_wait = False
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not started_marker.exists():
                    time.sleep(0.01)
                raise KeyboardInterrupt
            return real_wait(timeout=timeout)

        process.wait = wait
        return process

    monkeypatch.setattr(process_module.subprocess, "Popen", popen_then_interrupt)


def test_verification_protocol_exposes_explicit_terminal_states_and_errors() -> None:
    assert AgentState.VERIFIED.value == "verified"
    assert AgentState.VERIFICATION_FAILED.value == "verification_failed"
    assert AgentState.STALLED.value == "stalled"
    assert ErrorKind.PROTECTED_FILE_CHANGED.value == "protected_file_changed"
    assert ErrorKind.COMPLETION_MUST_BE_SINGLE_CALL.value == (
        "completion_must_be_single_call"
    )


def test_verification_results_are_frozen_provider_independent_evidence() -> None:
    command = VerificationCommandResult(
        argv=("python", "-m", "pytest", "-q"),
        cwd=".",
        exit_code=1,
        timed_out=False,
        started=True,
        stdout="",
        stderr="one failed",
        stdout_truncated=False,
        stderr_truncated=False,
        duration_ms=25,
        error_kind=ErrorKind.VERIFICATION_FAILED,
    )
    result = VerificationResult(
        phase=VerificationPhase.FINAL,
        passed=False,
        commands=(command,),
        protected_unchanged=False,
        protected_changes=(
            ProtectedFileChange("tests/test_value.py", ProtectedChangeKind.MODIFIED),
        ),
        mutation_seq=3,
        verified_seq=None,
        failure_kind=ErrorKind.PROTECTED_FILE_CHANGED,
    )

    assert result.commands[0].argv == ("python", "-m", "pytest", "-q")
    assert result.protected_changes[0].relative_path == "tests/test_value.py"
    with pytest.raises(FrozenInstanceError):
        result.passed = True  # type: ignore[misc]


def test_agent_result_verification_fields_have_backward_compatible_defaults() -> None:
    result = AgentResult(
        state=AgentState.COMPLETED_UNVERIFIED,
        final_message="plain final",
        step_count=1,
        tool_call_count=0,
        history=(),
    )

    assert result.mutation_seq == 0
    assert result.verified_seq is None
    assert result.baseline_verification is None
    assert result.final_verification is None
    assert result.repair_rounds_used == 0


def test_verification_config_loads_valid_frozen_values(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
max_repair_rounds = 3
max_same_failure = 4
protected_globs = ["tests/**", ".veriloop.toml"]

[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
cwd = "."
timeout_seconds = 90
""",
    )

    spec = load(tmp_path)

    assert spec.baseline_policy is BaselinePolicy.MUST_FAIL
    assert spec.max_repair_rounds == 3
    assert spec.max_same_failure == 4
    assert spec.protected_globs[:2] == ("tests/**", ".veriloop.toml")
    assert "pytest.py" in spec.protected_globs
    assert "sitecustomize.py" in spec.protected_globs
    assert "conftest.py" in spec.protected_globs
    assert "pytest.ini" in spec.protected_globs
    assert spec.commands[0].argv == ("python", "-m", "pytest", "-q")
    assert spec.commands[0].cwd == "."
    assert spec.commands[0].timeout_seconds == 90
    assert spec.config_path == ".veriloop.toml"
    with pytest.raises(FrozenInstanceError):
        spec.max_repair_rounds = 9  # type: ignore[misc]


def test_missing_config_freezes_an_empty_unverified_spec(tmp_path: Path) -> None:
    spec = load(tmp_path)

    assert spec.commands == ()
    assert spec.baseline_policy is BaselinePolicy.RECORD_ONLY
    assert spec.config_path == ".veriloop.toml"


def test_loaded_spec_does_not_follow_later_disk_changes(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )
    spec = load(tmp_path)

    config.write_text("[verification]\nbaseline_policy = \"must_fail\"\n", encoding="utf-8")

    assert spec.baseline_policy is BaselinePolicy.SKIP
    assert len(spec.commands) == 1


@pytest.mark.parametrize(
    "relative_path",
    [
        "pytest.py",
        "PyTeSt.py",
        "sitecustomize.py",
        "src/sitecustomize/__init__.py",
        "nested/usercustomize.py",
        "conftest.py",
        "pytest.ini",
        "nested/pyproject.toml",
    ],
)
def test_python_pytest_verifier_controls_are_implicitly_write_protected(
    tmp_path: Path,
    relative_path: str,
) -> None:
    (tmp_path / relative_path).parent.mkdir(parents=True, exist_ok=True)
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": relative_path,
            "content": "control = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / relative_path).exists()


def test_gate_rejects_command_side_creation_of_pytest_module_shadow(
    tmp_path: Path,
) -> None:
    _, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )
    (tmp_path / "pytest.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    result = verification_gate.run_final(mutation_seq=1)

    assert result.commands[0].exit_code == 0
    assert result.passed is False
    assert result.protected_unchanged is False
    assert result.protected_changes == (
        ProtectedFileChange("pytest.py", ProtectedChangeKind.CREATED),
    )
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.verified_seq is None
    assert verification_gate.grants_verified(result, mutation_seq=1) is False


def test_gate_manifests_workspace_python_site_pth_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    _, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )
    startup_control = site_packages / "force_green.pth"
    startup_control.write_text(
        "import os; os.environ['PYTEST_ADDOPTS'] = '-p force_green'\n",
        encoding="utf-8",
    )

    result = verification_gate.run_final(mutation_seq=1)

    relative = startup_control.relative_to(tmp_path).as_posix()
    assert result.passed is False
    assert result.protected_changes == (
        ProtectedFileChange(relative, ProtectedChangeKind.CREATED),
    )
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.verified_seq is None
    assert verification_gate.grants_verified(result, mutation_seq=1) is False


def test_gate_manifests_workspace_pythonpath_archive_creation(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "mutable-runtime.zip"
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(
        guard,
        CommandPolicy(),
        child_environment={"PYTHONPATH": str(archive)},
    )
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )
    spec = load_verification_spec(guard, runner)
    verification_gate = VerificationGate(spec, runner)
    archive.write_bytes(b"not-a-zip-yet")

    result = verification_gate.run_final(mutation_seq=1)

    assert result.passed is False
    assert result.protected_changes == (
        ProtectedFileChange("mutable-runtime.zip", ProtectedChangeKind.CREATED),
    )
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.verified_seq is None
    assert verification_gate.grants_verified(result, mutation_seq=1) is False


def test_workspace_python_pth_import_target_is_implicitly_write_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "existing_control.pth").write_text(
        "import force_startup\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "force_startup.py",
            "content": "startup = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "force_startup.py").exists()


def test_workspace_sitecustomize_import_target_is_implicitly_write_protected(
    tmp_path: Path,
) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        "import force_startup\n",
        encoding="utf-8",
    )
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "force_startup.py",
            "content": "startup = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "force_startup.py").exists()


def test_workspace_sitecustomize_dynamic_code_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        "import os\nos.environ['PYTEST_ADDOPTS'] = '-p force_green'\n",
        encoding="utf-8",
    )
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    with pytest.raises(VerificationConfigError, match="startup hook"):
        load(tmp_path)


def test_workspace_python_pth_dynamic_code_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "dynamic_control.pth").write_text(
        "import os; os.environ['PYTEST_ADDOPTS'] = '-p force_green'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    with pytest.raises(VerificationConfigError, match="executable .pth line"):
        load(tmp_path)


@pytest.mark.parametrize(
    "config_arguments",
    [
        ["-c", "verifier.ini"],
        ["-cverifier.ini"],
        ["--config-file", "verifier.ini"],
        ["--config-file=verifier.ini"],
    ],
)
def test_explicit_pytest_config_is_implicitly_write_protected(
    tmp_path: Path,
    config_arguments: list[str],
) -> None:
    command_dir = tmp_path / "checks"
    command_dir.mkdir()
    argv = ["python", "-m", "pytest", "-q", *config_arguments]
    registry, _ = verification_stack(
        tmp_path,
        "\n".join(
            [
                "[verification]",
                'baseline_policy = "skip"',
                "[[verification.commands]]",
                f"argv = {json.dumps(argv)}",
                'cwd = "checks"',
            ]
        ),
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "checks/verifier.ini",
            "content": "[pytest]\naddopts = --collect-only\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (command_dir / "verifier.ini").exists()


def test_explicit_pytest_config_outside_workspace_is_rejected(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-verifier.ini"
    write_config(
        tmp_path,
        "\n".join(
            [
                "[verification]",
                'baseline_policy = "skip"',
                "[[verification.commands]]",
                "argv = "
                + json.dumps(["python", "-m", "pytest", "-c", str(outside)]),
            ]
        ),
    )

    with pytest.raises(VerificationConfigError, match="stay inside the workspace"):
        load(tmp_path)


def test_explicit_pytest_config_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "actual.ini"
    target.write_text("[pytest]\n", encoding="utf-8")
    alias = tmp_path / "active.ini"
    try:
        os.symlink(target, alias)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this host: {exc}")
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-c", "active.ini"]
""",
    )

    with pytest.raises(VerificationConfigError, match="must not use a symlink"):
        load(tmp_path)


@pytest.mark.parametrize(
    "plugin_arguments",
    [
        ["-p", "verifier_plugin"],
        ["-pverifier_plugin"],
    ],
)
def test_explicit_pytest_plugin_module_is_implicitly_write_protected(
    tmp_path: Path,
    plugin_arguments: list[str],
) -> None:
    argv = ["python", "-m", "pytest", "-q", *plugin_arguments]
    registry, _ = verification_stack(
        tmp_path,
        "\n".join(
            [
                "[verification]",
                'baseline_policy = "skip"',
                "[[verification.commands]]",
                f"argv = {json.dumps(argv)}",
            ]
        ),
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "verifier_plugin.py",
            "content": (
                "def pytest_sessionfinish(session):\n"
                "    session.exitstatus = 0\n"
            ),
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "verifier_plugin.py").exists()


@pytest.mark.parametrize("plugin_name", ["verifier-plugin", "123verifier"])
def test_non_identifier_pytest_plugin_module_is_implicitly_write_protected(
    tmp_path: Path,
    plugin_name: str,
) -> None:
    registry, _ = verification_stack(
        tmp_path,
        f"""
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q", "-p", "{plugin_name}"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": f"{plugin_name}.py",
            "content": "plugin = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / f"{plugin_name}.py").exists()


def test_pytest_entrypoint_with_extras_protects_plugin_module(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "example.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "entry_points.txt").write_text(
        "[pytest11]\nexample = verifier_plugin [test]\n",
        encoding="utf-8",
    )
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "verifier_plugin.py",
            "content": "plugin = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "verifier_plugin.py").exists()


def test_installed_pytest_entrypoint_target_is_implicitly_write_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = importlib.metadata.EntryPoint(
        name="example",
        value="verifier_plugin",
        group="pytest11",
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: (entrypoint,) if kwargs == {"group": "pytest11"} else (),
    )
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "verifier_plugin.py",
            "content": "plugin = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "verifier_plugin.py").exists()


def test_pytest_config_declared_plugin_is_implicitly_write_protected(
    tmp_path: Path,
) -> None:
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = -p verifier_plugin\n",
        encoding="utf-8",
    )
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "verifier_plugin.py",
            "content": (
                "def pytest_sessionfinish(session):\n"
                "    session.exitstatus = 0\n"
            ),
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "verifier_plugin.py").exists()


def test_quoted_unsafe_pytest_config_plugin_name_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "pytest.ini").write_text(
        '[pytest]\naddopts = -p "evil plugin"\n',
        encoding="utf-8",
    )
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    with pytest.raises(VerificationConfigError, match="cannot be protected safely"):
        load(tmp_path)


def test_conftest_declared_plugin_is_implicitly_write_protected(
    tmp_path: Path,
) -> None:
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = ["verifier_plugin"]\n',
        encoding="utf-8",
    )
    registry, _ = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "verifier_plugin.py",
            "content": "plugin = True\n",
            "mode": "create",
        },
    )

    assert result.ok is False
    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / "verifier_plugin.py").exists()


def test_dynamic_conftest_pytest_plugins_declaration_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = []\npytest_plugins += ["verifier_plugin"]\n',
        encoding="utf-8",
    )
    write_config(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
""",
    )

    with pytest.raises(VerificationConfigError, match="pytest_plugins"):
        load(tmp_path)


def test_gate_rejects_command_side_creation_of_explicit_pytest_plugin(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_still_red.py").write_text(
        "def test_still_red():\n    assert False\n",
        encoding="utf-8",
    )
    _, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
protected_globs = ["tests/**"]
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q", "-p", "verifier_plugin"]
""",
    )
    baseline = verification_gate.run_baseline()
    assert baseline.passed is True
    assert baseline.commands[0].exit_code not in (None, 0)
    (tmp_path / "verifier_plugin.py").write_text(
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    session.exitstatus = 0\n",
        encoding="utf-8",
    )

    result = verification_gate.run_final(mutation_seq=1)

    assert result.commands[0].exit_code == 0
    assert result.passed is False
    assert result.protected_changes == (
        ProtectedFileChange("verifier_plugin.py", ProtectedChangeKind.CREATED),
    )
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.verified_seq is None
    assert verification_gate.grants_verified(result, mutation_seq=1) is False


def test_gate_manifests_pytest_shadow_inside_skipped_command_cwd(
    tmp_path: Path,
) -> None:
    command_dir = tmp_path / ".venv"
    command_dir.mkdir()
    protected_test = command_dir / "test_still_red.py"
    protected_test.write_text(
        "def test_still_red():\n    assert False\n",
        encoding="utf-8",
    )
    _, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
protected_globs = [".venv/test_still_red.py"]
[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
cwd = ".venv"
""",
    )
    baseline = verification_gate.run_baseline()
    assert baseline.passed is True
    assert baseline.commands[0].exit_code not in (None, 0)
    (command_dir / "pytest.py").write_text(
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )

    result = verification_gate.run_final(mutation_seq=1)

    assert result.commands[0].exit_code == 0
    assert result.passed is False
    assert result.protected_changes == (
        ProtectedFileChange(".venv/pytest.py", ProtectedChangeKind.CREATED),
    )
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.verified_seq is None
    assert verification_gate.grants_verified(result, mutation_seq=1) is False


@pytest.mark.parametrize(
    "text",
    [
        "[verification\n",
        "[verification]\nbaseline_policy = \"sometimes\"\n",
        "[verification]\nmax_repair_rounds = true\n",
        "[verification]\nmax_same_failure = 0\n",
        "[verification]\nprotected_globs = [\"../tests/**\"]\n",
        "[unexpected]\nvalue = 1\n",
    ],
)
def test_invalid_verification_config_is_rejected(tmp_path: Path, text: str) -> None:
    write_config(tmp_path, text)

    with pytest.raises(VerificationConfigError) as captured:
        load(tmp_path)

    assert captured.value.kind is ErrorKind.INVALID_VERIFICATION_CONFIG


@pytest.mark.parametrize(
    "command",
    [
        "argv = []",
        "argv = \"python -m pytest\"",
        "argv = [\"python\", 3]",
        "argv = [\"pip\", \"install\", \"thing\"]",
        "argv = [\"python\", \"-m\", \"pytest\"]\ncwd = \"missing\"",
        "argv = [\"python\", \"-m\", \"pytest\"]\ntimeout_seconds = 0",
        "argv = [\"python\", \"-m\", \"pytest\"]\ntimeout_seconds = 121",
        "argv = [\"python\", \"-m\", \"pytest\"]\ntimeout_seconds = true",
    ],
)
def test_invalid_verification_commands_are_rejected(
    tmp_path: Path, command: str
) -> None:
    write_config(tmp_path, f"[verification]\n[[verification.commands]]\n{command}\n")

    with pytest.raises(VerificationConfigError):
        load(tmp_path)


def test_provider_secret_is_rejected_before_command_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "provider-secret-for-test"
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    write_config(
        tmp_path,
        f'''[verification]
[[verification.commands]]
argv = ["python", "verify.py", "{secret}"]
''',
    )
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())

    def forbidden_policy(*args: object, **kwargs: object) -> None:
        raise AssertionError("command policy must not receive a provider secret")

    monkeypatch.setattr(runner.policy, "validate", forbidden_policy)

    with pytest.raises(
        VerificationConfigError, match="contains a host credential"
    ) as captured:
        load_verification_spec(guard, runner, known_secrets=(secret,))

    assert secret not in str(captured.value)


def test_provider_secret_is_rejected_from_config_path_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "provider-secret-for-test"
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())

    def forbidden_resolution(*args: object, **kwargs: object) -> None:
        raise AssertionError("config path reached WorkspaceGuard resolution")

    monkeypatch.setattr(guard, "resolve_for_read", forbidden_resolution)

    with pytest.raises(
        VerificationConfigError, match="config path contains a host credential"
    ) as captured:
        load_verification_spec(
            guard,
            runner,
            f"checks-{secret}.toml",
            known_secrets=(secret,),
        )

    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "config_text",
    [
        '[verification]\nprotected_globs = ["tests/provider-secret-for-test/**"]\n',
        "# provider-secret-for-test\n[verification]\nbaseline_policy = \"skip\"\n",
        (
            '[verification]\nprotected_globs = '
            '["tests/provider-\\u0073ecret-for-test/**"]\n'
        ),
        (
            '[verification]\n"field-provider-\\u0073ecret-for-test" = '
            '"value"\n'
        ),
    ],
)
def test_provider_secret_anywhere_in_config_is_rejected_before_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_text: str,
) -> None:
    secret = "provider-secret-for-test"
    write_config(tmp_path, config_text)
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())

    def forbidden_policy(*args: object, **kwargs: object) -> None:
        raise AssertionError("credential-bearing config reached command policy")

    monkeypatch.setattr(runner.policy, "validate", forbidden_policy)

    with pytest.raises(
        VerificationConfigError, match="config contains a host credential"
    ) as captured:
        load_verification_spec(guard, runner, known_secrets=(secret,))

    assert secret not in str(captured.value)


def test_config_path_must_be_workspace_relative(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())

    with pytest.raises(VerificationConfigError):
        load_verification_spec(guard, runner, tmp_path / "outside.toml")


def test_sensitive_file_cannot_be_used_as_verification_config(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("FICTIONAL_SECRET=do-not-read\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())

    with pytest.raises(VerificationConfigError) as captured:
        load_verification_spec(guard, runner, ".env")

    assert "do-not-read" not in str(captured.value)


def test_baseline_skip_does_not_start_commands(tmp_path: Path) -> None:
    (tmp_path / "must_not_run.py").write_text(
        "from pathlib import Path\nPath('marker').write_text('ran')\n",
        encoding="utf-8",
    )
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "must_not_run.py"]
""",
    )
    model = ScriptedModel([final_response()])

    result = AgentLoop(
        model,
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.baseline_verification is not None
    assert result.baseline_verification.skipped is True
    assert result.baseline_verification.commands == ()
    assert not (tmp_path / "marker").exists()


@pytest.mark.parametrize("exit_code", [0, 3])
def test_baseline_record_only_records_exit_and_continues(
    tmp_path: Path, exit_code: int
) -> None:
    (tmp_path / "baseline.py").write_text(
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "record_only"
[[verification.commands]]
argv = ["python", "baseline.py"]
""",
    )
    model = ScriptedModel([final_response()])

    result = AgentLoop(
        model,
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.baseline_verification is not None
    assert result.baseline_verification.passed is True
    assert result.baseline_verification.commands[0].exit_code == exit_code
    assert result.mutation_seq == 0
    assert model.call_count == 1


def test_baseline_must_fail_accepts_real_red_evidence(tmp_path: Path) -> None:
    (tmp_path / "red.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
[[verification.commands]]
argv = ["python", "red.py"]
""",
    )
    model = ScriptedModel([final_response()])

    result = AgentLoop(
        model,
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.baseline_verification is not None
    assert result.baseline_verification.passed is True
    assert result.baseline_verification.commands[0].started is True
    assert result.baseline_verification.commands[0].exit_code == 7
    assert model.call_count == 1


def test_baseline_must_fail_rejects_unexpected_green_before_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "green.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
[[verification.commands]]
argv = ["python", "green.py"]
""",
    )
    model = ScriptedModel([])

    result = AgentLoop(
        model,
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.BASELINE_UNEXPECTED_PASS
    assert result.step_count == 0
    assert model.call_count == 0


def test_baseline_timeout_is_infrastructure_failure_not_red(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text(
        "import time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
[[verification.commands]]
argv = ["python", "slow.py"]
timeout_seconds = 1
""",
    )
    model = ScriptedModel([])

    result = AgentLoop(
        model,
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.BASELINE_INFRASTRUCTURE_ERROR
    assert result.baseline_verification is not None
    command = result.baseline_verification.commands[0]
    assert command.started is True
    assert command.timed_out is True
    assert command.error_kind is ErrorKind.VERIFICATION_TIMEOUT
    assert model.call_count == 0


def test_baseline_start_error_is_infrastructure_failure(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "check.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "record_only"
[[verification.commands]]
argv = ["python", "check.py"]
""",
    )

    def fail_to_start(*args, **kwargs):
        raise OSError("fictional start failure")

    monkeypatch.setattr("veriloop.process.subprocess.Popen", fail_to_start)
    model = ScriptedModel([])

    result = AgentLoop(
        model,
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.BASELINE_INFRASTRUCTURE_ERROR
    assert result.baseline_verification is not None
    command = result.baseline_verification.commands[0]
    assert command.started is False
    assert command.error_kind is ErrorKind.VERIFICATION_START_ERROR
    assert model.call_count == 0


def test_baseline_runs_multiple_commands_in_frozen_order(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "must_fail"
[[verification.commands]]
argv = ["python", "first.py"]
[[verification.commands]]
argv = ["python", "second.py"]
""",
    )

    result = AgentLoop(
        ScriptedModel([final_response()]),
        ToolRegistry(),
        verification_gate=verification_gate,
    ).run("task")

    assert result.baseline_verification is not None
    assert [command.argv[-1] for command in result.baseline_verification.commands] == [
        "first.py",
        "second.py",
    ]
    assert [command.exit_code for command in result.baseline_verification.commands] == [
        2,
        0,
    ]


def test_protected_manifest_detects_modified_deleted_and_created_files(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    changed = tests_dir / "changed.py"
    deleted = tests_dir / "deleted.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    deleted.write_text("present = True\n", encoding="utf-8")
    write_config(
        tmp_path,
        "[verification]\nprotected_globs = [\"tests/**\"]\n",
    )
    guard, _, spec = loaded_components(tmp_path)
    initial = build_protected_manifest(guard, spec)

    changed.write_text("value = 2\n", encoding="utf-8")
    deleted.unlink()
    (tests_dir / "created.py").write_text("new = True\n", encoding="utf-8")
    current = build_protected_manifest(guard, spec)

    assert [
        (change.relative_path, change.kind)
        for change in compare_protected_manifests(initial, current)
    ] == [
        ("tests/changed.py", ProtectedChangeKind.MODIFIED),
        ("tests/created.py", ProtectedChangeKind.CREATED),
        ("tests/deleted.py", ProtectedChangeKind.DELETED),
    ]


def test_protected_manifest_supports_multiple_globs_and_ignores_caches(
    tmp_path: Path,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests" / "test_value.py").write_text("test = 1\n", encoding="utf-8")
    (tmp_path / "docs" / "contract.md").write_text("contract\n", encoding="utf-8")
    write_config(
        tmp_path,
        "[verification]\nprotected_globs = [\"**\", \"tests/**\", \"docs/*.md\"]\n",
    )
    guard, _, spec = loaded_components(tmp_path)
    initial = build_protected_manifest(guard, spec)

    cache = tmp_path / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "test_value.pyc").write_bytes(b"generated")
    run_dir = tmp_path / ".veriloop" / "runs" / "one"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    current = build_protected_manifest(guard, spec)

    paths = [record.relative_path for record in initial]
    assert paths == [".veriloop.toml", "docs/contract.md", "tests/test_value.py"]
    assert compare_protected_manifests(initial, current) == ()


def test_config_is_manifested_even_when_it_does_not_match_a_glob(
    tmp_path: Path,
) -> None:
    write_config(tmp_path, "[verification]\nprotected_globs = []\n")
    guard, runner, spec = loaded_components(tmp_path)
    verification_gate = VerificationGate(spec, runner)

    write_config(tmp_path, "[verification]\nmax_repair_rounds = 9\n")

    changes = verification_gate.protected_changes()
    assert len(verification_gate.protected_manifest) == 1
    assert [(item.relative_path, item.kind) for item in changes] == [
        (".veriloop.toml", ProtectedChangeKind.MODIFIED)
    ]
    assert guard.root == tmp_path.resolve()


def test_missing_config_creation_is_detected_from_frozen_spec(tmp_path: Path) -> None:
    guard, _, spec = loaded_components(tmp_path)
    initial = build_protected_manifest(guard, spec)

    write_config(tmp_path, "[verification]\n")
    current = build_protected_manifest(guard, spec)

    assert initial[0].relative_path == ".veriloop.toml"
    assert initial[0].existed is False
    assert compare_protected_manifests(initial, current) == (
        ProtectedFileChange(".veriloop.toml", ProtectedChangeKind.CREATED),
    )


def test_protected_file_type_replacement_is_detected(tmp_path: Path) -> None:
    config = write_config(tmp_path, "[verification]\n")
    guard, _, spec = loaded_components(tmp_path)
    initial = build_protected_manifest(guard, spec)

    config.unlink()
    config.mkdir()
    current = build_protected_manifest(guard, spec)

    assert compare_protected_manifests(initial, current) == (
        ProtectedFileChange(".veriloop.toml", ProtectedChangeKind.REPLACED),
    )


def test_manifest_change_evidence_contains_paths_not_protected_contents(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected = tests_dir / "secret_case.py"
    protected.write_text("fictional-sensitive-content\n", encoding="utf-8")
    write_config(
        tmp_path,
        "[verification]\nprotected_globs = [\"tests/**\"]\n",
    )
    guard, _, spec = loaded_components(tmp_path)
    initial = build_protected_manifest(guard, spec)

    protected.write_text("different-fictional-content\n", encoding="utf-8")
    changes = compare_protected_manifests(
        initial,
        build_protected_manifest(guard, spec),
    )

    assert changes[0].relative_path == "tests/secret_case.py"
    assert "fictional-sensitive-content" not in repr(changes)


def test_file_tools_deny_edit_create_and_overwrite_for_protected_globs(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    target = tests_dir / "test_value.py"
    original = b"value = 1\n"
    target.write_bytes(original)
    write_config(
        tmp_path,
        "[verification]\nprotected_globs = [\"tests/**\"]\n",
    )
    base_guard, _, spec = loaded_components(tmp_path)
    guarded = protected_guard_for_spec(base_guard, spec)
    registry = ToolRegistry()
    register_filesystem_tools(registry, guarded)

    edit = execute_file_tool(
        registry,
        "edit_file",
        {
            "path": "tests/test_value.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    overwrite = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "tests/test_value.py",
            "content": "value = 2\n",
            "mode": "overwrite",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    create = execute_file_tool(
        registry,
        "write_file",
        {
            "path": "tests/new_test.py",
            "content": "new = True\n",
            "mode": "create",
        },
    )

    assert [edit.error_kind, overwrite.error_kind, create.error_kind] == [
        ErrorKind.PATH_WRITE_DENIED,
        ErrorKind.PATH_WRITE_DENIED,
        ErrorKind.PATH_WRITE_DENIED,
    ]
    assert target.read_bytes() == original
    assert not (tests_dir / "new_test.py").exists()
    assert "fictional-sensitive-content" not in json.dumps(
        [edit.content, overwrite.content, create.content]
    )


def test_missing_config_path_is_also_denied_to_file_tools(tmp_path: Path) -> None:
    base_guard, _, spec = loaded_components(tmp_path)
    registry = ToolRegistry()
    register_filesystem_tools(registry, protected_guard_for_spec(base_guard, spec))

    result = execute_file_tool(
        registry,
        "write_file",
        {
            "path": ".veriloop.toml",
            "content": "[verification]\n",
            "mode": "create",
        },
    )

    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (tmp_path / ".veriloop.toml").exists()


def test_existing_config_is_denied_to_edit_and_overwrite_tools(tmp_path: Path) -> None:
    original = b"[verification]\nbaseline_policy = \"skip\"\n"
    (tmp_path / ".veriloop.toml").write_bytes(original)
    base_guard, _, spec = loaded_components(tmp_path)
    registry = ToolRegistry()
    register_filesystem_tools(registry, protected_guard_for_spec(base_guard, spec))
    digest = hashlib.sha256(original).hexdigest()

    edit = execute_file_tool(
        registry,
        "edit_file",
        {
            "path": ".veriloop.toml",
            "old_text": "skip",
            "new_text": "record_only",
            "expected_sha256": digest,
        },
    )
    overwrite = execute_file_tool(
        registry,
        "write_file",
        {
            "path": ".veriloop.toml",
            "content": "[verification]\n",
            "mode": "overwrite",
            "expected_sha256": digest,
        },
    )

    assert edit.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert overwrite.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert (tmp_path / ".veriloop.toml").read_bytes() == original


def test_read_list_and_search_do_not_advance_mutation_sequence(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("alpha\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    calls = (
        ToolCall(id="list", name="list_files", arguments={}),
        ToolCall(id="read", name="read_file", arguments={"path": "value.txt"}),
        ToolCall(id="search", name="search_text", arguments={"query": "alpha"}),
    )

    result = AgentLoop(
        ScriptedModel([tool_response(*calls), final_response()]),
        registry,
    ).run("inspect")

    assert result.mutation_seq == 0
    assert result.verified_seq is None


def test_successful_edit_create_and_overwrite_each_advance_mutation_sequence(
    tmp_path: Path,
) -> None:
    target = tmp_path / "value.txt"
    original = b"value = 1\n"
    created = b"first\n"
    target.write_bytes(original)
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    calls = (
        ToolCall(
            id="edit",
            name="edit_file",
            arguments={
                "path": "value.txt",
                "old_text": "value = 1",
                "new_text": "value = 2",
                "expected_sha256": hashlib.sha256(original).hexdigest(),
            },
        ),
        ToolCall(
            id="create",
            name="write_file",
            arguments={
                "path": "created.txt",
                "content": created.decode(),
                "mode": "create",
            },
        ),
        ToolCall(
            id="overwrite",
            name="write_file",
            arguments={
                "path": "created.txt",
                "content": "second\n",
                "mode": "overwrite",
                "expected_sha256": hashlib.sha256(created).hexdigest(),
            },
        ),
    )

    result = AgentLoop(
        ScriptedModel([tool_response(*calls), final_response()]),
        registry,
    ).run("mutate")

    assert result.mutation_seq == 3
    assert result.verified_seq is None
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "second\n"


def test_failed_edit_and_denied_command_do_not_advance_mutation_sequence(
    tmp_path: Path,
) -> None:
    target = tmp_path / "value.txt"
    target.write_text("value = 1\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    calls = (
        ToolCall(
            id="stale",
            name="edit_file",
            arguments={
                "path": "value.txt",
                "old_text": "value = 1",
                "new_text": "value = 2",
                "expected_sha256": "0" * 64,
            },
        ),
        ToolCall(
            id="denied",
            name="run_command",
            arguments={"argv": ["pip", "install", "fictional"]},
        ),
    )

    result = AgentLoop(
        ScriptedModel([tool_response(*calls), final_response()]),
        registry,
    ).run("do not mutate")

    assert result.mutation_seq == 0
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.parametrize("exit_code", [0, 5])
def test_started_command_advances_mutation_for_zero_and_nonzero_exit(
    tmp_path: Path,
    exit_code: int,
) -> None:
    (tmp_path / "command.py").write_text(
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    call = ToolCall(
        id="command",
        name="run_command",
        arguments={"argv": ["python", "command.py"]},
    )

    result = AgentLoop(
        ScriptedModel([tool_response(call), final_response()]),
        registry,
    ).run("run")

    assert result.mutation_seq == 1
    assert result.verified_seq is None


def test_timed_out_started_command_advances_mutation_sequence(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    call = ToolCall(
        id="timeout",
        name="run_command",
        arguments={
            "argv": ["python", "slow.py"],
            "timeout_seconds": 1,
        },
    )

    result = AgentLoop(
        ScriptedModel([tool_response(call), final_response()]),
        registry,
    ).run("run")

    assert result.mutation_seq == 1


def test_cancelled_started_command_is_paired_and_advances_mutation_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_marker = tmp_path / "started.txt"
    (tmp_path / "slow.py").write_text(
        "from pathlib import Path\n"
        "import time\n"
        "Path('started.txt').write_text('started', encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    interrupt_first_process_wait_after_start(monkeypatch, started_marker)
    call = ToolCall(
        id="cancelled-command",
        name="run_command",
        arguments={"argv": ["python", "slow.py"]},
    )

    result = AgentLoop(
        ScriptedModel([tool_response(call)]),
        registry,
    ).run("run then cancel")

    assert started_marker.read_text(encoding="utf-8") == "started"
    assert result.state is AgentState.CANCELLED
    assert result.error is not None
    assert result.error.kind is ErrorKind.CANCELLED
    assert result.tool_call_count == 1
    assert result.mutation_seq == 1
    paired_results = [
        message.tool_result
        for message in result.history
        if message.tool_result is not None
    ]
    assert len(paired_results) == 1
    paired = paired_results[0]
    assert paired.call_id == call.id
    assert paired.tool_name == call.name
    assert paired.error_kind is ErrorKind.CANCELLED
    assert paired.metadata["started"] is True
    assert paired.invalidates_verification is True


def test_cancelled_gate_command_remains_agent_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_marker = tmp_path / "gate-started.txt"
    (tmp_path / "slow_gate.py").write_text(
        "from pathlib import Path\n"
        "import time\n"
        "Path('gate-started.txt').write_text('started', encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "slow_gate.py"]
""",
    )
    interrupt_first_process_wait_after_start(monkeypatch, started_marker)
    completion = ToolCall(
        id="cancelled-gate",
        name="complete_task",
        arguments={"summary": "verify"},
    )

    result = AgentLoop(
        ScriptedModel([tool_response(completion)]),
        registry,
        verification_gate=verification_gate,
    ).run("verify then cancel")

    assert started_marker.read_text(encoding="utf-8") == "started"
    assert result.state is AgentState.CANCELLED
    assert result.error is not None
    assert result.error.kind is ErrorKind.CANCELLED
    assert result.tool_call_count == 1
    assert result.mutation_seq == 0
    assert result.final_verification is None
    paired = result.history[-1].tool_result
    assert paired is not None
    assert paired.call_id == completion.id
    assert paired.error_kind is ErrorKind.CANCELLED


def test_command_start_error_does_not_advance_mutation_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "command.py").write_text("print('never')\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    call = ToolCall(
        id="start-error",
        name="run_command",
        arguments={"argv": ["python", "command.py"]},
    )

    def fail_to_start(*args, **kwargs):
        raise OSError("fictional start failure")

    monkeypatch.setattr("veriloop.process.subprocess.Popen", fail_to_start)
    result = AgentLoop(
        ScriptedModel([tool_response(call), final_response()]),
        registry,
    ).run("run")

    assert result.mutation_seq == 0


def test_model_run_pytest_success_never_sets_verified_sequence(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n",
        encoding="utf-8",
    )
    guard = WorkspaceGuard(tmp_path)
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)
    call = ToolCall(
        id="self-test",
        name="run_command",
        arguments={"argv": ["python", "-m", "pytest", "-q"]},
    )

    result = AgentLoop(
        ScriptedModel([tool_response(call), final_response("tests pass")]),
        registry,
    ).run("run tests")

    assert result.mutation_seq == 1
    assert result.verified_seq is None
    assert result.state is AgentState.COMPLETED_UNVERIFIED


def test_final_gate_grants_only_all_green_fresh_host_evidence(tmp_path: Path) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )

    result = verification_gate.run_final(mutation_seq=4)

    assert result.passed is True
    assert result.commands[0].started is True
    assert result.commands[0].exit_code == 0
    assert result.protected_unchanged is True
    assert result.verified_seq == 4
    assert verification_gate.grants_verified(result, mutation_seq=4) is True
    assert verification_gate.grants_verified(result, mutation_seq=5) is False


def test_final_gate_without_commands_cannot_grant_verified(tmp_path: Path) -> None:
    guard, runner, spec = loaded_components(tmp_path)
    verification_gate = VerificationGate(spec, runner)

    result = verification_gate.run_final(mutation_seq=0)

    assert result.passed is False
    assert result.verified_seq is None
    assert result.failure_kind is ErrorKind.VERIFICATION_FAILED
    assert verification_gate.grants_verified(result, mutation_seq=0) is False
    assert guard.root == tmp_path.resolve()


def test_final_gate_runs_all_commands_in_order_and_rejects_any_nonzero(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("raise SystemExit(6)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "first.py"]
[[verification.commands]]
argv = ["python", "second.py"]
""",
    )

    result = verification_gate.run_final(mutation_seq=2)

    assert [command.argv[-1] for command in result.commands] == [
        "first.py",
        "second.py",
    ]
    assert [command.exit_code for command in result.commands] == [0, 6]
    assert result.failure_kind is ErrorKind.VERIFICATION_FAILED
    assert result.verified_seq is None
    assert verification_gate.grants_verified(result, mutation_seq=2) is False


def test_final_gate_timeout_cannot_grant_verified(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "slow.py"]
timeout_seconds = 1
""",
    )

    result = verification_gate.run_final(mutation_seq=1)

    assert result.failure_kind is ErrorKind.VERIFICATION_TIMEOUT
    assert result.commands[0].timed_out is True
    assert result.commands[0].started is True
    assert result.verified_seq is None


def test_final_gate_start_error_cannot_grant_verified(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )

    def fail_to_start(*args, **kwargs):
        raise OSError("fictional start failure")

    monkeypatch.setattr("veriloop.process.subprocess.Popen", fail_to_start)
    result = verification_gate.run_final(mutation_seq=1)

    assert result.failure_kind is ErrorKind.VERIFICATION_START_ERROR
    assert result.commands[0].started is False
    assert result.verified_seq is None


def test_final_gate_rejects_preexisting_protected_change_even_when_green(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected.txt"
    protected.write_text("original\n", encoding="utf-8")
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
protected_globs = ["protected.txt"]
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    protected.write_text("tampered\n", encoding="utf-8")

    result = verification_gate.run_final(mutation_seq=3)

    assert result.commands[0].exit_code == 0
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.protected_changes == (
        ProtectedFileChange("protected.txt", ProtectedChangeKind.MODIFIED),
    )
    assert result.verified_seq is None


def test_final_gate_detects_protected_change_caused_by_verification_command(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected.txt"
    protected.write_text("original\n", encoding="utf-8")
    (tmp_path / "mutating_check.py").write_text(
        "from pathlib import Path\nPath('protected.txt').write_text('changed\\n')\n",
        encoding="utf-8",
    )
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
protected_globs = ["protected.txt"]
[[verification.commands]]
argv = ["python", "mutating_check.py"]
""",
    )

    result = verification_gate.run_final(mutation_seq=8)

    assert result.commands[0].exit_code == 0
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.protected_unchanged is False
    assert result.verified_seq is None
    assert result.mutation_seq == 8


def test_final_gate_uses_frozen_commands_and_rejects_changed_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    verification_gate = gate(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    write_config(tmp_path, "[verification]\nbaseline_policy = \"skip\"\n")

    result = verification_gate.run_final(mutation_seq=0)

    assert len(result.commands) == 1
    assert result.commands[0].argv[-1] == "verify.py"
    assert result.commands[0].exit_code == 0
    assert result.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.verified_seq is None


def test_complete_task_without_commands_finishes_unverified_with_paired_result(
    tmp_path: Path,
) -> None:
    base_guard, _, spec = loaded_components(tmp_path)
    guarded = protected_guard_for_spec(base_guard, spec)
    runner = CommandRunner(guarded, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guarded, runner)
    completion = ToolCall(
        id="complete-no-config",
        name="complete_task",
        arguments={"summary": "implemented", "remaining_risks": "not verified"},
    )

    result = AgentLoop(
        ScriptedModel([tool_response(completion)]),
        registry,
        verification_gate=VerificationGate(spec, runner),
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "implemented"
    assert result.verified_seq is None
    assert result.tool_call_count == 1
    tool_result = result.history[-1].tool_result
    assert tool_result is not None
    assert tool_result.call_id == "complete-no-config"
    assert tool_result.ok is True
    evidence = json.loads(tool_result.content)
    assert evidence["verified"] is False
    assert "no verification commands" in evidence["message"]


def test_complete_task_green_gate_is_the_only_path_to_verified(tmp_path: Path) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    completion = ToolCall(
        id="complete-green",
        name="complete_task",
        arguments={"summary": "fixed"},
    )
    model = ScriptedModel([tool_response(completion), final_response("must not run")])

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.VERIFIED
    assert result.final_message == "fixed"
    assert result.final_verification is not None
    assert result.final_verification.passed is True
    assert result.verified_seq == result.mutation_seq == 0
    assert model.call_count == 1
    assert result.tool_call_count == 1
    tool_result = result.history[-1].tool_result
    assert tool_result is not None
    assert tool_result.call_id == "complete-green"
    assert tool_result.ok is True
    assert json.loads(tool_result.content)["verified"] is True
    assert AgentState.VERIFYING in result.state_history
    assert result.state_history[-1] is AgentState.VERIFIED


def test_complete_task_failed_gate_terminates_truthfully(tmp_path: Path) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(4)\n", encoding="utf-8")
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 0
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    completion = ToolCall(
        id="complete-red",
        name="complete_task",
        arguments={"summary": "attempted"},
    )
    model = ScriptedModel([tool_response(completion), final_response("must not run")])

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.VERIFICATION_FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.VERIFICATION_FAILED
    assert result.verified_seq is None
    assert model.call_count == 1
    tool_result = result.history[-1].tool_result
    assert tool_result is not None
    assert tool_result.call_id == "complete-red"
    assert tool_result.ok is False
    assert tool_result.error_kind is ErrorKind.VERIFICATION_FAILED
    assert json.loads(tool_result.content)["commands"][0]["exit_code"] == 4


def test_mixed_complete_task_response_executes_no_calls_and_pairs_every_id(
    tmp_path: Path,
) -> None:
    target = tmp_path / "value.txt"
    original = b"value = 1\n"
    target.write_bytes(original)
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "-m", "compileall", "-q", "."]
""",
    )
    edit = ToolCall(
        id="mixed-edit",
        name="edit_file",
        arguments={
            "path": "value.txt",
            "old_text": "value = 1",
            "new_text": "value = 2",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    completion = ToolCall(
        id="mixed-complete",
        name="complete_task",
        arguments={"summary": "done"},
    )
    model = ScriptedModel(
        [tool_response(edit, completion), final_response("replanned")]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "replanned"
    assert result.mutation_seq == 0
    assert target.read_bytes() == original
    mixed_results = [
        message.tool_result
        for message in result.history
        if message.tool_result is not None
    ]
    assert [item.call_id for item in mixed_results] == [
        "mixed-edit",
        "mixed-complete",
    ]
    assert [item.error_kind for item in mixed_results] == [
        ErrorKind.DEFERRED_REPLAN_REQUIRED,
        ErrorKind.COMPLETION_MUST_BE_SINGLE_CALL,
    ]
    assert model.call_count == 2
    assert len(
        [message for message in model.calls[1][0] if message.role is Role.TOOL]
    ) == 2


def test_two_complete_task_calls_are_both_rejected_as_non_unique(
    tmp_path: Path,
) -> None:
    registry, verification_gate = verification_stack(
        tmp_path,
        "[verification]\nbaseline_policy = \"skip\"\n",
    )
    calls = (
        ToolCall(id="complete-one", name="complete_task", arguments={"summary": "one"}),
        ToolCall(id="complete-two", name="complete_task", arguments={"summary": "two"}),
    )
    model = ScriptedModel([tool_response(*calls), final_response("replanned")])

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    paired = [
        message.tool_result
        for message in result.history
        if message.tool_result is not None
    ]
    assert [item.call_id for item in paired] == ["complete-one", "complete-two"]
    assert all(
        item.error_kind is ErrorKind.COMPLETION_MUST_BE_SINGLE_CALL
        for item in paired
    )
    assert result.state is AgentState.COMPLETED_UNVERIFIED


def test_complete_task_forged_verified_argument_is_rejected_by_registry(
    tmp_path: Path,
) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    forged = ToolCall(
        id="forged",
        name="complete_task",
        arguments={"summary": "done", "verified": True},
    )
    model = ScriptedModel([tool_response(forged), final_response("VERIFIED")])

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.verified_seq is None
    assert result.final_verification is None
    assert model.call_count == 2
    tool_result = next(
        message.tool_result
        for message in result.history
        if message.tool_result is not None
    )
    assert tool_result.error_kind is ErrorKind.INVALID_ARGUMENTS


def test_plain_final_claiming_verified_never_runs_gate(tmp_path: Path) -> None:
    marker = tmp_path / "gate-ran"
    (tmp_path / "verify.py").write_text(
        "from pathlib import Path\nPath('gate-ran').write_text('yes')\n",
        encoding="utf-8",
    )
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )

    result = AgentLoop(
        ScriptedModel([final_response("tests passed, VERIFIED")]),
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "tests passed, VERIFIED"
    assert result.final_verification is None
    assert result.verified_seq is None
    assert not marker.exists()


def test_failed_verification_evidence_is_seen_then_repair_can_verify(
    tmp_path: Path,
) -> None:
    verify = tmp_path / "verify.py"
    original = b"raise SystemExit(1)\n"
    verify.write_bytes(original)
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 1
max_same_failure = 9
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    first_completion = ToolCall(
        id="complete-first",
        name="complete_task",
        arguments={"summary": "first attempt"},
    )
    repair = ToolCall(
        id="repair",
        name="edit_file",
        arguments={
            "path": "verify.py",
            "old_text": "raise SystemExit(1)",
            "new_text": "raise SystemExit(0)",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    second_completion = ToolCall(
        id="complete-second",
        name="complete_task",
        arguments={"summary": "repaired"},
    )
    model = ScriptedModel(
        [
            tool_response(first_completion),
            tool_response(repair),
            tool_response(second_completion),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.VERIFIED
    assert result.repair_rounds_used == 1
    assert result.mutation_seq == result.verified_seq == 1
    assert model.call_count == 3
    failure_seen = [
        message.tool_result
        for message in model.calls[1][0]
        if message.tool_result is not None
    ][-1]
    assert failure_seen.call_id == "complete-first"
    assert failure_seen.retryable is True
    failure_payload = json.loads(failure_seen.content)
    assert failure_payload["commands"][0]["exit_code"] == 1
    assert failure_payload["remaining_repair_rounds"] == 1
    lifecycle = [
        state
        for state in result.state_history
        if state
        in {
            AgentState.VERIFYING,
            AgentState.RECOVERING,
            AgentState.THINKING,
            AgentState.VERIFIED,
        }
    ]
    first_verifying = lifecycle.index(AgentState.VERIFYING)
    assert lifecycle[first_verifying : first_verifying + 3] == [
        AgentState.VERIFYING,
        AgentState.RECOVERING,
        AgentState.THINKING,
    ]
    assert lifecycle[-2:] == [AgentState.VERIFYING, AgentState.VERIFIED]


@pytest.mark.parametrize("max_repair_rounds", [0, 1, 2])
def test_repair_budget_allows_exactly_one_plus_configured_final_attempts(
    tmp_path: Path,
    max_repair_rounds: int,
) -> None:
    (tmp_path / "always_fail.py").write_text(
        """
from pathlib import Path
path = Path("attempts.txt")
count = int(path.read_text()) if path.exists() else 0
path.write_text(str(count + 1))
raise SystemExit(3)
""",
        encoding="utf-8",
    )
    registry, verification_gate = verification_stack(
        tmp_path,
        f"""
[verification]
baseline_policy = "skip"
max_repair_rounds = {max_repair_rounds}
max_same_failure = 99
[[verification.commands]]
argv = ["python", "always_fail.py"]
""",
    )
    attempts = 1 + max_repair_rounds
    responses = [
        tool_response(
            ToolCall(
                id=f"complete-{index}",
                name="complete_task",
                arguments={"summary": f"attempt {index}"},
            )
        )
        for index in range(attempts)
    ]
    model = ScriptedModel(responses)

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.VERIFICATION_FAILED
    assert result.repair_rounds_used == max_repair_rounds
    assert model.call_count == attempts
    assert int((tmp_path / "attempts.txt").read_text(encoding="utf-8")) == attempts
    assert result.mutation_seq == 0
    completion_results = [
        message.tool_result
        for message in result.history
        if message.tool_result is not None
        and message.tool_result.tool_name == "complete_task"
    ]
    assert len(completion_results) == attempts
    assert sum(item.retryable for item in completion_results) == max_repair_rounds
    assert completion_results[-1].retryable is False
    assert json.loads(completion_results[-1].content)[
        "remaining_repair_rounds"
    ] == 0


@pytest.mark.parametrize(
    (
        "max_steps",
        "max_repair_rounds",
        "expected_state",
        "expected_rounds_used",
        "expected_remaining",
        "expected_retryable",
    ),
    [
        (1, 0, AgentState.VERIFICATION_FAILED, 0, 0, [False]),
        (1, 1, AgentState.MAX_STEPS, 0, 1, [False]),
        (1, 2, AgentState.MAX_STEPS, 0, 2, [False]),
        (2, 2, AgentState.MAX_STEPS, 1, 1, [True, False]),
    ],
)
def test_model_step_budget_only_counts_repair_rounds_that_reach_the_model(
    tmp_path: Path,
    max_steps: int,
    max_repair_rounds: int,
    expected_state: AgentState,
    expected_rounds_used: int,
    expected_remaining: int,
    expected_retryable: list[bool],
) -> None:
    (tmp_path / "always_fail.py").write_text(
        """
from pathlib import Path
path = Path("attempts.txt")
count = int(path.read_text()) if path.exists() else 0
path.write_text(str(count + 1))
raise SystemExit(3)
""",
        encoding="utf-8",
    )
    registry, verification_gate = verification_stack(
        tmp_path,
        f"""
[verification]
baseline_policy = "skip"
max_repair_rounds = {max_repair_rounds}
max_same_failure = 99
[[verification.commands]]
argv = ["python", "always_fail.py"]
""",
    )
    model = ScriptedModel(
        [
            tool_response(
                ToolCall(
                    id=f"step-limited-{index}",
                    name="complete_task",
                    arguments={"summary": f"attempt {index}"},
                )
            )
            for index in range(max_steps)
        ]
    )

    result = AgentLoop(
        model,
        registry,
        max_steps=max_steps,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is expected_state
    assert result.error is not None
    assert result.error.kind is (
        ErrorKind.MAX_STEPS
        if expected_state is AgentState.MAX_STEPS
        else ErrorKind.VERIFICATION_FAILED
    )
    assert result.repair_rounds_used == expected_rounds_used
    assert model.call_count == max_steps
    assert int((tmp_path / "attempts.txt").read_text(encoding="utf-8")) == max_steps
    completion_results = [
        message.tool_result
        for message in result.history
        if message.tool_result is not None
        and message.tool_result.tool_name == "complete_task"
    ]
    assert [item.retryable for item in completion_results] == expected_retryable
    final_result = completion_results[-1]
    assert final_result.error_kind is result.error.kind
    final_payload = json.loads(final_result.content)
    assert final_payload["state"] == expected_state.value
    assert final_payload["remaining_repair_rounds"] == expected_remaining
    assert result.state_history.count(AgentState.RECOVERING) == expected_rounds_used


def test_plain_final_during_recovery_stays_unverified_and_preserves_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 2
max_same_failure = 9
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    completion = ToolCall(
        id="complete-failed",
        name="complete_task",
        arguments={"summary": "not ready"},
    )
    model = ScriptedModel(
        [tool_response(completion), final_response("giving up, VERIFIED")]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "giving up, VERIFIED"
    assert result.repair_rounds_used == 1
    assert result.final_verification is not None
    assert result.final_verification.passed is False
    assert result.verified_seq is None


def test_failure_signature_is_stable_across_duration_and_workspace_paths(
    tmp_path: Path,
) -> None:
    signatures: list[str | None] = []
    volatile_values = (
        (
            "workspace-one",
            "2026-08-31T01:02:03.123Z",
            "0.07s",
            "1234",
            "11111111-1111-4111-8111-111111111111",
            "pytest-42",
        ),
        (
            "workspace-two",
            "2027-09-30T11:12:13.987Z",
            "8.91s",
            "9876",
            "22222222-2222-4222-8222-222222222222",
            "pytest-999",
        ),
    )
    for name, timestamp, duration, pid, run_id, temp_name in volatile_values:
        workspace = tmp_path / name
        workspace.mkdir()
        (workspace / "verify.py").write_text(
            (
                "from pathlib import Path\n"
                f"print('timestamp={timestamp}')\n"
                f"print('duration={duration}')\n"
                f"print('pid={pid} run_id={run_id} seq=1')\n"
                f"print(Path.cwd() / 'pytest-of-user' / '{temp_name}' / 'failure.txt')\n"
                "print('same deterministic assertion failure')\n"
                "raise SystemExit(3)\n"
            ),
            encoding="utf-8",
        )
        _, verification_gate = verification_stack(
            workspace,
            f"""
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py", "{(workspace / temp_name).as_posix()}"]
""",
        )
        signatures.append(
            verification_gate.run_final(mutation_seq=0).failure_signature
        )

    assert signatures[0] is not None
    assert signatures[0] == signatures[1]


def test_materially_different_failures_have_different_signatures(
    tmp_path: Path,
) -> None:
    verify = tmp_path / "verify.py"
    verify.write_text("print('alpha')\nraise SystemExit(1)\n", encoding="utf-8")
    _, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )

    first = verification_gate.run_final(mutation_seq=0)
    verify.write_text("print('time=fast')\nraise SystemExit(1)\n", encoding="utf-8")
    second = verification_gate.run_final(mutation_seq=1)
    verify.write_text("print('time=slow')\nraise SystemExit(1)\n", encoding="utf-8")
    third = verification_gate.run_final(mutation_seq=2)
    verify.write_text("print('time=slow')\nraise SystemExit(2)\n", encoding="utf-8")
    fourth = verification_gate.run_final(mutation_seq=3)

    assert first.failure_signature is not None
    assert second.failure_signature is not None
    assert third.failure_signature is not None
    assert fourth.failure_signature is not None
    assert first.failure_signature != second.failure_signature
    assert second.failure_signature != third.failure_signature
    assert third.failure_signature != fourth.failure_signature


def test_failure_signature_keeps_distinct_heads_with_shared_long_tail(
    tmp_path: Path,
) -> None:
    verify = tmp_path / "verify.py"
    stable_tail = "shared footer\n" + ("z" * 3000)
    verify.write_text(
        (
            "print('AssertionError: expected 1')\n"
            f"print({stable_tail!r})\n"
            "raise SystemExit(1)\n"
        ),
        encoding="utf-8",
    )
    _, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )

    first = verification_gate.run_final(mutation_seq=0)
    verify.write_text(
        (
            "print('AssertionError: expected 2')\n"
            f"print({stable_tail!r})\n"
            "raise SystemExit(1)\n"
        ),
        encoding="utf-8",
    )
    second = verification_gate.run_final(mutation_seq=1)

    assert first.commands[0].stdout_truncated is False
    assert second.commands[0].stdout_truncated is False
    assert first.failure_signature is not None
    assert second.failure_signature is not None
    assert first.failure_signature != second.failure_signature


def test_missing_failure_signature_does_not_trigger_stalled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 1
max_same_failure = 1
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    run_final = verification_gate.run_final

    def run_without_signature(*, mutation_seq: int) -> VerificationResult:
        return replace(
            run_final(mutation_seq=mutation_seq),
            failure_signature=None,
        )

    monkeypatch.setattr(verification_gate, "run_final", run_without_signature)
    model = ScriptedModel(
        [
            tool_response(
                ToolCall(
                    id="unsigned-first",
                    name="complete_task",
                    arguments={"summary": "first"},
                )
            ),
            tool_response(
                ToolCall(
                    id="unsigned-second",
                    name="complete_task",
                    arguments={"summary": "second"},
                )
            ),
            final_response("must not be requested"),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.VERIFICATION_FAILED
    assert result.repair_rounds_used == 1
    assert model.call_count == 2


def test_repeated_failure_signature_stops_as_stalled_before_budget_exhaustion(
    tmp_path: Path,
) -> None:
    (tmp_path / "always_fail.py").write_text(
        """
from pathlib import Path
path = Path("attempts.txt")
count = int(path.read_text()) if path.exists() else 0
path.write_text(str(count + 1))
raise SystemExit(5)
""",
        encoding="utf-8",
    )
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 5
max_same_failure = 2
[[verification.commands]]
argv = ["python", "always_fail.py"]
""",
    )
    completions = [
        tool_response(
            ToolCall(
                id=f"stalled-{index}",
                name="complete_task",
                arguments={"summary": f"attempt {index}"},
            )
        )
        for index in range(3)
    ]
    model = ScriptedModel(completions)

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.STALLED
    assert result.error is not None
    assert result.error.kind is ErrorKind.STALLED
    assert result.repair_rounds_used == 1
    assert model.call_count == 2
    assert (tmp_path / "attempts.txt").read_text(encoding="utf-8") == "2"
    last_result = result.history[-1].tool_result
    assert last_result is not None
    assert last_result.call_id == "stalled-1"
    assert last_result.error_kind is ErrorKind.STALLED
    assert last_result.retryable is False
    payload = json.loads(last_result.content)
    assert payload["same_failure_count"] == 2
    assert payload["state"] == AgentState.STALLED.value
    assert payload["failure_kind"] == ErrorKind.VERIFICATION_FAILED.value
    assert payload["termination_kind"] == ErrorKind.STALLED.value
    assert result.final_verification is not None
    assert payload["failure_signature"] == result.final_verification.failure_signature
    assert result.state_history[-1] is AgentState.STALLED


def test_max_same_failure_one_stalls_on_first_failure(tmp_path: Path) -> None:
    (tmp_path / "verify.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 0
max_same_failure = 1
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    completion = ToolCall(
        id="stall-now",
        name="complete_task",
        arguments={"summary": "attempt"},
    )
    model = ScriptedModel([tool_response(completion), final_response("extra")])

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.STALLED
    assert result.repair_rounds_used == 0
    assert model.call_count == 1


def test_failure_counter_restarts_before_a_new_signature_can_stall(
    tmp_path: Path,
) -> None:
    verify = tmp_path / "verify.py"
    original = b"print('failure-a')\nraise SystemExit(1)\n"
    verify.write_bytes(original)
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 5
max_same_failure = 2
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    first = ToolCall(
        id="restart-first",
        name="complete_task",
        arguments={"summary": "failure A"},
    )
    edit = ToolCall(
        id="restart-edit",
        name="edit_file",
        arguments={
            "path": "verify.py",
            "old_text": "print('failure-a')\nraise SystemExit(1)",
            "new_text": "print('failure-b')\nraise SystemExit(2)",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    second = ToolCall(
        id="restart-second",
        name="complete_task",
        arguments={"summary": "first failure B"},
    )
    third = ToolCall(
        id="restart-third",
        name="complete_task",
        arguments={"summary": "second failure B"},
    )
    model = ScriptedModel(
        [
            tool_response(first),
            tool_response(edit),
            tool_response(second),
            tool_response(third),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.STALLED
    assert result.repair_rounds_used == 2
    assert model.call_count == 4
    assert result.final_verification is not None
    last_result = result.history[-1].tool_result
    assert last_result is not None
    assert last_result.call_id == "restart-third"
    assert json.loads(last_result.content)["same_failure_count"] == 2


def test_different_consecutive_failure_resets_stall_counter(tmp_path: Path) -> None:
    stable_tail = "shared footer\n" + ("z" * 3000)
    (tmp_path / "verify.py").write_text(
        "from pathlib import Path\n"
        "diagnostic = Path('diagnostic.txt').read_text(encoding='utf-8').strip()\n"
        "print('AssertionError: expected ' + diagnostic)\n"
        f"print({stable_tail!r})\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    diagnostic = tmp_path / "diagnostic.txt"
    original = b"one\n"
    diagnostic.write_bytes(original)
    registry, verification_gate = verification_stack(
        tmp_path,
        """
[verification]
baseline_policy = "skip"
max_repair_rounds = 2
max_same_failure = 2
[[verification.commands]]
argv = ["python", "verify.py"]
""",
    )
    first = ToolCall(
        id="different-first",
        name="complete_task",
        arguments={"summary": "first"},
    )
    edit = ToolCall(
        id="different-edit",
        name="edit_file",
        arguments={
            "path": "diagnostic.txt",
            "old_text": "one",
            "new_text": "two",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    second = ToolCall(
        id="different-second",
        name="complete_task",
        arguments={"summary": "second"},
    )
    model = ScriptedModel(
        [
            tool_response(first),
            tool_response(edit),
            tool_response(second),
            final_response("repair opportunity preserved"),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=verification_gate,
    ).run("task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.error is None
    assert result.final_message == "repair opportunity preserved"
    assert result.repair_rounds_used == 2
    assert result.mutation_seq == 1
    assert model.call_count == 4
    assert result.final_verification is not None
    completion_results = {
        message.tool_result.call_id: message.tool_result
        for message in result.history
        if message.tool_result is not None
        and message.tool_result.tool_name == "complete_task"
    }
    first_result = completion_results[first.id]
    second_result = completion_results[second.id]
    first_payload = json.loads(first_result.content)
    second_payload = json.loads(second_result.content)
    assert first_payload["failure_signature"] != second_payload["failure_signature"]
    assert second_result.retryable is True
    assert second_payload["same_failure_count"] == 1
