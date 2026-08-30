from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from tests.scripted_model import ScriptedModel
from veriloop.agent import AgentLoop
from veriloop.protocol import (
    AgentResult,
    AgentState,
    ErrorKind,
    FinishReason,
    ModelResponse,
    ProtectedChangeKind,
    ProtectedFileChange,
    VerificationCommandResult,
    VerificationPhase,
    VerificationResult,
    ToolCall,
)
from veriloop.filesystem import WorkspaceGuard
from veriloop.process import CommandPolicy, CommandRunner
from veriloop.tools import ToolRegistry, register_filesystem_tools
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
    assert spec.protected_globs == ("tests/**", ".veriloop.toml")
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
