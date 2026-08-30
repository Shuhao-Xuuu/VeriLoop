from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from veriloop.protocol import (
    AgentResult,
    AgentState,
    ErrorKind,
    ProtectedChangeKind,
    ProtectedFileChange,
    VerificationCommandResult,
    VerificationPhase,
    VerificationResult,
)
from veriloop.filesystem import WorkspaceGuard
from veriloop.process import CommandPolicy, CommandRunner
from veriloop.verification import (
    BaselinePolicy,
    VerificationConfigError,
    load_verification_spec,
)


def write_config(workspace: Path, text: str) -> Path:
    path = workspace / ".veriloop.toml"
    path.write_text(text, encoding="utf-8")
    return path


def load(workspace: Path):
    guard = WorkspaceGuard(workspace)
    runner = CommandRunner(guard, CommandPolicy())
    return load_verification_spec(guard, runner)


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
