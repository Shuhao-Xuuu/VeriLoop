from __future__ import annotations

from dataclasses import FrozenInstanceError

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
