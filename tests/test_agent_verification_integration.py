from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from tests.scripted_model import ScriptedModel
from veriloop.agent import AgentLoop
from veriloop.context import ContextPolicy
from veriloop.filesystem import WorkspaceGuard
from veriloop.process import CommandPolicy, CommandRunner, host_child_environment
from veriloop.protocol import (
    AgentState,
    ErrorKind,
    FinishReason,
    ModelResponse,
    ProtectedChangeKind,
    Role,
    ToolCall,
    VerificationPhase,
)
from veriloop.tools import ToolRegistry, register_workspace_tools
from veriloop.trace import TraceWriter
from veriloop.verification import (
    VerificationGate,
    load_verification_spec,
    protected_guard_for_spec,
)


def tool_response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=tuple(calls),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def final_response(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=(),
        finish_reason=FinishReason.STOP,
    )


def write_pytest_config(
    workspace: Path,
    *,
    baseline_policy: str,
    max_repair_rounds: int = 0,
    max_same_failure: int = 9,
) -> Path:
    config = workspace / ".veriloop.toml"
    config.write_text(
        "\n".join(
            [
                "[verification]",
                f'baseline_policy = "{baseline_policy}"',
                f"max_repair_rounds = {max_repair_rounds}",
                f"max_same_failure = {max_same_failure}",
                'protected_globs = ["tests/**"]',
                "",
                "[[verification.commands]]",
                (
                    "argv = ["
                    f"{json.dumps(sys.executable)}, \"-m\", \"pytest\", \"-q\"]"
                ),
                'cwd = "."',
                "timeout_seconds = 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def production_runtime(
    workspace: Path,
    *,
    run_id: str,
) -> tuple[ToolRegistry, VerificationGate, TraceWriter]:
    child_environment = host_child_environment(os.environ)
    base_guard = WorkspaceGuard(workspace)
    policy = CommandPolicy()
    config_runner = CommandRunner(
        base_guard,
        policy,
        child_environment=child_environment,
    )
    spec = load_verification_spec(base_guard, config_runner)
    guarded = protected_guard_for_spec(base_guard, spec)
    runner = CommandRunner(
        guarded,
        policy,
        child_environment=child_environment,
    )
    registry = ToolRegistry()
    register_workspace_tools(registry, guarded, runner)
    gate = VerificationGate(spec, runner)
    trace = TraceWriter(
        workspace,
        run_id=run_id,
        artifact_runner=runner,
    )
    return registry, gate, trace


def read_events(trace: TraceWriter) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in trace.events_path.read_text(encoding="utf-8").splitlines()
    ]


def test_red_green_project_is_verified_by_the_production_gate(
    tmp_path: Path,
) -> None:
    implementation = tmp_path / "calculator.py"
    original = b"def add(left, right):\n    return left - right\n"
    implementation.write_bytes(original)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected_test = tests_dir / "test_calculator.py"
    protected_test.write_text(
        """from calculator import add


def test_adds_positive_numbers():
    assert add(2, 3) == 5


def test_adds_negative_numbers():
    assert add(-2, -3) == -5
""",
        encoding="utf-8",
    )
    protected_before = protected_test.read_bytes()
    config = write_pytest_config(tmp_path, baseline_policy="must_fail")
    config_before = config.read_bytes()
    registry, gate, trace = production_runtime(
        tmp_path,
        run_id="e2e-red-green",
    )
    read = ToolCall(
        id="read-calculator",
        name="read_file",
        arguments={"path": "calculator.py"},
    )
    edit = ToolCall(
        id="fix-calculator",
        name="edit_file",
        arguments={
            "path": "calculator.py",
            "old_text": "return left - right",
            "new_text": "return left + right  # fixed",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    proactive_test = ToolCall(
        id="run-tests",
        name="run_command",
        arguments={
            "argv": [sys.executable, "-m", "pytest", "-q"],
            "cwd": ".",
            "timeout_seconds": 30,
        },
    )
    completion = ToolCall(
        id="complete-green",
        name="complete_task",
        arguments={"summary": "fixed calculator addition"},
    )
    model = ScriptedModel(
        [
            tool_response(read),
            tool_response(edit),
            tool_response(proactive_test),
            tool_response(completion),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        context_policy=ContextPolicy(),
        trace_writer=trace,
    ).run("Fix calculator.add and prove it with the protected tests")

    assert result.state is AgentState.VERIFIED
    assert result.error is None
    assert result.final_message == "fixed calculator addition"
    assert result.step_count == result.tool_call_count == 4
    assert result.baseline_verification is not None
    assert result.baseline_verification.phase is VerificationPhase.BASELINE
    assert result.baseline_verification.passed is True
    assert result.baseline_verification.skipped is False
    assert result.baseline_verification.verified_seq is None
    assert result.baseline_verification.protected_unchanged is True
    assert result.baseline_verification.commands[0].started is True
    assert result.baseline_verification.commands[0].timed_out is False
    assert result.baseline_verification.commands[0].exit_code not in (None, 0)
    assert result.final_verification is not None
    assert result.final_verification.phase is VerificationPhase.FINAL
    assert result.final_verification.passed is True
    assert result.final_verification.skipped is False
    assert result.final_verification.protected_unchanged is True
    assert not result.final_verification.protected_changes
    assert result.final_verification.commands[0].started is True
    assert result.final_verification.commands[0].timed_out is False
    assert result.final_verification.commands[0].exit_code == 0
    assert result.mutation_seq == result.verified_seq == 2
    assert result.changed_files == ("calculator.py",)
    assert result.repair_rounds_used == 0
    assert model.call_count == 4
    assert implementation.read_text(encoding="utf-8").endswith(
        "return left + right  # fixed\n"
    )
    assert protected_test.read_bytes() == protected_before
    assert config.read_bytes() == config_before
    tool_messages = [
        message.tool_result
        for message in result.history
        if message.role is Role.TOOL and message.tool_result is not None
    ]
    assert [message.call_id for message in tool_messages] == [
        "read-calculator",
        "fix-calculator",
        "run-tests",
        "complete-green",
    ]
    proactive_result = next(
        message.tool_result
        for message in model.calls[3][0]
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.call_id == "run-tests"
    )
    assert proactive_result.ok is True
    assert json.loads(proactive_result.content)["exit_code"] == 0
    completion_result = next(
        message.tool_result
        for message in result.history
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.call_id == "complete-green"
    )
    assert completion_result.ok is True
    assert json.loads(completion_result.content)["verified"] is True

    events = read_events(trace)
    event_types = [str(event["event_type"]) for event in events]
    event_counts = Counter(event_types)
    assert event_types[0] == "run_started"
    assert event_types.index("baseline_finished") < event_types.index(
        "model_request_started"
    )
    assert event_counts["baseline_started"] == 1
    assert event_counts["baseline_finished"] == 1
    assert event_counts["model_request_started"] == 4
    assert event_counts["model_response_received"] == 4
    assert event_counts["tool_call_received"] == 4
    assert event_counts["tool_execution_started"] == 4
    assert event_counts["tool_execution_finished"] == 4
    assert event_counts["workspace_revision_changed"] == 2
    assert event_counts["completion_requested"] == 1
    assert event_counts["verification_started"] == 1
    assert event_counts["verification_finished"] == 1
    assert event_counts["recovery_started"] == 0
    assert event_types[-1] == "run_finished"
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    artifact = json.loads(trace.result_path.read_text(encoding="utf-8"))
    assert artifact["state"] == AgentState.VERIFIED.value
    assert artifact["step_count"] == artifact["tool_call_count"] == 4
    assert artifact["error_kind"] is None
    assert artifact["baseline"]["phase"] == VerificationPhase.BASELINE.value
    assert artifact["baseline"]["skipped"] is False
    assert artifact["baseline"]["commands"][0]["exit_code"] != 0
    assert (
        artifact["final_verification"]["phase"]
        == VerificationPhase.FINAL.value
    )
    assert artifact["final_verification"]["commands"][0]["exit_code"] == 0
    assert artifact["protected_unchanged"] is True
    assert artifact["protected_changes"] == []
    assert artifact["verified_seq"] == artifact["mutation_seq"] == 2
    assert artifact["changed_files"] == ["calculator.py"]


def test_workspace_pytest_shadow_cannot_turn_red_verification_green(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected_test = tests_dir / "test_still_red.py"
    protected_test.write_text(
        "def test_still_red():\n    assert False\n",
        encoding="utf-8",
    )
    protected_before = protected_test.read_bytes()
    config = write_pytest_config(tmp_path, baseline_policy="must_fail")
    config_before = config.read_bytes()
    registry, gate, trace = production_runtime(
        tmp_path,
        run_id="e2e-pytest-shadow",
    )
    shadow_pytest = ToolCall(
        id="shadow-pytest",
        name="write_file",
        arguments={
            "path": "pytest.py",
            "content": "raise SystemExit(0)\n",
            "mode": "create",
        },
    )
    completion = ToolCall(
        id="complete-shadowed",
        name="complete_task",
        arguments={"summary": "tests pass"},
    )
    model = ScriptedModel(
        [
            tool_response(shadow_pytest),
            tool_response(completion),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        context_policy=ContextPolicy(),
        trace_writer=trace,
    ).run("Make the protected failing test pass")

    assert result.state is AgentState.VERIFICATION_FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.VERIFICATION_FAILED
    assert result.baseline_verification is not None
    assert result.baseline_verification.passed is True
    assert result.baseline_verification.commands[0].exit_code not in (None, 0)
    assert result.final_verification is not None
    assert result.final_verification.passed is False
    assert result.final_verification.commands[0].exit_code not in (None, 0)
    assert result.verified_seq is None
    assert result.mutation_seq == 0
    assert not (tmp_path / "pytest.py").exists()
    assert protected_test.read_bytes() == protected_before
    assert config.read_bytes() == config_before
    shadow_result = next(
        message.tool_result
        for message in result.history
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.call_id == "shadow-pytest"
    )
    assert shadow_result.ok is False
    assert shadow_result.error_kind is ErrorKind.PATH_WRITE_DENIED


def test_failed_verification_evidence_drives_repair_to_verified(
    tmp_path: Path,
) -> None:
    implementation = tmp_path / "parity.py"
    original = b'def parity(value):\n    return "unknown"\n'
    incomplete = b'def parity(value):\n    return "even"\n'
    final = (
        b'def parity(value):\n'
        b'    return "even" if value % 2 == 0 else "odd"  # complete\n'
    )
    implementation.write_bytes(original)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected_test = tests_dir / "test_parity.py"
    protected_test.write_text(
        """from parity import parity


def test_even_number():
    assert parity(4) == "even"


def test_odd_number():
    assert parity(3) == "odd"
""",
        encoding="utf-8",
    )
    protected_before = protected_test.read_bytes()
    config = write_pytest_config(
        tmp_path,
        baseline_policy="must_fail",
        max_repair_rounds=1,
    )
    config_before = config.read_bytes()
    registry, gate, trace = production_runtime(
        tmp_path,
        run_id="e2e-repair-green",
    )
    first_edit = ToolCall(
        id="incomplete-fix",
        name="edit_file",
        arguments={
            "path": "parity.py",
            "old_text": 'return "unknown"',
            "new_text": 'return "even"',
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    first_completion = ToolCall(
        id="complete-incomplete",
        name="complete_task",
        arguments={"summary": "handled even values"},
    )
    second_edit = ToolCall(
        id="complete-fix",
        name="edit_file",
        arguments={
            "path": "parity.py",
            "old_text": 'return "even"',
            "new_text": (
                'return "even" if value % 2 == 0 else "odd"  # complete'
            ),
            "expected_sha256": hashlib.sha256(incomplete).hexdigest(),
        },
    )
    second_completion = ToolCall(
        id="complete-repaired",
        name="complete_task",
        arguments={"summary": "implemented both parity branches"},
    )
    model = ScriptedModel(
        [
            tool_response(first_edit),
            tool_response(first_completion),
            tool_response(second_edit),
            tool_response(second_completion),
        ]
    )

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        context_policy=ContextPolicy(),
        trace_writer=trace,
    ).run("Implement parity for even and odd integers")

    assert result.state is AgentState.VERIFIED
    assert result.error is None
    assert result.step_count == result.tool_call_count == 4
    assert result.repair_rounds_used == 1
    assert result.mutation_seq == result.verified_seq == 2
    assert result.changed_files == ("parity.py",)
    assert result.baseline_verification is not None
    assert result.baseline_verification.passed is True
    assert result.baseline_verification.commands[0].exit_code not in (None, 0)
    assert result.final_verification is not None
    assert result.final_verification.passed is True
    assert result.final_verification.commands[0].exit_code == 0
    assert model.call_count == 4
    assert implementation.read_bytes() == final
    assert protected_test.read_bytes() == protected_before
    assert config.read_bytes() == config_before

    failure_seen = next(
        message.tool_result
        for message in model.calls[2][0]
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.call_id == "complete-incomplete"
    )
    assert failure_seen.ok is False
    assert failure_seen.retryable is True
    assert failure_seen.error_kind is ErrorKind.VERIFICATION_FAILED
    failure_payload = json.loads(failure_seen.content)
    assert failure_payload["commands"][0]["exit_code"] not in (None, 0)
    assert failure_payload["remaining_repair_rounds"] == 1
    assert failure_payload["state"] == AgentState.RECOVERING.value
    assert failure_payload["verified"] is False
    assert failure_payload["mutation_seq"] == 1
    assert failure_payload["verified_seq"] is None
    assert failure_payload["protected_unchanged"] is True
    completion_results = [
        message.tool_result
        for message in result.history
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.tool_name == "complete_task"
    ]
    assert [item.call_id for item in completion_results] == [
        "complete-incomplete",
        "complete-repaired",
    ]
    assert [item.ok for item in completion_results] == [False, True]
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

    events = read_events(trace)
    counts = Counter(str(event["event_type"]) for event in events)
    assert counts["verification_started"] == 2
    assert counts["verification_finished"] == 2
    assert counts["recovery_started"] == 1
    assert counts["workspace_revision_changed"] == 2
    assert counts["run_failed"] == 0
    assert events[-1]["event_type"] == "run_finished"
    artifact = json.loads(trace.result_path.read_text(encoding="utf-8"))
    assert artifact["state"] == AgentState.VERIFIED.value
    assert artifact["repair_rounds_used"] == 1
    assert artifact["verified_seq"] == artifact["mutation_seq"] == 2
    assert artifact["final_verification"]["passed"] is True


def test_persistent_failure_stops_at_the_exact_repair_budget(
    tmp_path: Path,
) -> None:
    implementation = tmp_path / "status.py"
    implementation.write_text(
        'def status():\n    return "broken"\n',
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected_test = tests_dir / "test_status.py"
    protected_test.write_text(
        """from status import status


def test_status_is_ready():
    assert status() == "ready"
""",
        encoding="utf-8",
    )
    protected_before = protected_test.read_bytes()
    config = write_pytest_config(
        tmp_path,
        baseline_policy="must_fail",
        max_repair_rounds=1,
        max_same_failure=9,
    )
    config_before = config.read_bytes()
    registry, gate, trace = production_runtime(
        tmp_path,
        run_id="e2e-persistent-failure",
    )
    completions = [
        ToolCall(
            id=f"complete-failing-{index}",
            name="complete_task",
            arguments={"summary": f"attempt {index}"},
        )
        for index in range(2)
    ]
    model = ScriptedModel([tool_response(call) for call in completions])

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        context_policy=ContextPolicy(),
        trace_writer=trace,
    ).run("Make status report ready")

    assert result.state is AgentState.VERIFICATION_FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.VERIFICATION_FAILED
    assert result.step_count == result.tool_call_count == 2
    assert result.repair_rounds_used == 1
    assert result.mutation_seq == 0
    assert result.verified_seq is None
    assert result.final_verification is not None
    assert result.final_verification.passed is False
    assert result.final_verification.commands[0].exit_code not in (None, 0)
    assert AgentState.VERIFIED not in result.state_history
    assert model.call_count == 2
    assert protected_test.read_bytes() == protected_before
    assert config.read_bytes() == config_before

    completion_results = [
        message.tool_result
        for message in result.history
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.tool_name == "complete_task"
    ]
    assert [item.call_id for item in completion_results] == [
        "complete-failing-0",
        "complete-failing-1",
    ]
    assert [item.retryable for item in completion_results] == [True, False]
    assert [item.error_kind for item in completion_results] == [
        ErrorKind.VERIFICATION_FAILED,
        ErrorKind.VERIFICATION_FAILED,
    ]
    last_failure = json.loads(completion_results[-1].content)
    assert last_failure["remaining_repair_rounds"] == 0
    assert last_failure["commands"][0]["exit_code"] not in (None, 0)
    assert last_failure["state"] == AgentState.VERIFICATION_FAILED.value

    events = read_events(trace)
    counts = Counter(str(event["event_type"]) for event in events)
    assert counts["model_request_started"] == 2
    assert counts["verification_started"] == 2
    assert counts["verification_finished"] == 2
    assert counts["recovery_started"] == 1
    assert counts["workspace_revision_changed"] == 0
    assert "verified" not in [event["state"] for event in events]
    assert counts["run_failed"] == 1
    assert events[-1]["event_type"] == "run_finished"
    artifact = json.loads(trace.result_path.read_text(encoding="utf-8"))
    assert artifact["state"] == AgentState.VERIFICATION_FAILED.value
    assert artifact["error_kind"] == ErrorKind.VERIFICATION_FAILED.value
    assert artifact["repair_rounds_used"] == 1
    assert artifact["verified_seq"] is None
    assert artifact["final_verification"]["passed"] is False


def test_started_command_tampering_with_protected_test_cannot_verify(
    tmp_path: Path,
) -> None:
    (tmp_path / "contract.py").write_text(
        'def answer():\n    return 42\n',
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected_test = tests_dir / "test_contract.py"
    protected_test.write_text(
        """from contract import answer


def test_answer_contract():
    assert answer() == 42  # protected-v1
""",
        encoding="utf-8",
    )
    protected_before = protected_test.read_bytes()
    (tmp_path / "tamper.py").write_text(
        """from pathlib import Path

path = Path("tests/test_contract.py")
content = path.read_text(encoding="utf-8")
path.write_text(content.replace("protected-v1", "protected-v2"), encoding="utf-8")
""",
        encoding="utf-8",
    )
    config = write_pytest_config(tmp_path, baseline_policy="skip")
    config_before = config.read_bytes()
    registry, gate, trace = production_runtime(
        tmp_path,
        run_id="e2e-protected-tamper",
    )
    tamper = ToolCall(
        id="tamper-protected-test",
        name="run_command",
        arguments={
            "argv": [sys.executable, "tamper.py"],
            "cwd": ".",
            "timeout_seconds": 30,
        },
    )
    completion = ToolCall(
        id="complete-after-tamper",
        name="complete_task",
        arguments={"summary": "tests are green"},
    )
    model = ScriptedModel([tool_response(tamper), tool_response(completion)])

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        context_policy=ContextPolicy(),
        trace_writer=trace,
    ).run("Keep the contract implementation correct")

    assert result.state is AgentState.VERIFICATION_FAILED
    assert result.error is not None
    assert result.error.kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.step_count == result.tool_call_count == 2
    assert result.mutation_seq == 1
    assert result.verified_seq is None
    assert result.final_verification is not None
    assert result.final_verification.passed is False
    assert result.final_verification.commands[0].started is True
    assert result.final_verification.commands[0].exit_code == 0
    assert result.final_verification.failure_kind is ErrorKind.PROTECTED_FILE_CHANGED
    assert result.final_verification.protected_unchanged is False
    assert len(result.final_verification.protected_changes) == 1
    protected_change = result.final_verification.protected_changes[0]
    assert protected_change.relative_path == "tests/test_contract.py"
    assert protected_change.kind is ProtectedChangeKind.MODIFIED
    assert protected_test.read_bytes() != protected_before
    assert b"protected-v2" in protected_test.read_bytes()
    assert config.read_bytes() == config_before
    assert model.call_count == 2

    tamper_result = next(
        message.tool_result
        for message in result.history
        if message.role is Role.TOOL
        and message.tool_result is not None
        and message.tool_result.call_id == "tamper-protected-test"
    )
    assert tamper_result.ok is True
    tamper_payload = json.loads(tamper_result.content)
    assert tamper_payload["started"] is True
    assert tamper_payload["exit_code"] == 0
    completion_result = result.history[-1].tool_result
    assert completion_result is not None
    assert completion_result.call_id == "complete-after-tamper"
    assert completion_result.error_kind is ErrorKind.PROTECTED_FILE_CHANGED
    completion_payload = json.loads(completion_result.content)
    assert completion_payload["verified"] is False
    assert completion_payload["protected_changes"] == [
        {"kind": ProtectedChangeKind.MODIFIED.value, "path": "tests/test_contract.py"}
    ]

    events = read_events(trace)
    verification_event = next(
        event for event in events if event["event_type"] == "verification_finished"
    )
    verification_payload = verification_event["payload"]
    assert verification_payload["commands"][0]["exit_code"] == 0
    assert (
        verification_payload["failure_kind"]
        == ErrorKind.PROTECTED_FILE_CHANGED.value
    )
    assert verification_payload["protected_changes"] == [
        {"kind": ProtectedChangeKind.MODIFIED.value, "path": "tests/test_contract.py"}
    ]
    assert not any(event["state"] == AgentState.VERIFIED.value for event in events)
    artifact = json.loads(trace.result_path.read_text(encoding="utf-8"))
    assert artifact["state"] == AgentState.VERIFICATION_FAILED.value
    assert artifact["verified_seq"] is None
    assert artifact["protected_unchanged"] is False
    assert artifact["protected_changes"] == [
        {"kind": ProtectedChangeKind.MODIFIED.value, "path": "tests/test_contract.py"}
    ]
    assert artifact["final_verification"]["commands"][0]["exit_code"] == 0


def test_plain_final_claim_never_runs_the_configured_gate(tmp_path: Path) -> None:
    marker = tmp_path / "gate-ran"
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    protected_test = tests_dir / "test_gate_marker.py"
    protected_test.write_text(
        """from pathlib import Path


def test_gate_marker():
    Path("gate-ran").write_text("ran", encoding="utf-8")
""",
        encoding="utf-8",
    )
    protected_before = protected_test.read_bytes()
    config = write_pytest_config(tmp_path, baseline_policy="skip")
    config_before = config.read_bytes()
    registry, gate, trace = production_runtime(
        tmp_path,
        run_id="e2e-plain-final",
    )
    model = ScriptedModel([final_response("任务完成，测试通过，VERIFIED。")])

    result = AgentLoop(
        model,
        registry,
        verification_gate=gate,
        context_policy=ContextPolicy(),
        trace_writer=trace,
    ).run("Finish the task")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "任务完成，测试通过，VERIFIED。"
    assert result.step_count == 1
    assert result.tool_call_count == 0
    assert result.mutation_seq == 0
    assert result.verified_seq is None
    assert result.final_verification is None
    assert result.baseline_verification is not None
    assert result.baseline_verification.skipped is True
    assert AgentState.VERIFYING not in result.state_history
    assert AgentState.VERIFIED not in result.state_history
    assert model.call_count == 1
    assert not marker.exists()
    assert protected_test.read_bytes() == protected_before
    assert config.read_bytes() == config_before

    events = read_events(trace)
    event_types = [str(event["event_type"]) for event in events]
    assert event_types.count("baseline_started") == 1
    assert event_types.count("baseline_finished") == 1
    assert "completion_requested" not in event_types
    assert "verification_started" not in event_types
    assert "verification_finished" not in event_types
    assert "tool_execution_started" not in event_types
    assert event_types[-1] == "run_finished"
    assert events[-1]["state"] == AgentState.COMPLETED_UNVERIFIED.value
    artifact = json.loads(trace.result_path.read_text(encoding="utf-8"))
    assert artifact["state"] == AgentState.COMPLETED_UNVERIFIED.value
    assert artifact["tool_call_count"] == 0
    assert artifact["verified_seq"] is None
    assert artifact["final_verification"] is None
    assert artifact["error_kind"] is None
