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
    FinishReason,
    ModelResponse,
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
