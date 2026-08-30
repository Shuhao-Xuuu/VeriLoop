from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import pytest

from tests.scripted_model import ScriptedModel
from veriloop.agent import AgentLoop
from veriloop.cli import main
from veriloop.protocol import (
    AgentState,
    ErrorKind,
    FinishReason,
    Message,
    ModelResponse,
    Role,
    ToolCall,
)
from veriloop.tools import build_workspace_tools


def response(text: str = "", *calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tuple(calls),
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tool_messages(messages: list[Message] | tuple[Message, ...]) -> list[Message]:
    return [message for message in messages if message.role is Role.TOOL]


class HookedScriptedModel(ScriptedModel):
    def __init__(
        self,
        script,
        hooks: dict[int, Callable[[], None]],
    ) -> None:
        super().__init__(script)
        self._hooks = hooks

    def complete(self, messages, tools):
        hook = self._hooks.get(self.call_count)
        if hook is not None:
            hook()
        return super().complete(messages, tools)


def test_real_bug_fix_trajectory_uses_all_production_boundaries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calculator = workspace / "calculator.py"
    original = b"def clamp(value, upper):\n    return min(value, upper - 1)\n"
    calculator.write_bytes(original)
    (workspace / "check_behavior.py").write_text(
        """import sys
from calculator import clamp
if clamp(10, 10) != 10:
    print('boundary behavior is wrong', file=sys.stderr)
    raise SystemExit(1)
print('behavior is correct')
""",
        encoding="utf-8",
    )

    calls = [
        ToolCall(id="list-1", name="list_files", arguments={}),
        ToolCall(id="read-1", name="read_file", arguments={"path": "calculator.py"}),
        ToolCall(
            id="check-before",
            name="run_command",
            arguments={"argv": [sys.executable, "check_behavior.py"]},
        ),
        ToolCall(
            id="edit-1",
            name="edit_file",
            arguments={
                "path": "calculator.py",
                "old_text": "return min(value, upper - 1)",
                "new_text": "return min(value, upper)",
                "expected_sha256": sha256(original),
            },
        ),
        ToolCall(
            id="check-after",
            name="run_command",
            arguments={"argv": [sys.executable, "check_behavior.py"]},
        ),
    ]
    model = ScriptedModel(
        [
            response("", calls[0]),
            response("", calls[1]),
            response("", calls[2]),
            response("", calls[3]),
            response("", calls[4]),
            response("fixed the boundary behavior"),
        ]
    )
    registry = build_workspace_tools(workspace)

    result = AgentLoop(model, registry, max_steps=6).run("fix the clamp boundary bug")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "fixed the boundary behavior"
    assert result.step_count == 6
    assert result.tool_call_count == 5
    assert model.call_count == 6
    assert calculator.read_text(encoding="utf-8") == (
        "def clamp(value, upper):\n    return min(value, upper)\n"
    )

    schema_names = [item["function"]["name"] for item in model.calls[0][1]]
    assert schema_names == [
        "list_files",
        "read_file",
        "search_text",
        "edit_file",
        "write_file",
        "run_command",
        "complete_task",
    ]
    results = [message.tool_result for message in tool_messages(result.history)]
    assert len(results) == len(calls)
    assert [item.call_id for item in results] == [call.id for call in calls]
    read_payload = json.loads(results[1].content)
    assert read_payload["sha256"] == sha256(original)
    assert calls[3].arguments["expected_sha256"] == read_payload["sha256"]
    assert results[2].error_kind is ErrorKind.COMMAND_NONZERO_EXIT
    assert results[2].metadata["exit_code"] == 1
    assert "boundary behavior is wrong" in results[2].metadata["stderr"]
    assert results[4].ok is True
    assert json.loads(results[4].content)["exit_code"] == 0
    assert "behavior is correct" in json.loads(results[4].content)["stdout"]

    next_turn_after_failure = tool_messages(model.calls[3][0])[-1].tool_result
    assert next_turn_after_failure.call_id == "check-before"
    assert next_turn_after_failure.error_kind is ErrorKind.COMMAND_NONZERO_EXIT
    assert "boundary behavior is wrong" in next_turn_after_failure.content
    assert [message.role for message in result.history] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert result.state is not AgentState.VERIFIED


def test_stale_sha_failure_enters_history_and_model_recovers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.py"
    original = b"value = 1\n"
    external = b"value = 2\n"
    target.write_bytes(original)
    observed_after_stale: list[bytes] = []

    read_old = ToolCall(
        id="read-old",
        name="read_file",
        arguments={"path": "value.py"},
    )
    stale_edit = ToolCall(
        id="edit-stale",
        name="edit_file",
        arguments={
            "path": "value.py",
            "old_text": "value = 1",
            "new_text": "value = 3",
            "expected_sha256": sha256(original),
        },
    )
    read_new = ToolCall(
        id="read-new",
        name="read_file",
        arguments={"path": "value.py"},
    )
    corrected_edit = ToolCall(
        id="edit-corrected",
        name="edit_file",
        arguments={
            "path": "value.py",
            "old_text": "value = 2",
            "new_text": "value = 3",
            "expected_sha256": sha256(external),
        },
    )
    model = HookedScriptedModel(
        [
            response("", read_old),
            response("", stale_edit),
            response("", read_new),
            response("", corrected_edit),
            response("re-read and corrected the file"),
        ],
        hooks={
            1: lambda: target.write_bytes(external),
            2: lambda: observed_after_stale.append(target.read_bytes()),
        },
    )
    registry = build_workspace_tools(workspace)

    result = AgentLoop(model, registry, max_steps=5).run("update the value")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.tool_call_count == 4
    assert observed_after_stale == [external]
    assert target.read_bytes() == b"value = 3\n"
    results = [message.tool_result for message in tool_messages(result.history)]
    assert [item.call_id for item in results] == [
        "read-old",
        "edit-stale",
        "read-new",
        "edit-corrected",
    ]
    assert results[1].ok is False
    assert results[1].error_kind is ErrorKind.STALE_FILE
    assert json.loads(results[2].content)["sha256"] == sha256(external)
    assert corrected_edit.arguments["expected_sha256"] == json.loads(
        results[2].content
    )["sha256"]
    assert results[3].ok is True

    stale_seen_next_turn = tool_messages(model.calls[2][0])[-1].tool_result
    assert stale_seen_next_turn.call_id == "edit-stale"
    assert stale_seen_next_turn.error_kind is ErrorKind.STALE_FILE
    assert "stale_file" in stale_seen_next_turn.content
    assert result.state is not AgentState.VERIFIED


def test_command_timeout_is_visible_and_does_not_break_agent_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slow.py").write_text(
        "import time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    timeout_call = ToolCall(
        id="timeout-1",
        name="run_command",
        arguments={
            "argv": [sys.executable, "slow.py"],
            "timeout_seconds": 1,
        },
    )
    model = ScriptedModel(
        [response("", timeout_call), response("handled the timeout")]
    )

    result = AgentLoop(model, build_workspace_tools(workspace), max_steps=2).run(
        "observe timeout"
    )

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.final_message == "handled the timeout"
    seen = tool_messages(model.calls[1][0])
    assert len(seen) == 1
    assert seen[0].tool_result.call_id == "timeout-1"
    assert seen[0].tool_result.error_kind is ErrorKind.COMMAND_TIMEOUT
    assert "command_timeout" in seen[0].content


def test_cli_help_does_not_require_api_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as captured:
        main(["--help"])

    assert captured.value.code == 0
    output = capsys.readouterr().out
    assert "--workspace" in output
    assert "--max-steps" in output


def test_cli_missing_api_key_is_clear_and_does_not_echo_environment(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FICTIONAL_SECRET", "must-not-appear")

    with pytest.raises(SystemExit) as captured:
        main(["task", "--model", "test-model"])

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert "OPENAI_API_KEY must be set" in error
    assert "must-not-appear" not in error
