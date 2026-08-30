from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

import pytest

from tests.scripted_model import ScriptedModel
import veriloop.cli as cli
from veriloop.process import CommandPolicy, CommandRunner
from veriloop.protocol import (
    AgentState,
    ErrorKind,
    FinishReason,
    ModelResponse,
    ToolCall,
)
from veriloop.tools import ToolRegistry
from veriloop.trace import Redactor, TraceWriter


FAKE_API_KEY = "veriloop-cli-secret-value"


def response(text: str = "", *calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tuple(calls),
        finish_reason=(
            FinishReason.TOOL_CALLS if calls else FinishReason.STOP
        ),
    )


def use_model(
    monkeypatch: pytest.MonkeyPatch,
    model: ScriptedModel,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> ScriptedModel:
        captured.update(kwargs)
        return model

    monkeypatch.setattr(cli, "OpenAICompatibleModel", factory)
    return captured


def write_config(
    workspace: Path,
    *,
    name: str = ".veriloop.toml",
    baseline_policy: str = "skip",
    script: str | None = None,
    protected_globs: tuple[str, ...] = (),
    max_repair_rounds: int = 0,
) -> Path:
    lines = [
        "[verification]",
        f'baseline_policy = "{baseline_policy}"',
        f"max_repair_rounds = {max_repair_rounds}",
        "max_same_failure = 2",
        f"protected_globs = {json.dumps(list(protected_globs))}",
    ]
    if script is not None:
        lines.extend(
            [
                "",
                "[[verification.commands]]",
                f"argv = [{json.dumps(sys.executable)}, {json.dumps(script)}]",
                'cwd = "."',
                "timeout_seconds = 10",
            ]
        )
    path = workspace / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def one_result(workspace: Path) -> dict[str, Any]:
    paths = list((workspace / ".veriloop" / "runs").glob("*/result.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def workspace_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_cli_list_rendering_has_a_total_bound() -> None:
    values = tuple(f"path-{index}-{'x' * 2_048}" for index in range(100))

    rendered = cli._cli_list(values, Redactor())

    assert len(rendered) <= cli.CLI_TEXT_LIMIT
    assert cli.CLI_TRUNCATION_MARKER in json.loads(rendered)


def test_provider_secret_values_are_removed_from_the_frozen_child_environment() -> None:
    base_url = "https://provider.invalid/v1"
    source = {
        "PATH": "safe-path",
        "CI": f"copied-{FAKE_API_KEY}-value",
        "TERM": f"copied-{base_url}-value",
        "OPENAI_API_KEY": FAKE_API_KEY,
        "UNRELATED": "not-allowlisted",
    }

    filtered = cli._provider_safe_child_environment(
        source,
        (FAKE_API_KEY, base_url),
    )

    assert filtered == {"PATH": "safe-path"}
    assert FAKE_API_KEY not in repr(filtered)
    assert base_url not in repr(filtered)


def test_help_and_run_alias_are_key_free_and_preserve_legacy_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("help constructed a runtime component")

    monkeypatch.setattr(cli, "WorkspaceGuard", forbidden)
    monkeypatch.setattr(cli, "OpenAICompatibleModel", forbidden)
    monkeypatch.setattr(cli, "TraceWriter", forbidden)

    with pytest.raises(SystemExit) as root_help:
        cli.main(["--help"])
    assert root_help.value.code == 0
    root_output = capsys.readouterr().out
    assert "--workspace" in root_output
    assert "--max-steps" in root_output
    assert "--config" in root_output
    assert "veriloop replay" in root_output

    with pytest.raises(SystemExit) as run_help:
        cli.main(["run", "--help"])
    assert run_help.value.code == 0
    assert "--config" in capsys.readouterr().out

    with pytest.raises(SystemExit) as replay_help:
        cli.main(["replay", "--help"])
    assert replay_help.value.code == 0
    assert "events.jsonl" in capsys.readouterr().out


def test_missing_key_stops_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FICTIONAL_SECRET", "must-not-appear")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("missing-key path constructed a runtime component")

    monkeypatch.setattr(cli, "WorkspaceGuard", forbidden)

    with pytest.raises(SystemExit) as captured:
        cli.main(["task", "--model", "test-model", "--workspace", str(tmp_path)])

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert "OPENAI_API_KEY must be set" in error
    assert "must-not-appear" not in error
    assert not (tmp_path / ".veriloop").exists()


def test_invalid_config_is_redacted_and_precedes_runtime_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (tmp_path / ".veriloop.toml").write_text(
        "\n".join(
            [
                "[verification]",
                "[[verification.commands]]",
                (
                    "argv = ["
                    f"{json.dumps(sys.executable)}, \"verify.py\", "
                    f"{json.dumps(FAKE_API_KEY)}]"
                ),
                'cwd = "."',
                "timeout_seconds = 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid config reached runtime setup")

    monkeypatch.setattr(cli, "TraceWriter", forbidden)
    monkeypatch.setattr(cli, "OpenAICompatibleModel", forbidden)
    monkeypatch.setattr(cli, "protected_guard_for_spec", forbidden)
    monkeypatch.setattr(cli, "VerificationGate", forbidden)
    monkeypatch.setattr(CommandPolicy, "validate", forbidden)

    with pytest.raises(SystemExit) as captured:
        cli.main(["task", "--model", "test-model", "--workspace", str(tmp_path)])

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert ErrorKind.INVALID_VERIFICATION_CONFIG.value in error
    assert "contains a host credential" in error
    assert FAKE_API_KEY not in error
    assert not (tmp_path / ".veriloop").exists()


@pytest.mark.parametrize("target", ["WorkspaceGuard", "TraceWriter"])
def test_setup_os_errors_are_bounded_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)

    def fail_setup(*args: object, **kwargs: object) -> object:
        raise OSError(f"setup failed near {FAKE_API_KEY}")

    def fail_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("setup error constructed the model")

    monkeypatch.setattr(cli, target, fail_setup)
    if target == "WorkspaceGuard":
        monkeypatch.setattr(cli, "OpenAICompatibleModel", fail_model)
    else:
        use_model(monkeypatch, ScriptedModel([]))

    with pytest.raises(SystemExit) as captured:
        cli.main(["task", "--model", "test-model", "--workspace", str(tmp_path)])

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert "setup failed near [REDACTED]" in error
    assert FAKE_API_KEY not in error
    assert "Traceback" not in error


def test_model_initialization_failure_does_not_create_an_empty_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)

    def fail_model(*args: object, **kwargs: object) -> object:
        raise RuntimeError(f"provider setup echoed {FAKE_API_KEY}")

    monkeypatch.setattr(cli, "OpenAICompatibleModel", fail_model)

    with pytest.raises(SystemExit) as captured:
        cli.main(["task", "--model", "test-model", "--workspace", str(tmp_path)])

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert "model initialization failed: RuntimeError" in error
    assert FAKE_API_KEY not in error
    assert not (tmp_path / ".veriloop").exists()


def test_replay_cli_is_key_free_and_never_constructs_runtime_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    writer = TraceWriter(tmp_path, run_id="cli-replay")
    writer.emit("run_started", AgentState.INITIALIZING, {})
    writer.emit(
        "run_finished",
        AgentState.COMPLETED_UNVERIFIED,
        {"state": AgentState.COMPLETED_UNVERIFIED.value},
    )
    writer.close()
    before = workspace_snapshot(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls: list[str] = []

    def forbidden(name: str) -> Callable[..., Any]:
        def fail(*args: object, **kwargs: object) -> Any:
            calls.append(name)
            raise AssertionError(f"replay called {name}")

        return fail

    monkeypatch.setattr(cli, "WorkspaceGuard", forbidden("guard"))
    monkeypatch.setattr(cli, "OpenAICompatibleModel", forbidden("model"))
    monkeypatch.setattr(cli, "TraceWriter", forbidden("trace"))
    monkeypatch.setattr(ToolRegistry, "execute", forbidden("tool"))
    monkeypatch.setattr(CommandRunner, "run", forbidden("command"))

    exit_code = cli.main(["replay", str(writer.run_dir)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "run_id: cli-replay" in output
    assert "final_state=completed_unverified" in output
    assert calls == []
    assert workspace_snapshot(tmp_path) == before


def test_corrupt_replay_is_clear_nonzero_and_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text("not-json\n", encoding="utf-8")
    before = workspace_snapshot(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("corrupt replay constructed runtime components")

    monkeypatch.setattr(cli, "WorkspaceGuard", forbidden)
    monkeypatch.setattr(cli, "OpenAICompatibleModel", forbidden)
    monkeypatch.setattr(cli, "TraceWriter", forbidden)

    with pytest.raises(SystemExit) as captured:
        cli.main(["replay", str(source)])

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert "line 1 is not valid JSON" in error
    assert "Traceback" not in error
    assert workspace_snapshot(tmp_path) == before


def test_cli_wires_frozen_config_gate_trace_and_verified_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("CI", f"copied-{FAKE_API_KEY}-value")
    monkeypatch.setenv("TERM", "copied-https://provider.invalid/v1-value")
    (tmp_path / "verify.py").write_text(
        (
            "import os\n"
            "raise SystemExit("
            "0 if 'CI' not in os.environ and 'TERM' not in os.environ else 7"
            ")\n"
        ),
        encoding="utf-8",
    )
    write_config(tmp_path, name="checks.toml", script="verify.py")
    completion = ToolCall(
        id="complete-one",
        name="complete_task",
        arguments={
            "summary": (
                "verified without exposing credentials "
                "https://provider.invalid/v1"
            )
        },
    )
    model = ScriptedModel([response("", completion)])
    model_kwargs = use_model(monkeypatch, model)

    exit_code = cli.main(
        [
            "run",
            "verify the task",
            "--model",
            "test-model",
            "--workspace",
            str(tmp_path),
            "--config",
            "checks.toml",
            "--base-url",
            "https://provider.invalid/v1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "state: VERIFIED" in output
    assert "baseline: passed=true skipped=true commands=0" in output
    assert "final verification: passed=true skipped=false commands=1" in output
    assert "exit_codes=[0]" in output
    assert "protected_unchanged=true" in output
    assert "changed files: []" in output
    assert "trace path: .veriloop/runs/" in output
    assert "result path: .veriloop/runs/" in output
    assert "patch path:" in output
    assert "[REDACTED]" in output
    assert FAKE_API_KEY not in output
    assert "https://provider.invalid/v1" not in output
    assert callable(model_kwargs["retry_observer"])
    assert model_kwargs["api_key"] == FAKE_API_KEY
    assert model_kwargs["base_url"] == "https://provider.invalid/v1"
    schemas = model.calls[0][1]
    complete_schema = next(
        item for item in schemas if item["function"]["name"] == "complete_task"
    )
    assert complete_schema["function"]["parameters"]["additionalProperties"] is False
    artifact = one_result(tmp_path)
    assert artifact["state"] == AgentState.VERIFIED.value
    assert artifact["final_verification"]["passed"] is True
    for path in (tmp_path / ".veriloop" / "runs").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert FAKE_API_KEY not in text
            assert "https://provider.invalid/v1" not in text


def test_cli_binds_file_tools_to_the_config_protected_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)
    config = write_config(tmp_path, name="checks.toml")
    original = config.read_bytes()
    overwrite = ToolCall(
        id="overwrite-config",
        name="write_file",
        arguments={
            "path": "checks.toml",
            "content": "changed\n",
            "mode": "overwrite",
            "expected_sha256": None,
        },
    )
    model = ScriptedModel(
        [response("", overwrite), response("finished without gate")]
    )
    use_model(monkeypatch, model)

    exit_code = cli.main(
        [
            "task",
            "--model",
            "test-model",
            "--workspace",
            str(tmp_path),
            "--config",
            "checks.toml",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "state: COMPLETED_UNVERIFIED" in output
    assert config.read_bytes() == original
    tool_messages = [
        message for message in model.calls[1][0] if message.tool_result is not None
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_result.error_kind is ErrorKind.PATH_WRITE_DENIED


def test_provider_secret_tool_arguments_stop_before_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)
    leaked_call = ToolCall(
        id="leaked-argument",
        name="read_file",
        arguments={"path": f"notes-{FAKE_API_KEY}.txt"},
    )
    model = ScriptedModel([response("", leaked_call)])
    use_model(monkeypatch, model)

    def forbidden_execute(*args: object, **kwargs: object) -> object:
        raise AssertionError("provider credential reached ToolRegistry.execute")

    monkeypatch.setattr(ToolRegistry, "execute", forbidden_execute)

    exit_code = cli.main(
        ["task", "--model", "test-model", "--workspace", str(tmp_path)]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "state: FAILED" in output
    assert "model response contains a host credential" in output
    assert FAKE_API_KEY not in output
    assert model.call_count == 1
    assert FAKE_API_KEY not in repr(model.calls)
    artifact = one_result(tmp_path)
    assert artifact["state"] == AgentState.FAILED.value
    assert artifact["error_kind"] == ErrorKind.INVALID_ARGUMENTS.value
    for path in (tmp_path / ".veriloop" / "runs").rglob("*"):
        if path.is_file():
            assert FAKE_API_KEY not in path.read_text(encoding="utf-8")


def test_baseline_failure_emits_artifacts_without_calling_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)
    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    write_config(
        tmp_path,
        baseline_policy="must_fail",
        script="verify.py",
    )
    model = ScriptedModel([])
    use_model(monkeypatch, model)

    exit_code = cli.main(
        ["task", "--model", "test-model", "--workspace", str(tmp_path)]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert model.call_count == 0
    assert "state: FAILED" in output
    assert "baseline: passed=false skipped=false commands=1" in output
    assert "failure_kind=baseline_unexpected_pass" in output
    assert "final verification: not_run" in output
    artifact = one_result(tmp_path)
    assert artifact["state"] == AgentState.FAILED.value
    assert artifact["error_kind"] == ErrorKind.BASELINE_UNEXPECTED_PASS.value


def test_plain_final_claim_remains_unverified_and_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_API_KEY)
    model = ScriptedModel(
        [
            response(
                "task complete, tests pass, VERIFIED "
                "Authorization: Bearer second-secret \x1b[31m"
            )
        ]
    )
    use_model(monkeypatch, model)

    exit_code = cli.main(
        ["task", "--model", "test-model", "--workspace", str(tmp_path)]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "state: COMPLETED_UNVERIFIED" in output
    assert "final verification: not_run" in output
    assert "state: VERIFIED\n" not in output
    assert FAKE_API_KEY not in output
    assert "second-secret" not in output
    assert "\x1b" not in output
    assert "[REDACTED]" in output
    artifact = one_result(tmp_path)
    assert artifact["state"] == AgentState.COMPLETED_UNVERIFIED.value
    assert artifact["final_verification"] is None
    for path in (tmp_path / ".veriloop" / "runs").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert FAKE_API_KEY not in text
            assert "second-secret" not in text
