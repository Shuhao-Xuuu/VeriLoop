"""Command-line composition root for the verified local harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Mapping, Sequence

from .agent import AgentLoop
from .context import ContextPolicy
from .filesystem import WorkspaceGuard
from .model import OpenAICompatibleModel
from .process import (
    CommandPolicy,
    CommandRunner,
    host_child_environment,
)
from .protocol import AgentResult, AgentState, VerificationResult
from .tools import ToolRegistry, register_workspace_tools
from .trace import Redactor, ReplayError, TraceError, TraceWriter, replay_trace
from .verification import (
    DEFAULT_CONFIG_PATH,
    ProtectedManifestError,
    VerificationConfigError,
    VerificationGate,
    load_verification_spec,
    protected_guard_for_spec,
)


CLI_TEXT_LIMIT = 2_048
CLI_LIST_LIMIT = 100
CLI_TRUNCATION_MARKER = "...[veriloop cli truncated]..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the VeriLoop verified local harness",
        epilog="Replay saved evidence with: veriloop replay <run-dir-or-events.jsonl>",
    )
    parser.add_argument("task", help="task to send to the model")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--workspace",
        default=".",
        help="local workspace root (default: current directory)",
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"workspace-relative verification config (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def build_replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veriloop replay",
        description="Read and display saved VeriLoop evidence without re-execution",
    )
    parser.add_argument("source", help="run directory or events.jsonl path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["replay"]:
        return _replay_main(arguments[1:])
    if arguments[:1] == ["run"]:
        arguments = arguments[1:]

    parser = build_parser()
    args = parser.parse_args(arguments)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY must be set in the environment")
    if not args.model:
        parser.error("--model or OPENAI_MODEL is required")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")

    provider_secrets = (api_key,)
    redaction_secrets = tuple(
        value for value in (api_key, args.base_url) if isinstance(value, str) and value
    )
    redactor = Redactor(redaction_secrets)
    child_environment = _provider_safe_child_environment(
        os.environ,
        redaction_secrets,
    )

    try:
        base_guard = WorkspaceGuard(args.workspace)
        policy = CommandPolicy()
        config_runner = CommandRunner(
            base_guard,
            policy,
            child_environment=child_environment,
        )
        spec = load_verification_spec(
            base_guard,
            config_runner,
            args.config,
            known_secrets=provider_secrets,
        )
        guard = protected_guard_for_spec(base_guard, spec)
        runner = CommandRunner(
            guard,
            policy,
            child_environment=child_environment,
        )
        gate = VerificationGate(spec, runner)
    except VerificationConfigError as exc:
        parser.error(f"{exc.kind.value}: {_cli_text(str(exc), redactor)}")
    except (OSError, ProtectedManifestError, ValueError) as exc:
        parser.error(_cli_text(str(exc), redactor))

    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)

    trace_writer: TraceWriter | None = None

    def record_provider_retry(
        attempt: int,
        error: str,
        delay_seconds: float,
    ) -> None:
        if trace_writer is not None:
            trace_writer.record_provider_retry(attempt, error, delay_seconds)

    try:
        model = OpenAICompatibleModel(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            retry_observer=record_provider_retry,
        )
    except Exception as exc:
        parser.error(f"model initialization failed: {type(exc).__name__}")
    try:
        trace_writer = TraceWriter(
            guard.root,
            known_secrets=redaction_secrets,
            artifact_runner=runner,
        )
    except (OSError, TraceError, ValueError) as exc:
        parser.error(_cli_text(str(exc), redactor))

    try:
        result = AgentLoop(
            model,
            registry,
            max_steps=args.max_steps,
            verification_gate=gate,
            context_policy=ContextPolicy(),
            trace_writer=trace_writer,
            known_secrets=provider_secrets,
        ).run(args.task)
    except Exception as exc:
        parser.error(f"agent run failed unexpectedly: {type(exc).__name__}")
    finally:
        _close_trace(trace_writer)

    _print_result(result, redactor)
    return 0 if result.state is AgentState.VERIFIED else 1


def _replay_main(argv: Sequence[str]) -> int:
    parser = build_replay_parser()
    args = parser.parse_args(argv)
    try:
        rendered = replay_trace(args.source)
    except ReplayError as exc:
        parser.error(str(exc))
    print(rendered)
    return 0


def _print_result(result: AgentResult, redactor: Redactor) -> None:
    print(f"state: {result.state.name}")
    print(f"baseline: {_verification_evidence(result.baseline_verification)}")
    print(f"final verification: {_verification_evidence(result.final_verification)}")
    print(f"changed files: {_cli_list(result.changed_files, redactor)}")
    print(f"trace path: {_cli_path(result.trace_path, redactor)}")
    print(f"result path: {_cli_path(result.result_path, redactor)}")
    print(f"patch path: {_cli_path(result.patch_path, redactor)}")
    if result.final_message:
        print(f"message: {_cli_text(result.final_message, redactor)}")
    if result.error is not None:
        print(
            "error: "
            f"{result.error.kind.value}: {_cli_text(result.error.message, redactor)}"
        )


def _verification_evidence(result: VerificationResult | None) -> str:
    if result is None:
        return "not_run"
    exit_codes: list[int | None | str] = [
        command.exit_code for command in result.commands[:CLI_LIST_LIMIT]
    ]
    if len(result.commands) > CLI_LIST_LIMIT:
        exit_codes.append(CLI_TRUNCATION_MARKER)
    failure_kind = (
        result.failure_kind.value if result.failure_kind is not None else "none"
    )
    return " ".join(
        (
            f"passed={str(result.passed).lower()}",
            f"skipped={str(result.skipped).lower()}",
            f"commands={len(result.commands)}",
            f"exit_codes={json.dumps(exit_codes, separators=(',', ':'))}",
            f"protected_unchanged={str(result.protected_unchanged).lower()}",
            f"failure_kind={failure_kind}",
        )
    )


def _cli_list(values: Sequence[str], redactor: Redactor) -> str:
    bounded: list[str] = []
    truncated = False
    for value in values[:CLI_LIST_LIMIT]:
        candidate = [*bounded, _cli_text(value, redactor)]
        if len(_cli_json(candidate)) > CLI_TEXT_LIMIT:
            truncated = True
            break
        bounded = candidate
    if len(bounded) < len(values):
        truncated = True
    if truncated:
        while bounded and len(_cli_json([*bounded, CLI_TRUNCATION_MARKER])) > (
            CLI_TEXT_LIMIT
        ):
            bounded.pop()
        bounded.append(CLI_TRUNCATION_MARKER)
    return _cli_json(bounded)


def _cli_json(values: Sequence[str]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _cli_path(value: str | None, redactor: Redactor) -> str:
    return "unavailable" if value is None else _cli_text(value, redactor)


def _cli_text(value: str, redactor: Redactor) -> str:
    redacted = redactor.text(value)
    normalized = " ".join(redacted.split())
    safe = "".join(
        character if character.isprintable() else "?" for character in normalized
    )
    if len(safe) <= CLI_TEXT_LIMIT:
        return safe
    available = CLI_TEXT_LIMIT - len(CLI_TRUNCATION_MARKER)
    return safe[:available] + CLI_TRUNCATION_MARKER


def _close_trace(trace_writer: TraceWriter) -> None:
    try:
        trace_writer.close()
    except TraceError:
        pass


def _provider_safe_child_environment(
    source: Mapping[str, str],
    provider_secrets: Sequence[str],
) -> dict[str, str]:
    return host_child_environment(source, provider_secrets)


if __name__ == "__main__":
    raise SystemExit(main())
