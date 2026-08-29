"""Command-line composition root for the Milestone 2 local harness."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .agent import AgentLoop
from .filesystem import WorkspaceGuard
from .model import OpenAICompatibleModel
from .process import CommandPolicy, CommandRunner
from .protocol import AgentState
from .tools import ToolRegistry, register_workspace_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VeriLoop Milestone 2 harness")
    parser.add_argument("task", help="task to send to the model")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--workspace",
        default=".",
        help="local workspace root (default: current directory)",
    )
    parser.add_argument("--max-steps", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY must be set in the environment")
    if not args.model:
        parser.error("--model or OPENAI_MODEL is required")

    try:
        guard = WorkspaceGuard(args.workspace)
    except ValueError as exc:
        parser.error(str(exc))
    runner = CommandRunner(guard, CommandPolicy())
    registry = ToolRegistry()
    register_workspace_tools(registry, guard, runner)

    model = OpenAICompatibleModel(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
    )
    result = AgentLoop(
        model,
        registry,
        max_steps=args.max_steps,
    ).run(args.task)

    if result.final_message:
        print(result.final_message)
    elif result.error is not None:
        print(f"{result.error.kind.value}: {result.error.message}")
    print(f"state: {result.state.name}")
    return 0 if result.state is AgentState.COMPLETED_UNVERIFIED else 1


if __name__ == "__main__":
    raise SystemExit(main())
