"""Minimal command-line entry point for the Milestone 1 harness."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .agent import AgentLoop
from .model import OpenAICompatibleModel
from .protocol import AgentState
from .tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VeriLoop Milestone 1 harness")
    parser.add_argument("task", help="task to send to the model")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
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

    model = OpenAICompatibleModel(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
    )
    result = AgentLoop(
        model,
        ToolRegistry(),
        max_steps=args.max_steps,
    ).run(args.task)

    if result.final_message:
        print(result.final_message)
    elif result.error is not None:
        print(f"{result.error.kind.value}: {result.error.message}")
    return 0 if result.state is AgentState.COMPLETED_UNVERIFIED else 1


if __name__ == "__main__":
    raise SystemExit(main())
