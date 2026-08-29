"""The minimal synchronous model → tool → model loop."""

from __future__ import annotations

from .model import ModelClient, ModelClientError
from .protocol import (
    AgentError,
    AgentResult,
    AgentState,
    ErrorKind,
    Message,
    Role,
)
from .tools import ToolRegistry


DEFAULT_SYSTEM_PROMPT = (
    "You are the model inside VeriLoop. Use the supplied tools when needed, "
    "then return a concise final response."
)


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        *,
        max_steps: int = 12,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._system_prompt = system_prompt

    def run(self, task: str) -> AgentResult:
        state = AgentState.INITIALIZING
        history = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=task),
        ]
        step_count = 0
        tool_call_count = 0

        while True:
            if step_count >= self._max_steps:
                state = AgentState.MAX_STEPS
                return AgentResult(
                    state=state,
                    final_message="",
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                    history=tuple(history),
                    error=AgentError(
                        kind=ErrorKind.MAX_STEPS,
                        message=f"maximum model steps reached: {self._max_steps}",
                    ),
                )

            state = AgentState.THINKING
            step_count += 1
            try:
                response = self._model.complete(
                    list(history),
                    self._tools.schemas(),
                )
            except KeyboardInterrupt:
                state = AgentState.CANCELLED
                return AgentResult(
                    state=state,
                    final_message="",
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                    history=tuple(history),
                    error=AgentError(
                        kind=ErrorKind.CANCELLED,
                        message="agent run cancelled",
                    ),
                )
            except ModelClientError as exc:
                state = AgentState.FAILED
                return AgentResult(
                    state=state,
                    final_message="",
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                    history=tuple(history),
                    error=AgentError(
                        kind=exc.kind,
                        message=str(exc),
                        retryable=exc.retryable,
                    ),
                )
            except Exception as exc:
                state = AgentState.FAILED
                return AgentResult(
                    state=state,
                    final_message="",
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                    history=tuple(history),
                    error=AgentError(
                        kind=ErrorKind.INTERNAL_ERROR,
                        message=f"unexpected model error: {type(exc).__name__}: {exc}",
                    ),
                )

            history.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:
                state = AgentState.COMPLETED_UNVERIFIED
                return AgentResult(
                    state=state,
                    final_message=response.text,
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                    history=tuple(history),
                )

            state = AgentState.EXECUTING
            for call in response.tool_calls:
                try:
                    result = self._tools.execute(call)
                except KeyboardInterrupt:
                    state = AgentState.CANCELLED
                    return AgentResult(
                        state=state,
                        final_message="",
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        history=tuple(history),
                        error=AgentError(
                            kind=ErrorKind.CANCELLED,
                            message="agent run cancelled during tool execution",
                        ),
                    )
                tool_call_count += 1
                history.append(
                    Message(
                        role=Role.TOOL,
                        content=result.content,
                        tool_result=result,
                    )
                )
