# VeriLoop Milestone 1 specification

## Scope

Milestone 1 implements the minimum synchronous, provider-independent harness:

- immutable internal protocol objects;
- a `ModelClient` protocol and non-streaming OpenAI-compatible Chat Completions
  adapter;
- conversion of provider responses into internal `ModelResponse` values;
- bounded provider retry classification;
- a `ToolRegistry` with a deliberately small JSON Schema subset;
- a synchronous `AgentLoop`;
- a deterministic, network-free `ScriptedModel` and unit tests;
- a minimal `argparse` CLI with no production tools.

No agent framework or agent SDK is used. The harness owns message conversion,
tool-call argument parsing, call/result identity, parameter validation, history,
loop control, error classification, retries, and termination.

Milestone 2 functionality is not implemented: file/search/edit/write tools,
command execution, `WorkspaceGuard`, path or symlink protection, and command
safety. Milestone 3 functionality is also not implemented: a Verification Gate,
mutation/verification sequencing, context compression, trace, replay, rollback,
or debugging logs. There are no placeholder implementations for these features.

## Internal protocol

`protocol.py` defines string enums for roles, finish reasons, error kinds, and
agent states. Frozen dataclasses define:

- `ToolCall(id, name, arguments)`;
- `ToolResult(call_id, tool_name, ok, content, error_kind, retryable, metadata)`;
- `ModelResponse(text, tool_calls, finish_reason, usage)`;
- `Message(role, content, tool_calls, tool_result)`;
- `AgentError` and `AgentResult`.

Protocol objects never contain OpenAI SDK response objects. A `ToolCall.id` is
copied unchanged to exactly one `ToolResult.call_id` so a later model request can
associate results with calls.

## Model contract and provider errors

`ModelClient.complete(messages, tools) -> ModelResponse` is the only model
contract known by the loop. `OpenAICompatibleModel` sends a non-streaming Chat
Completions request and translates the first assistant choice to this contract.
It preserves tool-call order, IDs, names, parsed object arguments, text, internal
finish reason, and available input/output/total token counts.

Assistant `content=None` becomes an empty string. Tool arguments are parsed with
the standard-library `json` module. Invalid JSON or a valid non-object top level
raises `ModelProtocolError`; no tool can execute from such a response.

A `complete` operation makes at most three provider requests: the initial
request and two retries. Timeouts, connection errors, rate limits, HTTP 408/409,
HTTP 429, and HTTP 5xx are retryable. Authentication, permission, bad-request,
not-found, model/configuration, and other non-transient failures are fatal.
Retry exhaustion becomes `PROVIDER_RETRY_EXHAUSTED`; a fatal error becomes
`PROVIDER_FATAL_ERROR` immediately. Sleep is injectable, so tests never wait.

## Tool registry contract

`ToolSpec` stores a name, description, input schema, Python handler, and
`mutates_workspace` flag. `ToolRegistry.schemas()` exposes only standard model
tool schemas—not handlers or registry internals. `execute()` is the only handler
invocation path used by `AgentLoop`.

Duplicate names raise an explicit registration error. Execution always returns
one `ToolResult`, including for unknown tools, invalid arguments, and handler
exceptions. Supported schema types are object, string, integer, number, boolean,
and array, with object `properties`, `required`, and `additionalProperties`.
Undeclared object fields are rejected by default. Python `bool` is deliberately
excluded from integer and number matches.

Tool errors do not terminate the loop. Their `ToolResult` is appended to history
so the next assistant turn can correct the name or arguments.

## Agent loop and termination

One step is one `ModelClient.complete` call. For `max_steps=N`, the loop makes no
more than N such calls and checks the remaining budget before each call.

The loop:

1. appends `SYSTEM`, then `USER` to history;
2. calls the model with a copy of complete history and current tool schemas;
3. appends one `ASSISTANT` message containing all text and tool calls;
4. if calls exist, executes them serially in model order through `ToolRegistry`;
5. appends exactly one `TOOL` message/result per call, preserving each ID;
6. calls the model again only after every result from that turn is present;
7. stops on a response with no tool calls.

A final text response terminates only as `COMPLETED_UNVERIFIED`. This means the
model has stopped asking for tools and supplied final text; no independent
Verification Gate exists yet, so the state must not be described as verified or
successful. Other terminal states are `FAILED`, `MAX_STEPS`, and `CANCELLED`.
Model protocol/fatal/retry-exhausted errors produce `FAILED`; `KeyboardInterrupt`
produces `CANCELLED`.
