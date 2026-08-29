# Milestone 1 implementation handoff

This guide is for understanding and defending the implementation, not for
advertising it. Read the modules in protocol → model/tools → agent order.

## What exists and what does not

The current harness implements internal messages, OpenAI-compatible response
translation, provider error mapping and retry, a small validating tool registry,
the synchronous loop, an empty-registry CLI, and deterministic offline tests.

It does not implement file tools, command execution, `WorkspaceGuard`, path or
command safety, context compression, a Verification Gate, trace, replay, or
mutation/verification tracking. A model's final response is therefore
`COMPLETED_UNVERIFIED`.

## End-to-end execution

`cli.main` reads the API key only from the environment and reads other model
configuration from arguments/environment. It constructs
`OpenAICompatibleModel`, an empty `ToolRegistry`, and `AgentLoop`, then calls
`run(task)`.

`AgentLoop.run` initializes `SYSTEM` and `USER` messages. It passes complete
history plus `ToolRegistry.schemas()` to the injected `ModelClient`. The adapter
uses a non-streaming Chat Completions request and returns a plain
`ModelResponse`; no provider object reaches the loop.

Every response becomes an `ASSISTANT` message. When it contains calls, the loop
passes each call, in order, to `ToolRegistry.execute`. The registry finds the
spec, validates the small schema subset, invokes the handler inside an exception
boundary, and returns a `ToolResult`. The loop appends one `TOOL` message per
call. Its `call_id` is always the original `ToolCall.id`, including failures.

Unknown tools, bad arguments, and handler exceptions stay in history instead of
ending the run. The next request sees those results and can correct itself. A
turn without calls ends `COMPLETED_UNVERIFIED`. Model protocol, fatal provider,
or retry-exhausted errors end `FAILED`; step exhaustion ends `MAX_STEPS`; an
interrupt ends `CANCELLED`.

Provider retries happen inside one adapter `complete` call. There are at most
three requests: initial, retry 1, retry 2. Retryable failures sleep through an
injected callable; fatal failures do not sleep or retry. The loop's `max_steps`
counts calls to `ModelClient.complete`, not provider-level retry attempts or
individual tool calls, and it checks before calling so request N+1 cannot occur.

## Behavior-to-test map

| Behavior | Test location |
| --- | --- |
| Text, `None` content, usage, finish mapping | `tests/test_model.py` response tests |
| JSON arguments, invalid JSON, non-object JSON, multiple calls | `tests/test_model.py` tool-call tests |
| Temporary success, three-request exhaustion, fatal no-retry | `tests/test_model.py` retry tests |
| Schema exposure and successful execution | `tests/test_tools.py` registry tests |
| Duplicate/unknown/missing/type/extra/handler errors | `tests/test_tools.py` error tests |
| `bool` excluded from integer/number | `tests/test_tools.py::test_bool_is_not_an_integer_or_number` |
| Direct response and single-tool round trip | `tests/test_agent_loop.py` opening tests |
| Full read → transform → final three-turn trajectory | `test_complete_three_turn_deterministic_trajectory` |
| Same-turn tool order and one result per call | `test_multiple_calls_in_one_response_execute_serially_in_order` |
| Unknown-tool recovery | `test_unknown_tool_error_is_seen_and_model_recovers` |
| Invalid arguments/tool exception visible next turn | `test_tool_errors_enter_next_request_without_ending_loop` |
| Exact step limit/no extra call | `test_max_steps_stops_exactly_without_extra_model_call` |
| Fatal, retry-exhausted, protocol, cancellation states | final error tests in `tests/test_agent_loop.py` |

`tests/scripted_model.py` records copies of every message list and schema list,
returns or raises the next scripted outcome, fails clearly when exhausted, and
never imports or calls a network client. Its recorded calls prove that each next
turn receives prior tool results.

## Code reading checklist

1. `src/veriloop/protocol.py`: understand every immutable boundary value and why
   provider SDK types are absent.
2. `src/veriloop/model.py::_parse_provider_response` and `_parse_tool_call`:
   understand how nullable content, call IDs, JSON objects, finish reasons, and
   usage become a `ModelResponse`.
3. `src/veriloop/model.py::OpenAICompatibleModel.complete` and
   `_is_retryable_provider_error`: understand the three-request bound and why
   protocol/fatal errors are not blindly retried.
4. `src/veriloop/tools.py::ToolRegistry.execute`, `_validate_value`, and
   `_matches_type`: understand the handler boundary, default extra-field
   rejection, and Python `bool` special case.
5. `src/veriloop/agent.py::AgentLoop.run`: trace the budget check, assistant
   append, serial execution, result append, error feedback, and every terminal
   result.

## Project author must be able to answer these 10 questions

1. **Why is `ModelClient` a `Protocol`?** It expresses only the behavior the loop
   needs, so production and scripted clients are injected without inheritance or
   provider coupling.
2. **Why can the provider response not pass directly to `AgentLoop`?** SDK
   objects would couple loop logic and tests to one vendor. Translation produces
   stable, immutable internal values.
3. **What does `ToolCall.id` do?** It is the correlation key. The unchanged value
   becomes `ToolResult.call_id` and the provider's later tool message ID.
4. **Why does a tool failure not exit the loop?** Unknown names and invalid
   arguments are model-correctable. Returning the error as a tool result gives
   the next turn evidence needed to repair the call.
5. **How are several calls in one turn ordered?** The adapter preserves provider
   order in a tuple; the loop iterates it synchronously and immediately appends
   each corresponding result.
6. **Why does `max_steps` count model calls?** A step represents one reasoning
   turn. Counting tools or HTTP retries would make the assistant-turn budget
   ambiguous. A pre-call check prevents the N+1 turn.
7. **How do retryable and fatal provider errors differ?** Timeouts, connections,
   rate limits, and transient HTTP statuses may recover and get two retries.
   configuration/client failures are normalized as fatal and return immediately.
8. **Why exclude `bool` from integer and number?** In Python, `bool` subclasses
   `int`; JSON Schema treats boolean and numeric types as distinct.
9. **Why not call the final state `SUCCESS`?** Milestone 1 only observes that the
   model returned final text. No independent Verification Gate has validated the
   task, so `COMPLETED_UNVERIFIED` is the honest state.
10. **How does `ScriptedModel` prove the loop is offline and provider-agnostic?**
    It satisfies the structural `ModelClient` contract with predetermined values,
    records histories/schemas for assertions, and contains no provider or network
    access.
