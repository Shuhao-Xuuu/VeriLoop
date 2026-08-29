# Milestone 1 architecture

## Current call chain

```text
CLI
 |
 v
AgentLoop ----------------------+
 |                              |
 v                              v
ModelClient                ToolRegistry
 |                              |
 v                              v
OpenAI-compatible provider   test-injected handler
```

The model only decides whether to return text and/or structured tool calls. It
does not execute tools. `AgentLoop` coordinates turns but never invokes a Python
handler directly; every call goes through `ToolRegistry.execute`. A handler has
no dependency on the model.

In the production CLI, the concrete model is `OpenAICompatibleModel` and the
registry is intentionally empty in Milestone 1. Tests inject `ScriptedModel` and
test-only handlers to exercise the complete loop without network or workspace
access.

## Module dependency direction

```text
protocol.py       <- provider-independent values; no harness implementation deps
     ^
     |---- model.py  <- ModelClient boundary, provider translation, retries
     |---- tools.py  <- ToolSpec, validation, handler isolation
     +---- agent.py  <- depends only on ModelClient + ToolRegistry contracts
              ^
              |
            cli.py   <- composition root
```

`protocol.py` knows nothing about OpenAI, registries, handlers, or the loop.
`model.py` knows how to serialize internal history and parse one provider
response, but it cannot see or run Python handlers. `tools.py` knows nothing
about provider SDKs. This keeps provider objects outside `AgentLoop`.

## Core data and provider boundary

The adapter converts outgoing `Message` values to provider dictionaries and a
provider assistant choice back to:

```text
ModelResponse
├── text
├── tool_calls[] -> ToolCall(id, name, arguments object)
├── finish_reason -> internal FinishReason
└── usage -> plain dict[str, int]
```

No raw response, choice, message, or SDK exception crosses the boundary. JSON
tool arguments are parsed before the loop sees them. Malformed/non-object
arguments stop as `MODEL_PROTOCOL_ERROR`, so they cannot reach the registry.

## State flow

```text
INITIALIZING
     |
     v
  THINKING -- model error --------------------> FAILED
     |  \
     |   \-- KeyboardInterrupt --------------> CANCELLED
     |
     +-- no tool calls -----------------------> COMPLETED_UNVERIFIED
     |
     v
 EXECUTING -- append every ToolResult in order --+
     ^                                            |
     +---------------- THINKING <-----------------+
                         |
                         +-- no budget ----------> MAX_STEPS
```

The loop checks the budget immediately before `THINKING` calls the model. A
provider adapter may make its bounded HTTP retries internally, but that remains
one `ModelClient.complete` step from the loop's perspective.

## History growth and call identity

For one assistant turn containing calls A, B, and C, history grows as:

```text
SYSTEM
USER
ASSISTANT(tool_calls=[A, B, C])
TOOL(result A, call_id=A.id)
TOOL(result B, call_id=B.id)
TOOL(result C, call_id=C.id)
ASSISTANT(...next response...)
```

Calls execute serially in their tuple order. `ToolRegistry.execute` returns a
result on success, unknown name, invalid arguments, or handler exception. Thus
each call yields exactly one result with the unchanged call ID. The following
model request receives the entire accumulated history, including tool errors,
and can repair its decision.
