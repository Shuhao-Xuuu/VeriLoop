# VeriLoop Milestone 3 architecture

## Complete production chain

The CLI is the composition root:

    CLI arguments + provider environment
      -> frozen provider-safe child environment
      -> base WorkspaceGuard + CommandPolicy + config runner
      -> load and freeze VerificationSpec
      -> protected WorkspaceGuard + production CommandRunner
      -> VerificationGate(initial protected manifest)
      -> ToolRegistry + ModelClient + TraceWriter + ContextPolicy
      -> AgentLoop
           -> host baseline before first model request
           -> ContextPolicy -> ModelClient
           -> ToolCall -> ToolRegistry -> local tool -> ToolResult
           -> complete_task -> host final VerificationGate
                -> VERIFIED
                -> RECOVERING -> repairs
                -> STALLED
                -> VERIFICATION_FAILED
      -> events.jsonl -> result.json -> optional patch.diff

Every model action actually executed passes through ToolRegistry.execute.
AgentLoop has no concrete file handler or subprocess path. Gate-owned baseline
and final commands are not model tools; VerificationGate runs their frozen argv
through the policy-bound CommandRunner without advancing model mutation state.

Replay is an early separate branch:

    veriloop replay RUN_DIR_OR_EVENTS_JSONL
      -> load_trace_events -> validate -> format_trace_replay

It runs before provider-key checks and has no edge to ModelClient, ToolRegistry,
CommandRunner, workspace writes, or session restoration.

## Dependency direction

    protocol.py    <- provider-independent values, states, and errors
      model.py     <- provider conversion and retry
      tools.py     <- schemas, validation, Registry, production registration
        filesystem.py <- WorkspaceGuard and file handlers
        process.py    <- CommandPolicy and CommandRunner
      verification.py <- frozen spec, manifests, baseline/final Gate
      context.py      <- deterministic history projection
      trace.py        <- events, artifacts, read-only replay
      agent.py        <- model/Registry plus high-level Gate/context/trace
      cli.py          <- composition root and provider-secret boundary

filesystem.py and process.py know no model/provider objects. verification.py
uses the existing guard, runner, protocol, and safe tool-error boundary but
never drives AgentLoop. context.py handles provider-neutral messages. AgentLoop
coordinates high-level contracts without knowing path canonicalization, SHA
installation, command allowlists, Popen, or provider SDK types.

## Freezing and secret boundaries

cli.main builds one filtered child-environment snapshot before constructing
runners. Sensitive names and allowed values containing provider or secret-named
environment values are removed. CommandRunner filters again and stores a
read-only mapping.

The base guard and policy validate config before model construction.
load_verification_spec parses TOML, validates frozen commands through the same
policy, and copies values into immutable dataclasses/tuples. The protected guard
is derived from that spec, then VerificationGate hashes the initial manifest.
Policy, command order, limits, globs, config path, and manifest therefore exist
before the first model request. A missing config path is also frozen/protected.

The API key is passed to the provider model and host credential checks, not to
Registry, file handlers, CommandPolicy, VerificationGate, or subprocesses.
Exact known credentials in model responses/tool requests fail before execution;
protocol, trace, artifact, and CLI boundaries apply bounded redaction.

## Baseline data flow

AgentLoop starts INITIALIZING, transitions to BASELINE_VERIFYING, emits
baseline_started, and calls VerificationGate.run_baseline before context
projection, step increment, or model invocation:

    frozen policy + commands
      -> CommandRunner.run in order
      -> VerificationCommandResult tuple
      -> VerificationResult(BASELINE, verified_seq=None)
      -> baseline_finished
      -> THINKING or FAILED

must_fail requires all commands to start, no timeout, and at least one nonzero
exit. record_only accepts zero/nonzero exits but not infrastructure failures.
skip/empty commands run nothing and yield explicit skipped evidence. Gate calls
the runner directly and cannot recursively mutate mutation_seq.

The manifest exists before baseline. Baseline does not grant acceptance; final
verification rejects protected changes made at any point since that manifest.

## Registry pairing and completion

For ordinary calls, AgentLoop records the assistant response and executes calls
serially through Registry. Registry validates the schema, invokes the bound
handler, and returns one ToolResult per original ID. Expected
ToolExecutionError values become failed observations.

complete_task is registered normally, but its handler only validates/returns
summary and risks. A solitary valid request continues to the Gate. If completion
is mixed with any other call, AgentLoop intercepts before Registry execution,
executes none, and appends one failed result per ID. This preserves pairing and
guarantees no mixed-response side effect.

## Protected manifest

VerificationGate.__init__ builds deterministic records with relative path,
existence, entry kind, size, and digest. Internal/cache traversal, including
.veriloop evidence, is excluded. The exact config path is included even absent.

protected_guard_for_spec applies the same globs and config path as file-tool
write denies. This does not constrain code launched by run_command, so final
verification compares the initial manifest before and after all frozen commands:

    initial vs pre-command manifest
      + all final commands
      + initial vs post-command manifest
      -> merged CREATED/DELETED/MODIFIED/REPLACED paths
      -> PROTECTED_FILE_CHANGED if non-empty

Only relative paths/change kinds enter evidence, never protected contents.

## Freshness

Registry results carry host-created invalidates_verification. Successful
mutating specs set it automatically; expected errors set it only when an
operation actually started.

    successful edit/create/overwrite
    or started model run_command (zero/nonzero/timeout)
      -> invalidates_verification=true
      -> mutation_seq += 1
      -> verified_seq=None
      -> workspace_revision_changed

Read-only calls, schema/unknown/denied calls, command start errors, failed file
operations, completion, and Gate commands do not advance it. run_final receives
the current sequence. It sets result verified_seq only on pass, and
grants_verified rechecks command count, started/timeout/exits, protected
integrity, result sequence, and verified_seq == mutation_seq before AgentLoop
transitions VERIFIED.

## Why the model cannot grant VERIFIED

1. AgentState is host-owned; model text remains content.
2. complete_task rejects verified/passed/state and all extra fields.
3. Model-requested pytest is an ordinary mutation and cannot set verified_seq.
4. The only VERIFIED transition follows host final evidence and
   VerificationGate.grants_verified.

The model cannot choose Gate commands because the spec was frozen before its
first request. Tool-result metadata is created by Registry/host helpers, not
accepted from the provider as verification authority.

## Repair and failure signature

    EXECUTING -> valid complete_task -> VERIFYING -> Gate.run_final
      -> pass -> paired success result -> VERIFIED
      -> failure -> paired evidence result
           -> same-signature threshold -> STALLED
           -> budget remains -> RECOVERING -> THINKING
           -> exhausted -> VERIFICATION_FAILED

The failure result enters canonical history before recovery. Entering recovery
consumes one round; max_repair_rounds=N permits the initial attempt plus N
repaired attempts.

The signature hashes deterministic failure kind, normalized argv/cwd/start/
timeout/exit/error data, bounded output tails, and sorted protected changes.
Workspace/temp roots, timestamps, elapsed values, volatile IDs, and random
temporary paths are normalized; duration is excluded. Equal consecutive
signatures increment the counter. New/missing signatures reset it. Reaching
max_same_failure stops before another model call.

## ContextPolicy

Canonical history remains in AgentLoop. ContextPolicy.project creates a copy,
requires system/user anchors, and partitions later messages into complete
assistant groups. A tool-call assistant message and all ordered ToolResults are
indivisible.

Over budget, it pins recent groups and the latest verification-failure group,
then removes oldest other groups. Remaining copied text/arguments receive an
explicit deterministic marker. Anchors, IDs, and pairing take precedence over
the soft limit, so structural minimum may exceed it. Canonical history is never
mutated and no summarization model is called.

## Trace, result, and patch lifecycle

TraceWriter creates a new run directory and create-only events.jsonl. emit adds
monotonic seq, timestamp, run ID, event type, state, and bounded payload, then
flushes. Events cover baseline, transitions, model summaries/retries, tool
lifecycle, revisions, completion, verification, recovery, and termination.

    source evidence
      -> event-specific safe payload construction
      -> exact-known-secret/Bearer redaction
      -> preview/collection/depth bounds
      -> deterministic JSON -> append + flush

Redaction precedes preview truncation. Content arguments become length/digest;
streams and outputs are bounded; sensitive keys are omitted. This is defense in
depth, not universal DLP.

At termination AgentLoop emits run_finished, closes trace, and requests
artifacts. result.json is atomic/create-only and mirrors in-memory state and
verification evidence. Optional patch.diff uses policy-approved read-only
git rev-parse/diff, rejects incomplete output, redacts, and installs atomically.
It excludes untracked files and is not a complete changeset.

Trace/artifact failures make evidence unavailable rather than changing
freshness or terminal state.

## Read-only replay

load_trace_events performs bounded UTF-8 reads and rejects empty/oversized
input, malformed/nonstandard JSON, bad schema/event/state/run ID, nonmonotonic
sequence, and malformed display payloads. format_trace_replay revalidates and
renders an allowlisted bounded summary. replay_trace merely composes them: no
model, tool, command, Git operation, write, resume, or pending action exists.

## State machine

    INITIALIZING -> BASELINE_VERIFYING -> FAILED | THINKING
    THINKING
      -> plain final -> COMPLETED_UNVERIFIED
      -> ordinary tools -> EXECUTING -> THINKING
      -> mixed completion -> EXECUTING -> paired deferred results -> THINKING
      -> solitary completion -> EXECUTING
           -> no commands -> COMPLETED_UNVERIFIED
           -> commands -> VERIFYING
                -> VERIFIED
                -> RECOVERING -> THINKING
                -> STALLED
                -> VERIFICATION_FAILED
                -> FAILED/CANCELLED on unrecoverable host failure
      -> provider/internal failure -> FAILED
      -> step bound -> MAX_STEPS
      -> interrupt -> CANCELLED

VERIFIED and all other named terminal states end that run. Plain final during
recovery remains unverified and cannot reuse prior failed evidence as success.

## Preserved M2 trust boundary and excluded scope

WorkspaceGuard still enforces canonical containment, symlink defenses, bounded
UTF-8 reads, SHA preconditions, atomic replacement, and no-clobber create.
CommandPolicy remains argv-only with hard denials; CommandRunner remains
shell=False with bounded temporary-file output, filtered frozen environment,
timeouts, and platform-specific cleanup. POSIX process-group cleanup remains
stronger than Windows direct-child/best-effort descendant cleanup.

Neither protected paths, cwd containment, nor command policy is an OS/network
sandbox. Allowed repository code may access outside paths or the network.
Milestone 3 adds no multiple agents, streaming, parallel tools, session resume,
rollback, automatic Git mutation, dependency installation, or replayed effects.
