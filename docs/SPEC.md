# VeriLoop milestone specification

## Milestone status

Milestone 1 is complete and passed independent review. It established the
provider-independent protocol, non-streaming ModelClient, bounded provider
retry, ToolRegistry, ScriptedModel, synchronous AgentLoop, exact call/result
pairing, step limits, and honest COMPLETED_UNVERIFIED completion.

Milestone 2 is complete and passed independent review. Its green commit is
038b6298bc67c95998ff7df21af96fd1d6d4f221. It added WorkspaceGuard, bounded
UTF-8 file tools, SHA-guarded atomic writes, argv-only CommandPolicy and
CommandRunner, timeout/cleanup, bounded output, filtered child environments,
production registration, and offline temporary-workspace tests. Those path,
SHA, atomic-write, command, timeout, output, and environment boundaries remain.

Milestone 3 is implemented. It adds a frozen VerificationSpec, baseline and
final VerificationGate, complete_task, mutation freshness, protected manifests,
bounded repair and stall detection, deterministic ContextPolicy, redacted JSONL
traces, result and optional patch artifacts, read-only replay, CLI wiring, and
offline production-component end-to-end tests. This does not claim an
independent review of the Milestone 3 candidate.

## Host-owned authority and preserved loop contract

The model may request acceptance, but only host code may grant VERIFIED.
complete_task has required non-empty summary, optional remaining_risks, and
additionalProperties=false; it cannot accept verified, passed, state, or exit
evidence. Model-requested pytest is only a tool observation and mutation. Model
prose, arguments, and result metadata cannot set host state or verified_seq.

AgentLoop enters VERIFIED only after a host-created final VerificationResult
satisfies VerificationGate.grants_verified. Plain final text always becomes
COMPLETED_UNVERIFIED without final verification. Completion with no configured
verification commands is also COMPLETED_UNVERIFIED.

Provider SDK objects do not enter the internal protocol. ModelClient never
executes tools. AgentLoop has no concrete file handler or subprocess branch;
every model action actually executed passes through ToolRegistry.execute. Host
baseline/final commands instead use the gate's policy-bound CommandRunner.

Every executed or deliberately deferred model call produces one ToolResult with
its unchanged ID. If a response mixes complete_task with other calls, none
execute: completion receives COMPLETION_MUST_BE_SINGLE_CALL and all others
receive DEFERRED_REPLAN_REQUIRED. The complete group enters canonical history.

## Frozen VerificationSpec

load_verification_spec reads workspace-relative UTF-8 TOML before the first
model request. It rejects unsafe, symlinked, oversized, unknown, malformed,
policy-denied, or known-credential-bearing input. Values become frozen
dataclasses and tuples containing baseline_policy, ordered tuple-argv commands
with normalized cwd and bounded timeout, repair/stall limits, normalized
protected_globs, and the config path.

A representative configuration has a verification table with baseline_policy,
max_repair_rounds, max_same_failure, protected_globs, and an array of command
tables containing argv, cwd, and timeout_seconds. Commands are argv lists, never
shell strings, and must pass the Milestone 2 command policy.

Later disk edits cannot change the in-memory spec. A missing config freezes an
empty command list while retaining and protecting the expected path. File tools
cannot create it mid-run; command-side creation is detected by the final
manifest. Empty commands permit only unverified execution. Production
config/manifest errors stop during CLI composition before an AgentResult exists.

## Baseline policies

The production CLI supplies a gate. Before the first model request AgentLoop
enters BASELINE_VERIFYING and stores a baseline result.

- must_fail requires all commands to start, no timeout, and at least one
  nonzero exit. All-zero is BASELINE_UNEXPECTED_PASS.
- record_only permits zero or nonzero exits, but not start errors/timeouts.
- skip starts no command and records an explicit skipped result.

No commands also yields a skipped baseline. Start error or timeout is
BASELINE_INFRASTRUCTURE_ERROR and ends FAILED. Gate commands never advance
mutation_seq. Baseline evidence enters AgentResult, trace, result.json, and CLI
summary when those evidence outputs are available.

## Protected manifest

VerificationGate builds its initial manifest before baseline/model execution.
It records deterministic workspace-relative matched entries with existence,
kind, size, and SHA-256 where applicable, and includes the config path even
when absent. Link-like entries are recorded without following them.

Final verification compares the original manifest before and after its commands,
detecting created, deleted, modified, and type-replaced entries. Evidence has
paths and change kinds, not contents. Any change produces
PROTECTED_FILE_CHANGED, even if commands exit zero.

protected_guard_for_spec adds the globs and exact config path to file-tool write
denies. This blocks edit/create/overwrite but is not an OS sandbox: repository
code launched by run_command may still mutate protected paths, so the final
manifest check remains mandatory.

## Freshness and final Gate invariants

Runs start with mutation_seq=0 and verified_seq=None. Successful edit, create,
and overwrite advance mutation. A model run_command advances it whenever the
process starts, including zero exit, nonzero exit, and timeout. Reads, searches,
lists, unknown/invalid calls, denied commands, start errors, failed file writes,
complete_task, and Gate commands do not advance it.

Every invalidating model result increments mutation_seq and clears verified_seq.
A passing final gate sets verified_seq to the current mutation sequence.
VERIFIED requires all of the following:

- a non-empty frozen command list and final passed result;
- exactly the frozen command count;
- every command started, did not time out, and exited zero;
- no protected change; and
- result/run mutation agreement with verified_seq == mutation_seq.

VERIFIED is terminal; no later model or tool execution occurs.

## Completion, repair, and terminal states

A solitary valid complete_task passes Registry validation, records completion,
enters VERIFYING, and receives one paired Gate result. Failure evidence,
including bounded command output and protected changes, enters history.

Entering RECOVERING consumes one repair round. max_repair_rounds=N permits one
initial final attempt plus N repaired attempts. Exhaustion becomes
VERIFICATION_FAILED with no extra model call.

Failure signatures hash failure kind, normalized command results/output tails,
and sorted protected changes. Workspace/temp roots, timestamps, durations, and
other volatile text are normalized. Equal consecutive signatures increment the
counter; a different or missing signature resets it. Reaching max_same_failure
yields STALLED before more repair budget is spent.

Milestone 3 adds BASELINE_VERIFYING, VERIFYING, RECOVERING, VERIFIED,
VERIFICATION_FAILED, and STALLED while preserving INITIALIZING, THINKING,
EXECUTING, COMPLETED_UNVERIFIED, FAILED, MAX_STEPS, and CANCELLED.

## Deterministic ContextPolicy

ContextPolicy.project deep-copies canonical history. Within its soft limit it
does not prune. Otherwise it retains system/user structure, groups each
assistant call message with all ordered results, pins recent groups and the
latest verification-failure group, and removes oldest unpinned groups first.
It never creates an orphan call/result.

If selected structure remains oversized, copied text and arguments receive an
explicit deterministic marker. The limit is soft: anchors, IDs, and complete
pairing take precedence. Canonical history is unchanged and no model summary API
is used.

## Trace, artifacts, and replay

TraceWriter creates .veriloop/runs/{run_id}/events.jsonl with create-only
semantics, sequence starting at 1, and flush after each line. Events cover state,
model summaries, retries, tools, revisions, completion, verification, recovery,
and terminal outcomes.

Exact configured secrets and Authorization Bearer values are redacted before
preview truncation. Forbidden keys are omitted; content arguments become
length/digest records; outputs/collections are bounded. This is deterministic
defense in depth, not general DLP for every unknown or transformed secret.

At termination the host attempts atomic non-overwriting result.json containing
state, counts, freshness, baseline/final evidence, protected changes,
file-tool changed paths, usage, error, and artifact metadata. Optional
patch.diff is a bounded redacted ordinary git diff; it excludes untracked files
and is not a complete changeset. Trace/artifact failure may make paths
unavailable but cannot change the already determined agent state.

replay_trace accepts a run directory or events file, performs bounded strict
JSONL/schema/run-ID/sequence/event validation, and formats allowlisted evidence.
It never creates a model, registry, runner, or writable session; it does not
execute calls, modify the workspace, restore pending work, or resume.

## Preserved M2 boundaries and CLI

WorkspaceGuard continues to reject absolute, drive, UNC, parent-traversal,
alternate-data-stream, symlink-escape, and canonical outside-root paths.
Sensitive files and .git/.veriloop remain denied. File content is bounded UTF-8;
edits/overwrites retain SHA preconditions and atomic replacement; create remains
atomic no-clobber. No append/delete/rename/chmod/directory-create API exists.

run_command remains argv-only with relative cwd, bounded timeout, shell=False,
no interactive stdin, hard policy denials, narrow Python allowances, bounded
temporary-file output, and process cleanup. POSIX targets process groups;
Windows guarantees the direct child and only best-effort descendants.

The CLI freezes a filtered child environment before runner construction.
Credential-like names and allowed values containing known provider or
secret-named values are removed. CommandRunner stores a read-only snapshot and
does not consult live os.environ; provider credentials never reach Registry or
subprocesses.

The CLI supports veriloop TASK, veriloop run TASK, and key-free replay.
Only VERIFIED returns run exit code 0; other agent terminals return 1, while
argument/config/setup errors use argparse exit code 2. Successful replay returns
0 and malformed replay input is an argparse error with exit code 2.

Workspace and command policy are not filesystem/network isolation. Allowed
repository code may access outside paths or the network. VeriLoop has no OS
sandbox and must run in trusted/disposable workspaces. M3 intentionally excludes
multiple agents, streaming, parallel tools, UI, MCP/plugins, RAG, patch engines,
dependency installation, session resume, rollback, automatic Git mutation,
long-term memory, background agents, and replayed side effects.
