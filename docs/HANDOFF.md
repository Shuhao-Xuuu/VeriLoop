# VeriLoop Milestone 3 implementation handoff

This guide is for learning and defending the implementation. Milestone 1 was
independently reviewed and remains intact. Milestone 2 added real local tools
without changing who owns the loop or how termination works, and its reviewed
green commit is `038b6298bc67c95998ff7df21af96fd1d6d4f221`. Milestone 3 adds
host-owned verification and evidence without replacing those foundations.

## Current state

The harness now provides the complete intended local agent loop:
provider-independent messages, bounded provider retry, validated tool calling,
synchronous loop control, five guarded file tools, one allowlisted process tool,
an explicit `complete_task` request, a host-owned `VerificationGate`, bounded
repair, deterministic context projection, redacted JSONL evidence, structured
result artifacts, and read-only replay. Every model tool that is actually
executed still passes through `ToolRegistry`; mixed completion responses are
rejected by `AgentLoop` before any handler runs while preserving call/result
pairing.

The model can request acceptance but cannot grant it. Ordinary final text is
always `COMPLETED_UNVERIFIED`, as is `complete_task` when there are no configured
verification commands. Only a successful final Gate result can produce
`VERIFIED`.

## Milestone 1 knowledge that still governs the system

`ModelClient.complete(messages, tools) -> ModelResponse` remains the only model
contract visible to `AgentLoop`. `OpenAICompatibleModel` translates provider
objects to plain protocol dataclasses and keeps provider retry inside one model
step. No provider SDK response enters the loop.

`ToolRegistry.execute` remains the sole handler invocation boundary. The loop
appends one assistant message, runs all tool calls serially in returned order,
and appends exactly one tool message per call. Every result retains the original
call ID. Unknown tools, invalid arguments, expected local failures, and handler
errors enter history so the next model request can react.

`max_steps` counts calls to `ModelClient.complete`, not provider retries or tool
calls. The loop checks before each model call, so it never makes request N+1.
Model errors end `FAILED`, exhaustion ends `MAX_STEPS`, interruption ends
`CANCELLED`, and ordinary final text ends `COMPLETED_UNVERIFIED`.

## End-to-end production composition

`cli.main` accepts both `veriloop <task>` and `veriloop run <task>`. It handles
`veriloop replay <run-dir-or-events.jsonl>` before reading any provider key or
constructing runtime components. Normal run parses the task, model
configuration, workspace, step limit, and workspace-relative verification
configuration. It reads `OPENAI_API_KEY` only after argument parsing, so
`--help` works without a key.

Normal run then constructs, in order:

1. a canonical base `WorkspaceGuard` and `CommandPolicy`;
2. a provider-safe frozen child environment and configuration `CommandRunner`;
3. an immutable `VerificationSpec` loaded before the first model request;
4. a model-facing guard containing the frozen protected write policy;
5. a `CommandRunner` bound to that protected guard and the existing policy;
6. a `VerificationGate`, which records the initial protected manifest;
7. a `ToolRegistry` populated by `register_workspace_tools`;
8. `OpenAICompatibleModel`;
9. `TraceWriter` and `ContextPolicy`;
10. `AgentLoop` with those host-owned components injected.

The model is initialized before the trace directory is created, so a model
constructor failure does not leave an empty run. Once the loop starts, baseline
verification occurs before its first model request.

The complete runtime path is:

```text
CLI
 -> load/freeze VerificationSpec
 -> protected manifest / protected WorkspaceGuard
 -> optional baseline verification
 -> AgentLoop
 -> ContextPolicy
 -> ModelClient
 -> ToolRegistry
 -> WorkspaceGuard / CommandPolicy
 -> File Tool / CommandRunner
 -> ToolResult
 -> history
 -> ModelClient
 -> complete_task
 -> VerificationGate
    -> VERIFIED / RECOVERING / STALLED / VERIFICATION_FAILED
 -> TraceWriter
 -> result.json / optional patch.diff
```

`build_workspace_tools` still provides production registry construction for
tests or embedding. The verified CLI deliberately performs the more explicit
composition above so the same frozen spec controls the protected guard, Gate,
trace artifact runner, and loop. Neither path creates a dependency framework.

## Every production tool entry

All seven schemas and bindings live in `src/veriloop/tools.py`.

- `list_files` binds `filesystem.list_files(guard, **arguments)`. It is read-only
  and exposes path/depth/result limits with no extra properties.
- `read_file` binds `filesystem.read_file`. It requires a path and supplies
  bounded default line numbers.
- `search_text` binds `filesystem.search_text`. It requires a literal query and
  exposes case sensitivity plus a result bound.
- `edit_file` binds `filesystem.edit_file` and marks the existing
  `ToolSpec.mutates_workspace` flag true. The schema requires both text values
  and expected SHA.
- `write_file` binds `filesystem.write_file`, marks mutation true, and restricts
  mode to create/overwrite with a nullable digest.
- `run_command` binds the callable `CommandRunner`. Its schema accepts only an
  argv array, relative cwd, and bounded integer timeout.
- `complete_task` has only a required non-empty `summary` and optional
  `remaining_risks`, rejects additional properties, and is side-effect-free in
  the registry. It requests host acceptance; it cannot set verification fields
  or grant `VERIFIED`.

`ToolExecutionError` is the common expected-failure channel. A handler supplies
an `ErrorKind`, safe message, retryability, and bounded details. The registry
injects `call_id` and `tool_name` and serializes deterministic JSON content. The
provider receives that content on the next tool message; error kinds are not
hidden only in an internal field.

Post-launch command cancellation uses `ToolCancelledError`, which remains a
`KeyboardInterrupt` when the Gate calls `CommandRunner` directly. When a model
tool invokes the same runner, `ToolRegistry` converts cancellation to a paired
result so the loop can record start/mutation evidence before ending `CANCELLED`.

## WorkspaceGuard call path

`WorkspaceGuard.__init__` requires an existing directory and stores
`root.resolve(strict=True)`. For each model path `_resolve`:

1. checks for a non-empty string;
2. rejects native absolute paths and Windows drive/UNC/rooted paths;
3. rejects every `..` component and Windows colon/alternate-data-stream form;
4. joins the relative path to the canonical root;
5. calls `resolve(strict=False)` so existing symlink components are accounted
   for and missing create targets remain representable;
6. proves `resolved.is_relative_to(root)`;
7. rejects the root itself when a file is required.

`resolve_for_read` then checks sensitive basenames and every `.git`/`.veriloop`
component on both lexical and canonical paths. `resolve_for_write` applies the
same metadata-component policy and additionally rejects a final
symlink/reparse point and sensitive files. This is why a path alias cannot bypass
protection and why rejection happens before a temporary file or process is
created.

Traversal uses sorted `Path.iterdir` results. It yields a symlink entry but
checks link/reparse state before testing directories and never recurses through
one. The named cache/build directories are omitted as whole components.

## How read produces the edit digest

`_read_text_file` opens one target in binary mode and reads at most
`max_file_bytes + 1`. It rejects an over-limit result, NUL bytes, and UTF-8 decode
failure. It returns both original bytes and decoded text.

`read_file` hashes the original bytes with SHA-256, not newline-normalized or
re-encoded text. It independently slices at most 500 logical lines and formats
numbered content. Therefore the digest still identifies the complete file even
when the model received only a line window.

## How edit validates and replaces

`edit_file` applies checks in deterministic order:

1. guard and write protection;
2. existing regular text file, size, and UTF-8;
3. current byte SHA equals `expected_sha256`;
4. `old_text` is non-empty and differs from `new_text`;
5. exact search finds one start position and no second, including overlapping
   starts;
6. candidate text is built in memory and encoded/size-checked;
7. `_atomic_replace` performs the disk mutation.

Stale SHA, absent text, ambiguous text, and no-change results never alter the
target. The stale/ambiguous kinds are model-correctable tool failures, not loop
failures.

## Atomic replacement flow

`_atomic_replace` creates a named temporary file in `target.parent`, writes all
bytes, flushes, calls `os.fsync`, closes the handle, and applies original
permission bits for edit/overwrite. Immediately before overwrite's
`os.replace`, it re-reads the target and compares SHA again. Create performs a
quick absence check, then atomically installs without clobbering: Windows uses
non-replacing `os.rename`, while POSIX uses `os.link` and removes the temporary
name. If a competing target appears, it remains intact and the tool returns
`FILE_ALREADY_EXISTS`.

The `finally` path removes any temporary name that was not successfully moved.
Tests inject both `os.replace` failure and a change between temporary write and
replacement, proving original/external bytes win and temporary files are not
left behind. The design assumes the project's single-agent, single-process
scope; it does not claim an adversarial transaction protocol.

`write_file` uses this same helper. Create requires an existing parent and a
missing target and never makes parent directories. Overwrite requires a current
digest and a valid existing text file. Neither mode exposes append, deletion,
rename, or chmod.

## How CommandRunner starts a process

`CommandRunner.run` first asks `CommandPolicy` to validate and copy argv. It then
checks timeout, resolves cwd through `WorkspaceGuard`, and asks the policy to
validate an in-workspace Python script relative to the resolved cwd. Common
Python aliases are replaced with the current `sys.executable`.

Two binary `TemporaryFile` objects are opened for stdout and stderr. `Popen`
receives an argv list, canonical cwd, filtered environment, `stdin=DEVNULL`, the
two output handles, `shell=False`, and the platform process-group option. There
is no command-string parser and no interactive stdin.

The default allowlist is deliberately small:

- read-only Git subcommands `status`, `diff`, `log`, `show`, `ls-files`, and
  `rev-parse`;
- current/common Python executable names with either a relative workspace
  `.py` file or `pytest`, `unittest`, or `compileall` module;
- optional host-injected bare program names or absolute paths that cannot
  override hard denials.

Shell hosts, destructive/privileged commands, downloaders, SSH tools, every
package-manager invocation, Python `-c`/pip, and mutating Git operations are
denied. Git is accepted only as a bare program token, not a supplied path that
could impersonate it. The policy is intentionally not a general shell-language
parser.

## Timeout and process cleanup

`process.wait(timeout=...)` supplies the primary deadline. On timeout:

- POSIX sends `SIGTERM` to the new process group, polls briefly, sends `SIGKILL`
  if the group remains, and finally waits for the direct child;
- Windows starts a new process group, terminates the direct child, waits, kills
  it if necessary, and waits again.

The same cleanup runs before re-raising an interrupt so `AgentLoop` can return
`CANCELLED`. Windows does not use Job Objects in this milestone, so arbitrary
grandchild cleanup is best effort; the direct child cleanup is tested. The POSIX
group test remains executable but is explicitly skipped on Windows because that
host cannot validate POSIX behavior.

## Output truncation

The child writes directly to temporary files, not `PIPE` buffers collected in
Python memory. `_output_preview` gets the total byte count with seek. Output is
decoded with UTF-8 replacement; when that representation would exceed the byte
limit (including for invalid source bytes), it returns a roughly equal head and
tail separated by an exact source-byte omission marker. Stdout and stderr have
independent limits, flags, totals, and previews.

This bounds model-visible content and Python heap use. Temporary disk use during
the allowed runtime is not assigned a fixed byte quota and must not be described
otherwise.

## Child environment filtering

`filtered_subprocess_environment` starts with an empty dict. It copies only the
documented path/home/system/temp/locale/virtual-environment/terminal names. It
then case-insensitively rejects any allowed name containing `KEY`, `TOKEN`,
`SECRET`, `PASSWORD`, `PASSWD`, `AUTH`, `COOKIE`, or `CREDENTIAL`.

The runner never copies all of `os.environ` and never returns the environment.
Tests set only fictional values and make a helper report presence booleans, so
no real value is printed. They prove provider/common secret names are absent
while `PATH` and the current Python runtime still work.

## Why nonzero exit is not a Harness crash

A process with exit code nonzero did start and finish. `CommandRunner` records
that real observation, previews, byte totals, duration, and exit code, then
raises expected `COMMAND_NONZERO_EXIT`. `ToolRegistry` turns it into one failed
`ToolResult`; `AgentLoop` appends it and asks the model again. Timeout follows the
same route with `COMMAND_TIMEOUT` after cleanup. Only an unexpected model/Harness
failure changes the loop to `FAILED`.

The real bug-fix integration trajectory proves this: the first local check exits
1, the model's next request sees its stderr, a SHA-guarded edit changes the real
file, the second check exits 0, and the final state is still
`COMPLETED_UNVERIFIED`.

## Milestone 2 behavior-to-test map

| Behavior | Test location |
| --- | --- |
| Canonical root, nested/missing path, `..`, drive/absolute/prefix/colon rejection | `tests/test_filesystem.py` opening guard tests |
| Missing file, directory-as-file, sensitive names and protected metadata components | `test_read_missing_directory_and_sensitive_path_errors`, `test_read_file_rejects_protected_*`, `test_sensitive_matching_is_component_aware` |
| UTF-8 ranges, SHA, line bounds, NUL/invalid UTF-8/size | `test_read_file_*` |
| Listing order/depth/result truncation/cache skip | `test_list_files_*` |
| Literal/case search, empty query, bounds, binary/large skip | `test_search_text_*` |
| Unique edit, permissions, stale/not-found/ambiguous/no-change/no mutation | `test_edit_file_*` |
| Same-directory replace, replace failure cleanup, final SHA recheck | `test_atomic_replace_*`, `test_edit_rechecks_sha_immediately_before_replace` |
| Create/overwrite modes, no-clobber race, missing SHA/parent, stale/protected/content limits | `test_write_file_*`, `test_write_create_does_not_clobber_target_appearing_at_install` |
| External file/directory symlinks and no traversal | final symlink tests in `tests/test_filesystem.py` |
| Exit zero/nonzero and simultaneous stdout/stderr | opening tests in `tests/test_process.py` |
| Timeout/direct-child stop and POSIX group stop | `test_timeout_ends_direct_child_and_returns_output`, `test_timeout_ends_posix_process_group` |
| Output head/tail, exact totals, and invalid UTF-8 bound | `test_large_stdout_and_stderr_keep_head_tail_and_byte_counts`, `test_non_utf8_output_preview_remains_bounded` |
| cwd valid/escape/absolute/missing/not-directory | `test_cwd_*`, `test_absolute_missing_and_file_cwd_are_rejected` |
| argv schema/types, shell/download/install/Python policy | policy/schema parameterized tests in `tests/test_process.py` |
| read-only versus mutating/path-spoofed Git and dangerous options | `test_command_policy_*git*` |
| timeout maximum, no shell field, `shell=False`, start error | corresponding closing process tests |
| fictional secret isolation and required PATH | `test_child_environment_filters_fictional_secrets_and_keeps_path` |
| Real list/read/nonzero/edit/zero/final bug-fix trajectory | `test_real_bug_fix_trajectory_uses_all_production_boundaries` |
| External change, stale result in next turn, re-read and corrected edit | `test_stale_sha_failure_enters_history_and_model_recovers` |
| Timeout result in next turn without loop failure | `test_command_timeout_is_visible_and_does_not_break_agent_loop` |
| CLI help without key and redacted missing-key error | final CLI tests in `tests/test_agent_tools_integration.py` |
| Milestone 1 provider, registry, history, ordering, retry, budget invariants | unchanged `tests/test_model.py`, `tests/test_tools.py`, `tests/test_agent_loop.py` |

All filesystem mutation tests use pytest `tmp_path`. All subprocess tests use
`sys.executable` and temporary local helper scripts. The default suite makes no
provider or network call and does not run a user shell profile.

## Platform boundary

The Milestone 2 development environment is Windows. Ordinary file and directory
symlink creation is available there, so external symlink rejection and
non-traversal tests execute rather than skip. Windows timeout and direct-child
termination also execute. The POSIX process-group test is retained with an
explicit platform reason and executes only on POSIX.

Path parsing also uses `PureWindowsPath` so drive and UNC forms are rejected on
non-Windows hosts. Windows reparse points are treated as link-like when the
standard-library stat flag exposes them. Platform-specific permission semantics
remain limited to what `os.chmod`/`os.replace` provide.

## Frozen verification configuration

`src/veriloop/verification.py` owns `BaselinePolicy`,
`VerificationCommandSpec`, and `VerificationSpec`. The specification is a
frozen dataclass whose commands and protected globs are tuples, so no model turn
can mutate the in-memory policy.

`load_verification_spec` resolves one workspace-relative TOML path through the
base `WorkspaceGuard`, rejects links, oversized or non-UTF-8 input, unknown
fields, invalid types, unsafe cwd values, excessive timeouts, and commands
rejected by the existing `CommandPolicy`. It checks both decoded TOML text and
the parsed document for the known provider credential before those values reach
policy or tool execution. It then copies validated data into immutable protocol
values. Later edits to the TOML file cannot change the current run.

A missing configuration freezes an empty command set rather than preventing the
agent from running. The requested config path is still retained, protected from
model writes, and represented as missing in the manifest so later creation is
detectable. Empty commands can never grant `VERIFIED`.

## Baseline before model execution

`VerificationGate.run_baseline` implements three policies, and `AgentLoop.run`
invokes it before the first context projection or model request:

- `must_fail` requires every command to start without timeout and at least one
  nonzero exit. An all-zero baseline is `BASELINE_UNEXPECTED_PASS` and ends
  `FAILED`; it is not evidence of a repair.
- `record_only` records zero or nonzero exits and continues, but a start error or
  timeout is still a baseline infrastructure failure.
- `skip` starts no commands and records an explicit skipped baseline.

An empty command set is also represented as a skipped baseline. Baseline
commands are Gate operations, not model tool calls, and therefore do not advance
`mutation_seq`. Baseline evidence is copied into `AgentResult`, JSONL trace,
`result.json`, and the CLI summary.

## Completion request and host-owned final verification

`register_completion_tool` registers `complete_task` in the same production
`ToolRegistry` as every other tool. Registry execution performs ordinary schema
validation and returns the request values, but does not invoke the Gate and does
not set state. `AgentLoop` recognizes the validated request and owns the
completion protocol:

1. ordinary final model text immediately becomes `COMPLETED_UNVERIFIED` and
   never runs final verification;
2. `complete_task` with no frozen commands returns a paired, successful request
   result whose `verified` field is false, then ends `COMPLETED_UNVERIFIED`;
3. with commands configured, the loop enters `VERIFYING` and calls
   `VerificationGate.run_final`;
4. only `VerificationGate.grants_verified` can move the loop to `VERIFIED`.

The Gate runs every frozen command in order through the existing
`CommandRunner`. A grant requires a final-phase result, a non-empty command set,
one result per configured command, every command started, no timeout, every exit
code equal to zero, unchanged protected inputs, and
`result.verified_seq == result.mutation_seq`, with both equal to the loop's
current `mutation_seq`.

The model cannot forge this outcome in text, tool arguments, or tool-result
metadata. `complete_task` rejects extra properties such as `verified`, `state`,
or `exit_code`. A model-initiated `run_command` that runs pytest successfully is
only a tool observation and invalidates prior verification; it does not set
`verified_seq`.

## Mixed completion calls

`complete_task` must be the only call in one assistant response. If it appears
with any other call, AgentLoop executes none of them. It still appends exactly
one `ToolResult` for every original call ID in original order:

- each `complete_task` receives `COMPLETION_MUST_BE_SINGLE_CALL`;
- each other call receives `DEFERRED_REPLAN_REQUIRED`.

The paired failures enter canonical history and the next model request can
replan. This preserves the Milestone 1 pairing invariant without allowing a
write or process side effect to occur beside a completion request.

## Protected manifest and write policy

`VerificationGate.__init__` builds the initial manifest before the model first
receives execution permission. `build_protected_manifest` deterministically
enumerates paths matched by the frozen `protected_globs`, does not follow
link-like entries, skips host metadata/cache directories, and always adds the
configuration path. Each record contains the relative path, existence, file
kind, size, and SHA-256 when meaningful.

`compare_protected_manifests` emits only stable relative-path evidence for
created, deleted, modified, or replaced entries. Final verification compares
both before and after running commands, so it detects a change that existed
before the Gate as well as one caused by a verification command. Any change
wins over green command exits and produces `PROTECTED_FILE_CHANGED`.

`protected_guard_for_spec` also copies the frozen globs and exact config path
into the model-facing `WorkspaceGuard`, so `edit_file` and both `write_file`
modes reject direct protected writes. This is defense in depth, not a substitute
for the final manifest: allowed workspace code launched through `run_command`
can still modify files outside the file-tool API.

`.veriloop/**` is host-owned trace metadata and is excluded from protected
wildcard enumeration, preventing evidence written by the Harness from creating
a false protected-file failure.

## Mutation and verification freshness

AgentLoop starts with `mutation_seq = 0` and `verified_seq = None`.
`ToolRegistry` marks successful mutating tool results with
`invalidates_verification`; expected command failures can carry the same flag
when a process really started. The loop advances `mutation_seq` and clears
`verified_seq` for:

- successful `edit_file`;
- successful `write_file` create or overwrite;
- a model-requested `run_command` that started, whether it exits zero, exits
  nonzero, times out, or is cancelled while running.

Reads, searches, listings, unknown tools, invalid arguments, denied commands,
process start errors, failed file changes, `complete_task`, and Gate-owned
baseline/final commands do not advance the sequence. A successful final Gate
sets `verified_seq` to the exact revision it observed. Consequently stale green
evidence cannot survive a later mutation.

`changed_files` is deliberately narrower than mutation tracking: it is a sorted
summary of paths reported by successful `edit_file` and `write_file` calls. It
does not pretend to enumerate unknown process side effects.

## Repair rounds and repeated failures

A repair round is consumed only when one failed final verification is returned
to the model as a retryable `complete_task` result and the loop continues in
`RECOVERING`. Therefore `max_repair_rounds = N` permits at most `1 + N` final
verification attempts. Once the budget is exhausted, the last evidence is
paired into history, state becomes `VERIFICATION_FAILED`, and no extra model
request occurs.

If repair budget remains but `max_steps` prevents another model request, the
last verification result is paired as non-retryable, unused repair rounds stay
unused, and the run ends `MAX_STEPS` without entering `RECOVERING`.

Before spending another repair round, the loop counts consecutive equal failure
signatures. Reaching `max_same_failure` ends `STALLED`, even if repair budget
remains. A materially different signature resets the consecutive count; a
missing signature cannot accidentally trigger stalling.

`_failure_signature` hashes deterministic JSON containing failure kind, command
argv/cwd, start/timeout/exit/error facts, normalized stdout/stderr tails, and
sorted path-only protected changes. It excludes duration and normalizes
workspace/temp paths, timestamps, process IDs, run IDs, elapsed values, and
random temporary path components. This makes repeated substantive failures
stable without storing full output.

If the model emits plain final text while recovering, it still ends
`COMPLETED_UNVERIFIED`; the last failed verification remains in the result as
evidence.

## Deterministic context projection

`ContextPolicy.project` operates on provider-independent `Message` objects and
never mutates canonical history. System and initial user messages are permanent
anchors. Every later assistant message plus all of its ordered tool results is
one atomic group. The policy removes the oldest removable whole group first,
retains a configurable number of recent groups, and pins the most recent
verification-failure group.

If retained content still exceeds the soft character limit, bounded head/tail
previews are made on a deep copy. Assistant/tool pairing, call IDs, ordering, and
the canonical history remain intact. Malformed or orphaned groups fail before a
provider request rather than being projected into an invalid message sequence.

## Redacted JSONL trace

`TraceWriter` creates one exclusive run directory under
`.veriloop/runs/<run-id>/` and writes one flushed JSON object per event. Each
event contains `schema_version`, strictly increasing `seq` starting at 1, UTC
`timestamp`, `run_id`, `event_type`, current `state`, and a bounded `payload`.
The event vocabulary covers run start/finish/failure/cancellation, baseline,
state transitions, model requests/responses, provider retries, tool receipt and
execution, workspace revision changes, completion, final verification, and
recovery.

Trace payloads contain summaries rather than hidden reasoning or unrestricted
content. Write arguments are represented by lengths and digests; command and
verification streams use bounded previews. Exact known secrets and obvious
Authorization Bearer values are redacted before preview truncation, including
when a secret crosses the old preview boundary. Forbidden environment, header,
provider-client, and reasoning keys are discarded.

The provider API key is host data: it is used for exact in-memory rejection and
redaction but is not passed to `ToolRegistry`, tool handlers, policy, Gate, or
child processes. The CLI freezes a minimal child environment and removes both
sensitive names and allowed variables whose values contain a discovered
provider secret. Redaction reduces accidental disclosure risk; it is not a
general DLP guarantee.

A trace write failure closes tracing but does not fabricate a different agent
state or freshness result.

## `result.json` and optional `patch.diff`

At termination, `TraceWriter.write_artifacts` derives `result.json` from the
host-created `AgentResult`. It includes state, final text, step/tool counts,
mutation and verified sequences, repair usage, baseline and final verification,
protected and changed-file summaries, trace/result/patch paths, duration, model
usage, and bounded error evidence. The model cannot supply its authoritative
state.

Artifacts use exclusive, same-directory atomic installation and never
overwrite an existing target. The same redactor and collection bounds apply to
events, result, and patch data.

`patch.diff` is optional. The writer uses the existing `CommandRunner` for only
`git rev-parse --is-inside-work-tree` and a bounded working-tree `git diff`.
There is no commit, checkout, restore, reset, or history mutation. A non-Git
workspace, unavailable Git, failed or truncated diff, no changes, or an artifact
write failure is recorded as patch metadata instead of producing invented
content or changing the Agent result. Ordinary `git diff -- .` does not include
untracked files.

## Read-only replay

`replay_trace` accepts a run directory or `events.jsonl`, reads it as bounded
UTF-8 JSONL, validates JSON value shapes, schema version, sequential `seq`, one
stable run ID, event/state vocabulary, and event-specific payload facts, then
renders a concise allowlisted view in order.

Replay does not read provider credentials and never constructs a model,
`ToolRegistry`, `CommandRunner`, or `TraceWriter`. It cannot execute a tool,
apply a patch, resume a session, or write workspace files. Corrupt or oversized
input produces a clear parser error and no other behavior.

## CLI evidence and terminal semantics

Normal CLI output gives the terminal state, bounded baseline and final
verification summaries, changed files, trace/result/patch paths, final message,
and error evidence. It never renders hidden reasoning. `--help` and replay are
key-free; configuration and provider setup errors are bounded and redacted.

Only `VERIFIED` returns exit code zero for a normal run. Every other run state,
including `COMPLETED_UNVERIFIED`, returns one. Successful replay returns zero;
argparse, setup, or corrupt-replay errors return two.

## Trust boundary and known limits

VeriLoop is not an OS sandbox and is not described as production-ready or
absolutely safe. `WorkspaceGuard` and `CommandPolicy` constrain Harness-owned
interfaces, but an allowed in-workspace Python script or test suite is repository
code and can attempt external file or network access. Run only in a trusted or
disposable workspace.

On POSIX, timeout cleanup targets a process group. On Windows, direct-child
cleanup is reliable while arbitrary descendants remain best effort. Trace
redaction cannot recognize every possible secret encoding. The project has no
multi-Agent execution, parallel tools, streaming, session resume, automatic
rollback, OS sandbox, or automatic Git commit/push facility.

Milestone 3 is the final core feature milestone. The project is now in feature
freeze; subsequent work is limited to bug fixes, tests, documentation, and
release work.

## Milestone 3 behavior-to-test map

| Behavior | Test location |
| --- | --- |
| Frozen valid, missing, changed-on-disk, invalid, unsafe, and secret-bearing configuration | opening configuration tests in `tests/test_verification.py` |
| `must_fail`, `record_only`, `skip`, timeout/start errors, and frozen command order | `test_baseline_*` in `tests/test_verification.py` |
| Modified/deleted/created/replaced protected paths, automatic config protection, and file-tool write denial | protected-manifest and protected-tool tests in `tests/test_verification.py` |
| Read-only tools, successful/failed file changes, started/non-started/cancelled commands, timeout, and proactive pytest freshness | mutation-sequence tests in `tests/test_verification.py` |
| Final Gate success, command failure, timeout, start error, protected changes, and frozen commands | `test_final_gate_*` in `tests/test_verification.py` |
| Completion without commands, green completion, forged arguments, plain final, and mixed calls | `test_complete_task_*`, `test_mixed_complete_task_*`, and `test_plain_final_*` in `tests/test_verification.py` |
| Repair evidence visibility, exact attempt budget, recovery plain final, repeated signatures, and counter reset | repair and failure-signature tests in `tests/test_verification.py` |
| Atomic context groups, recent/failure retention, deterministic copying, bounded projection, and malformed pairing | `tests/test_context.py` |
| Append-only events, lifecycle, bounded summaries, redaction-before-truncation, retries, trace failure, artifacts, and patch races | `tests/test_trace.py` |
| Strict, bounded, side-effect-free trace loading and formatting | `tests/test_replay.py` |
| Key-free help/replay, config ordering, protected composition, secret boundary, CLI evidence, and exit codes | `tests/test_cli.py` |
| Production Red-to-Green Gate success | `test_red_green_project_is_verified_by_the_production_gate` in `tests/test_agent_verification_integration.py` |
| Failed verification followed by repair and VERIFIED | `test_failed_verification_evidence_drives_repair_to_verified` in the same integration file |
| Persistent failure stopping at the exact budget | `test_persistent_failure_stops_at_the_exact_repair_budget` in the same integration file |
| Command-side protected test tampering cannot verify | `test_started_command_tampering_with_protected_test_cannot_verify` in the same integration file |
| Plain final cannot run the configured Gate | `test_plain_final_claim_never_runs_the_configured_gate` in the same integration file |

## Recommended code-reading order

1. `src/veriloop/protocol.py`: follow provider-independent messages, states,
   verification evidence, and the authoritative `AgentResult`.
2. `src/veriloop/model.py::OpenAICompatibleModel`: confirm provider conversion
   and retry end before data enters the loop.
3. `src/veriloop/tools.py::ToolRegistry` and production registrations: trace
   schema validation, call/result pairing, mutation flags, and `complete_task`.
4. `src/veriloop/filesystem.py::WorkspaceGuard`: understand lexical/canonical
   containment, metadata protection, and frozen protected-write checks.
5. `src/veriloop/filesystem.py::read_file`, `edit_file`, `write_file`, and
   `_atomic_replace`: trace raw-byte SHA and no-clobber mutation.
6. `src/veriloop/process.py::CommandPolicy`, `CommandRunner`, and
   `host_child_environment`: trace allowlisting, child lifecycle, bounded output,
   and provider-secret removal.
7. `src/veriloop/verification.py::load_verification_spec`: follow TOML bytes
   into validated immutable values before the model exists.
8. `src/veriloop/verification.py::build_protected_manifest`,
   `compare_protected_manifests`, and `protected_guard_for_spec`: understand the
   two protected-input defenses.
9. `src/veriloop/verification.py::VerificationGate`: compare baseline policy,
   final evidence, and every `grants_verified` conjunct.
10. `src/veriloop/agent.py::AgentLoop.run`: follow baseline, projected requests,
    mixed-call handling, mutation invalidation, completion, repair, stall, and
    terminal artifact truth without direct file or subprocess access.
11. `src/veriloop/context.py::ContextPolicy`: trace atomic group partitioning,
    deterministic removal, failure retention, and projection-only truncation.
12. `src/veriloop/trace.py::Redactor` and `TraceWriter.emit`: inspect
    redaction-before-truncation and the append-only event envelope.
13. `src/veriloop/trace.py::TraceWriter.write_artifacts` and `_result_payload`:
    trace host result serialization and optional read-only Git diff generation.
14. `src/veriloop/trace.py::load_trace_events`, `format_trace_replay`, and
    `replay_trace`: verify the bounded read-only evidence path.
15. `src/veriloop/cli.py::main`: confirm configuration precedes the model,
    protected components share one frozen spec, secrets stop at host boundaries,
    and only `VERIFIED` exits zero.

## Milestone 1 questions that remain mandatory

1. Why is `ModelClient` a protocol rather than a provider base class?
2. Why are provider objects translated before entering `AgentLoop`?
3. How does `ToolCall.id` become exactly one `ToolResult.call_id`?
4. Why do recoverable tool errors enter history instead of ending the loop?
5. Where is same-turn tool order guaranteed?
6. Why does `max_steps` count model calls and how is N+1 prevented?
7. Which provider failures retry and where is the three-request bound enforced?
8. Why is Python `bool` excluded from numeric schema types?
9. Why is final text not a success/verification claim?
10. How does `ScriptedModel` prove provider independence and offline behavior?

## 项目作者后续统一学习时必须回答的 Milestone 2 问题

1. Why must path containment compare resolved components instead of string
   prefixes, and which attacks do the prefix and symlink tests represent?
2. Why is `read_file` SHA computed from raw bytes even when only a line window is
   returned?
3. In what exact order does `edit_file` reject stale, absent, ambiguous, and
   unchanged edits, and why can none mutate the file?
4. Why is the target SHA checked a second time after writing the temporary file,
   and what does the injected race test prove?
5. Why do stdout and stderr go to separate temporary files, and how does the
   preview preserve both the beginning and the usually-important tail?
6. What is the difference between `COMMAND_NONZERO_EXIT`, `COMMAND_TIMEOUT`,
   `COMMAND_START_ERROR`, and an unexpected Harness failure?
7. How does the child environment allowlist prevent `OPENAI_API_KEY` inheritance
   while keeping the current Python runtime usable?
8. What cleanup guarantee is provided on POSIX, what weaker guarantee is
   provided on Windows, and why must the documentation state the difference?
9. Why does relative cwd enforcement not isolate code executed by Python or
   pytest from outside files or the network?
10. How do the two full integration trajectories prove that real tool failure is
    visible to the next model turn without adding any Milestone 3 verification?

## 项目作者必须掌握的 Milestone 3 问题

1. `VerificationSpec` 在 CLI 的什么位置加载，哪些不可变类型使磁盘配置在
   run 中无法漂移，为什么这必须发生在第一次模型请求之前？
2. `must_fail`、`record_only` 和 `skip` 对非零退出、全零退出、timeout 与
   start error 分别如何解释，为什么 timeout 不能作为有效 Red 证据？
3. `VerificationGate.grants_verified` 的每个合取条件是什么，缺失任一条件时
   为什么都不能用 `VERIFIED` 表示结果？
4. 为什么 `complete_task` 必须先经过普通 `ToolRegistry` schema 验证，却又
   不能由它的 handler 直接授予状态？
5. 同轮出现 `complete_task` 与其他调用时，为何所有调用都不执行但仍必须按
   原顺序为每个 call ID 产生恰好一个结果？
6. 哪些普通工具结果会推进 `mutation_seq`，为什么已启动但 nonzero 或 timeout
   的 `run_command` 也必须使旧验证失效？
7. 为什么 Gate 自己的 baseline/final command 不推进 `mutation_seq`，以及
   `verified_seq == mutation_seq` 如何证明 green evidence 的 freshness？
8. protected write deny 与 final manifest 各自阻止什么攻击面，为什么只有文件
   工具层面的 deny 仍不足以保护 tests 和配置？
9. `max_repair_rounds = N` 为什么对应最多 `1 + N` 次 final verification，在哪个
   精确时刻 repair round 才算被消耗？
10. failure signature 包含和排除哪些字段，如何标准化易变路径、时间和输出，
    为什么 stall 只统计连续相同签名？
11. ContextPolicy 如何定义一个不可拆分的 assistant/tool group，哪些 anchor 和
    failure group 必须保留，为什么 canonical history 从不被裁剪？
12. Trace 为什么必须先做 known-secret/Bearer redaction 再做 preview truncation，
    哪些 payload 可以记录而哪些 provider、环境和推理数据必须丢弃？
13. `result.json` 的 state 从哪里产生，为什么模型 summary、工具参数或 metadata
    都不能成为 authoritative result？
14. `patch.diff` 使用哪些只读 Git 命令，哪些失败会使其 unavailable，为什么这
    不影响内存 AgentResult 的真假？
15. replay 对 JSONL 做哪些协议和边界验证，哪些组件明确不会构造，什么测试
    证明它不会重新执行任何副作用？
