# VeriLoop Milestone 2 implementation handoff

This guide is for learning and defending the implementation. Milestone 1 was
independently reviewed and remains intact; Milestone 2 adds real local tools
without changing who owns the loop or how termination works.

## Current state

The harness now has provider-independent messages, bounded provider retry,
validated tool calling, synchronous loop control, five guarded file tools, one
allowlisted process tool, production CLI composition, and deterministic offline
tests. It has no independent Verification Gate. A final model message still
means only `COMPLETED_UNVERIFIED`.

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

## End-to-end Milestone 2 composition

`cli.main` parses the task, model configuration, workspace, and model-step
limit. It reads `OPENAI_API_KEY` only after argument parsing, so `--help` works
without a key. It then constructs:

1. canonical `WorkspaceGuard`;
2. `CommandPolicy`;
3. `CommandRunner` bound to that guard and policy;
4. `ToolRegistry` populated by `register_workspace_tools`;
5. `OpenAICompatibleModel`;
6. the unchanged `AgentLoop`.

The complete runtime path is:

```text
CLI
 -> AgentLoop
 -> ModelClient
 -> ToolRegistry
 -> WorkspaceGuard / CommandPolicy
 -> File Tool / CommandRunner
 -> ToolResult
 -> history
 -> ModelClient
```

`build_workspace_tools` provides the same production registry construction for
tests or embedding. Neither registration path creates a dependency framework.

## Every production tool entry

All schemas and bindings live in `src/veriloop/tools.py`.

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

`ToolExecutionError` is the common expected-failure channel. A handler supplies
an `ErrorKind`, safe message, retryability, and bounded details. The registry
injects `call_id` and `tool_name` and serializes deterministic JSON content. The
provider receives that content on the next tool message; error kinds are not
hidden only in an internal field.

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

## Behavior-to-test map

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

## No Verification Gate yet

No completion tool, baseline/final verification, mutation sequence, freshness,
repair round, trace writer, replay, or rollback exists. Running pytest through
`run_command` can help the model decide what to do, but the Harness has not
independently selected or enforced acceptance. No code path can produce
`VERIFIED`.

## Recommended code-reading order

1. `src/veriloop/protocol.py`: immutable values and M2 `ErrorKind` additions.
2. `src/veriloop/tools.py`: schema validation, expected failure conversion, and
   all production registrations.
3. `src/veriloop/filesystem.py::WorkspaceGuard`: lexical/canonical containment
   and protected paths.
4. `src/veriloop/filesystem.py::read_file`, `edit_file`, and `_atomic_replace`:
   trace byte SHA through deterministic mutation.
5. `src/veriloop/filesystem.py::list_files` and `search_text`: trace sorted,
   bounded, non-symlink traversal.
6. `src/veriloop/process.py::CommandPolicy`: understand every allow/deny shape.
7. `src/veriloop/process.py::CommandRunner`, `_terminate_process`, and
   `_output_preview`: trace child lifecycle and bounded return data.
8. `src/veriloop/agent.py::AgentLoop.run`: verify it remains ignorant of every
   concrete local tool and still only returns `COMPLETED_UNVERIFIED` on text.

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
