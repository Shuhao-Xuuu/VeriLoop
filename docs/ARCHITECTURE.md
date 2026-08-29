# VeriLoop Milestone 2 architecture

## Complete call chain

```text
CLI
 |
 v
AgentLoop <-------------------------------+
 |                                        |
 v                                        |
ModelClient                               |
 |                                        |
 v                                        |
OpenAI-compatible provider                |
                                          |
ToolCall                                  |
 |                                        |
 v                                        |
ToolRegistry                              |
 |                                        |
 +--> WorkspaceGuard --> file tool -------+
 |                                        | ToolResult -> history
 +--> CommandPolicy --> CommandRunner -----+
```

In sequence, the runtime path is:

```text
CLI -> AgentLoop -> ModelClient -> ToolRegistry
    -> WorkspaceGuard / CommandPolicy
    -> File Tool / CommandRunner
    -> ToolResult -> history -> ModelClient
```

The model proposes structured calls; it never executes them. `AgentLoop`
coordinates turns but has no direct filesystem or subprocess access. Every call
still enters `ToolRegistry.execute`, which validates arguments, invokes the
bound handler, and creates exactly one result with the unchanged call ID.

## Dependency direction

```text
protocol.py  <- immutable provider-independent values and error kinds
     ^
     |--- model.py       <- provider conversion and retry
     |--- tools.py       <- schemas, validation, failure/result boundary,
     |                      production registration
     |       ^
     |       |--- filesystem.py <- WorkspaceGuard and five file handlers
     |       +--- process.py    <- CommandPolicy and CommandRunner
     |
     +--- agent.py       <- ModelClient + ToolRegistry contracts only
             ^
             |
           cli.py        <- composition root
```

`filesystem.py` and `process.py` do not depend on `ModelClient` or provider SDK
objects. `ToolRegistry` does not depend on the CLI. The registration functions
bind a concrete guard/runner with small callables; there is no general dependency
injection system.

## File trust boundary

`WorkspaceGuard` canonicalizes its root once. For every model path it rejects
native and Windows absolute/drive/UNC forms, every `..` component, and Windows
colon/alternate-data-stream components. It joins the remaining relative path to
the root, calls `resolve(strict=False)`, then proves component containment with
`Path.is_relative_to`. This handles sibling-prefix deception and symlink
directories without relying on string prefixes.

Protection checks apply to complete relative components or basenames/globs.
Sensitive file contents cannot be read or searched. `.git` and `.veriloop`
components cannot be written. Traversal is explicit and sorted; excluded
directories and symlink/reparse-point directories are never descended into.

All file reads take at most `max_file_bytes + 1` bytes, then enforce size, NUL,
and UTF-8 rules. `read_file` hashes the original bytes before any text rendering,
so the digest identifies exactly what `edit_file` or overwrite observed.

## Deterministic mutation path

An edit follows this sequence:

```text
canonical/protection checks
 -> bounded UTF-8 read
 -> current SHA == expected SHA
 -> old text non-empty and different from new text
 -> exactly one exact occurrence (overlaps count)
 -> candidate built and size-checked in memory
 -> temporary file created in target directory
 -> write -> flush -> fsync -> preserve permission bits -> close
 -> re-read target and re-check SHA
 -> os.replace(temp, target)
 -> cleanup any remaining temporary path
```

Create uses a second absence check immediately before replacement. The project
assumes one agent and one process; this is a deterministic stale-write guard, not
an adversarial concurrent transaction manager. Failure before `os.replace`
leaves the original bytes unchanged. The API has no delete, rename, append, or
chmod operation.

## Command trust boundary

`CommandPolicy` first validates that argv is a non-empty list of non-empty
strings. Hard denials cover shell hosts, destructive/privileged programs,
downloaders, remote shells, every package-manager invocation, and mutating Git.
Git must be invoked by a bare program name so a supplied path cannot impersonate
it. Python acceptance is deliberately narrow: a relative workspace `.py` file
or one of three modules, never `-c` or pip. Host-provided bare program names or
absolute program paths cannot override hard-denied program names.

`CommandRunner` separately resolves `cwd` through `WorkspaceGuard`, requires an
existing directory, and revalidates any Python script relative to that directory.
It then launches an argv list with `Popen`, `shell=False`, and no interactive
stdin. `AgentLoop` has no subprocess branch; nonzero exit and timeout return
through the same registry/history route as other tool failures.

## Output, timeout, and environment

Stdout and stderr are separate binary `TemporaryFile` objects. Child output is
therefore spooled outside Python heap rather than collected with `PIPE` and
`communicate`. After the process is reaped, each file yields an exact byte count
and either complete output or a head/omission-marker/tail preview whose UTF-8
encoded model-visible form remains bounded even for invalid source bytes.
Temporary files are closed and removed at the end of the handler. Temporary disk
usage itself is not assigned a fixed quota in Milestone 2.

On POSIX, `start_new_session=True` creates a process group. Timeout sends
`SIGTERM` to the group, waits briefly, sends `SIGKILL` if it remains, and finally
waits for the direct child. On Windows, `CREATE_NEW_PROCESS_GROUP` is used, then
the direct child is terminated and killed if necessary. Windows descendants are
only best effort because Milestone 2 does not add a Job Object implementation.

The child environment starts empty and copies only selected runtime variables
such as `PATH`, home/system/temp/locale values, virtual-environment markers, and
`PYTHONPATH`. A second name filter removes common credential keywords. The
runner never copies `os.environ` wholesale and never returns the environment in
tool output.

## Agent state and honest boundary

The Milestone 1 state machine is unchanged:

```text
THINKING -> EXECUTING -> append TOOL result -> THINKING
    |
    +-> no calls -> COMPLETED_UNVERIFIED
    +-> model failure -> FAILED
    +-> no remaining model step -> MAX_STEPS
    +-> interrupt -> CANCELLED
```

There is no `VERIFIED` transition. A successful `run_command` containing pytest
is only a model-observed tool result. Independent task acceptance belongs to
Milestone 3.

`WorkspaceGuard` applies to the built-in file functions. `CommandPolicy` reduces
obvious accidental command risk. Neither `cwd` nor argv filtering is an OS
sandbox: allowed Python/tests are repository code and can attempt outside file
or network access. Run only in a trusted or disposable workspace.
