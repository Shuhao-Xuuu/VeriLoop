# VeriLoop milestone specification

## Milestone status

Milestone 1 is complete and passed independent review. It established the
provider-independent protocol, `ModelClient`, non-streaming OpenAI-compatible
adapter, bounded provider retry, `ToolRegistry`, `ScriptedModel`, synchronous
`AgentLoop`, exact call/result pairing, error feedback, step limits, and
`COMPLETED_UNVERIFIED` termination.

Milestone 2 is complete. It adds:

- `WorkspaceGuard` and canonical path containment;
- bounded UTF-8 `list_files`, `read_file`, and `search_text` tools;
- SHA-guarded `edit_file` and `write_file` with atomic replacement;
- `CommandPolicy` and argv-only `run_command`;
- in-workspace `cwd`, process timeout/cleanup, bounded stdout/stderr previews,
  and child-environment filtering;
- a production tool-registration entry point and CLI composition;
- fully offline unit and `ScriptedModel` integration tests against real
  temporary workspaces and local processes.

Milestone 3 is not implemented. Its future scope is the Verification Gate,
explicit completion request, mutation/verification tracking, verification
freshness, repair rounds, JSONL trace data, and replay/debugging.

There is currently no `VERIFIED` state. A model may call pytest and observe exit
code zero, but that is not independent Harness verification. Normal completion
remains `COMPLETED_UNVERIFIED`.

## Preserved loop contract

Provider SDK values do not enter `AgentLoop`. `ModelClient` never executes a
tool. `AgentLoop` never calls a concrete file or process handler and knows no
path, SHA, command, or subprocess rule. It executes each returned `ToolCall`
serially and only through `ToolRegistry.execute`.

Each call produces exactly one `ToolResult` with the unchanged call ID. Expected
tool failures are represented by `ToolExecutionError`, which the registry turns
into a failed result whose deterministic JSON content includes the error kind,
message, retryability, and bounded details. Thus both internal tests and a real
provider can see errors on the next turn. Tool failure does not terminate the
loop. `max_steps` still counts model calls and prevents call N+1.

## Workspace and path rules

`WorkspaceGuard` requires an existing directory and stores its canonical
resolved root. Model paths must be non-empty relative paths; native absolute
paths, Windows drive-relative/absolute paths, UNC paths, and canonical paths
outside the root are rejected. Every `..` component is rejected even when later
components would return inside the workspace. Windows colon/alternate-data-
stream components are also rejected. Containment uses path components and
`Path.is_relative_to`, never string prefix comparison.

Read and traversal paths are resolved before use. Directory traversal does not
follow symlink/reparse-point directories. A symlink that resolves outside the
workspace cannot be read or used as a write route. Edit and overwrite reject a
final symlink. File tools protect the following basenames case-insensitively:

- `.env` and `.env.*`;
- `*.pem` and `*.key`;
- `id_rsa` and `id_ed25519`;
- `credentials.json` and `serviceAccountKey.json`.

Writes additionally reject any `.git` or `.veriloop` path component. Matching
uses complete components/globs, so ordinary names containing `env` or `key` do
not match by substring.

File content is limited to ordinary UTF-8 text with no NUL byte and at most
1 MiB by default. Directories, non-regular files, invalid UTF-8, binary/NUL
content, and oversized data produce structured failures without traceback.

## File tools

### `list_files`

Inputs are `path` (default `.`), `max_depth` (default 3, range 1..20), and
`max_results` (default 300, range 1..1000). Results use workspace-relative POSIX
paths, contain file/directory/symlink types and file sizes, and are sorted
deterministically. Traversal skips `.git`, `.veriloop`, Python/tool caches,
virtual environments, `node_modules`, `dist`, and `build`. It never follows a
symlink directory and reports `truncated=true` when another result exceeds the
bound.

### `read_file`

Inputs are `path`, `start_line` (default 1), and `end_line` (default 400).
Line numbering starts at 1, ranges must be ordered, and one call may request at
most 500 lines. The result contains requested and actual ranges, total line
count, SHA-256 of the exact original bytes, numbered text, and a truncation flag.

### `search_text`

Inputs are non-empty literal `query`, `path` (default `.`), `case_sensitive`
(default false), and `max_results` (default 50, range 1..500). Search is standard
library only, deterministic, and recursive. It skips the same directories,
sensitive files, symlinks, binary/non-UTF-8 files, and oversized files. Each
match has a relative path, 1-based line number, and bounded line preview.

### `edit_file`

Inputs are `path`, non-empty `old_text`, `new_text`, and `expected_sha256`.
The existing regular text file must have the expected digest. Identical old/new
text is `NO_CHANGE`; zero exact occurrences is `EDIT_TEXT_NOT_FOUND`; multiple,
including overlapping, occurrences are `EDIT_TEXT_AMBIGUOUS`. No failure changes
the file. Exactly one occurrence is replaced in memory, encoded/size-checked,
written to a temporary file in the same directory, flushed and fsynced, given
the original permission bits, SHA-checked again, then installed with
`os.replace`. Success returns before/after digests, one replacement, line counts,
and a bounded diff preview.

### `write_file`

Inputs are `path`, `content`, `mode` (`create` or `overwrite`), and optional
nullable `expected_sha256`. Create requires a missing target, an existing parent
directory, and no expected digest. It never creates parent directories or
overwrites an existing target. Overwrite requires an existing regular text file
and matching digest. Both modes use the same-directory temporary file and atomic
replacement flow. Append, delete, rename, chmod, and arbitrary directory
creation are not exposed.

## Command tool

`run_command` accepts only `argv: list[str]`, relative `cwd` (default `.`), and
integer `timeout_seconds` (default 60, maximum 120). The registry validates the
schema and `CommandPolicy` validates the command shape. No command string or
shell option exists; `CommandRunner` always calls `subprocess.Popen` with
`shell=False`, `stdin=DEVNULL`, and a canonical in-workspace directory.

Shell hosts, privilege/destructive tools, downloaders, remote shells, package
managers, and mutating Git subcommands are denied. Git must use its bare program
name and is limited to `status`, `diff`, `log`, `show`, `ls-files`, and
`rev-parse`, with selected side-effect options denied. Python is limited to the
current interpreter or common aliases, an in-workspace relative `.py` script,
or modules `pytest`, `unittest`, and `compileall`; `-c` and `-m pip` are denied.
A host may inject additional bare program names or absolute program paths, but
hard denials still apply.

Stdout and stderr go to separate temporary files rather than Python pipes, so
they do not accumulate without bound in Python memory. Each model-visible
preview is about 16 KiB, remains bounded after UTF-8 replacement decoding, and
preserves both head and tail with an omitted-byte marker. The result includes
exit code, duration, both previews, truncation flags, and exact total byte
counts. Temporary output files are removed after the tool.

Exit zero is a successful tool result. Nonzero exit is
`COMMAND_NONZERO_EXIT`; timeout is `COMMAND_TIMEOUT`; process creation failure is
`COMMAND_START_ERROR`. All are tool observations returned to the model rather
than AgentLoop crashes. POSIX starts a new session and terminates/kills the
process group on timeout. Windows creates a new process group but guarantees
cleanup only for the direct child; descendant cleanup is best effort.

The child environment is constructed from an explicit allowlist. Environment
names containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `AUTH`,
`COOKIE`, or `CREDENTIAL` are filtered again. Provider keys, including
`OPENAI_API_KEY`, are not inherited.

## Security boundary and excluded work

Workspace containment governs VeriLoop file handlers, not code executed by an
allowed process. `cwd` containment is not filesystem isolation. A repository
Python script or test can still try to access outside paths or the network. The
project has no container, VM, namespace, SELinux, Landlock, or Windows Job
Object enforcement and should run only in trusted or disposable workspaces.

Milestone 2 does not contain a Verification Gate, completion tool, baseline or
final verification, mutation/verification sequence, repair loop, protected
acceptance tests, persistent trace/event writer, replay, rollback, session
resume, Git mutation tool, context compression, repo map, AST editing, patch
parser, streaming, parallel tools, UI, multiple agents, MCP, plugins, or
dependency download/install behavior.
