# VeriLoop

VeriLoop is a small, local coding-agent harness whose model/tool loop remains
owned by the project and easy to test. Milestone 2 connects the existing
provider-independent `AgentLoop` to guarded local file tools and an allowlisted
command runner. It does not use an agent framework or agent SDK.

## Current capabilities

The production registry exposes six tools:

- `list_files`: deterministic, depth/result-bounded workspace listing;
- `read_file`: bounded UTF-8 line reads with a SHA-256 digest;
- `search_text`: deterministic literal search with bounded previews/results;
- `edit_file`: one exact, unique replacement guarded by the prior SHA-256;
- `write_file`: atomic create or SHA-guarded overwrite;
- `run_command`: argv-only execution through `CommandPolicy`, with timeout,
  filtered environment, separate stdout/stderr, and bounded previews.

Every tool call still passes through `ToolRegistry.execute`. Expected path,
file, policy, exit-code, and timeout failures become structured `ToolResult`
values in history, so the model can inspect the real result and make another
decision. A normal final model response is always
`COMPLETED_UNVERIFIED`.

## Requirements and installation

- Python 3.11 or newer
- Runtime dependency: `openai`
- Test dependency: `pytest`

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The default test suite is deterministic and offline. It uses temporary
workspaces and local helper processes; it does not require a provider key.

```powershell
python -m pytest -q
```

## CLI

Set `OPENAI_API_KEY` only in the environment and choose a model with
`OPENAI_MODEL` or `--model`. `OPENAI_BASE_URL` or `--base-url` may select an
OpenAI-compatible endpoint. The workspace defaults to the current directory.

```powershell
veriloop "Fix the boundary bug" --model your-model-name --workspace D:\work\repo
```

`--help` does not require an API key. The CLI binds a canonical
`WorkspaceGuard`, `CommandPolicy`, `CommandRunner`, production `ToolRegistry`,
and the existing `AgentLoop`. It neither runs tests automatically nor announces
independent verification. Its successful terminal state remains
`COMPLETED_UNVERIFIED`.

API keys are read only from environment variables. File and process tools do
not print them, store them, or pass them into child processes.

## Trust boundary and limits

`WorkspaceGuard` constrains VeriLoop's own file tools. It rejects absolute and
escaping paths, canonical symlink escapes, sensitive reads, and protected
writes. `CommandPolicy` blocks obvious destructive, shell-host, downloader,
installer, and mutating Git command shapes. Commands always use `shell=False`,
an in-workspace `cwd`, a timeout, and a newly constructed child environment.

These controls are not an OS sandbox. An in-workspace Python script, pytest, or
other explicitly allowed program is repository code and can still attempt to
read outside the workspace or access the network. VeriLoop does not use a
container, VM, namespace, SELinux, Landlock, or Windows Job Object. POSIX timeout
cleanup targets a process group; Windows reliably targets the direct child and
provides only best-effort behavior for descendants. Run it only in a trusted or
disposable local workspace.

Milestone 3 is not implemented. It will cover the independent Verification
Gate, an explicit completion request, mutation/verification freshness, repair
rounds, JSONL trace data, and replay/debugging. A model choosing to run pytest
and observing exit code zero is evidence in conversation history, not Harness
verification.

See [SPEC.md](docs/SPEC.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[HANDOFF.md](docs/HANDOFF.md) for the exact behavior and implementation guide.
