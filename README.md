# VeriLoop

VeriLoop is a small local coding-agent harness with a complete
model → tool → verification loop. The project owns its provider conversion,
history, tool orchestration, retries, termination, verification, trace, and
replay code; it does not use an agent framework or agent SDK.

Milestones 1 and 2 established the provider-independent loop and guarded local
file/process tools. Milestone 3 adds a host-controlled Verification Gate,
bounded repair, deterministic context projection, JSONL evidence, result
artifacts, and read-only replay.

## How verification works

The production registry exposes seven tools:

- `list_files`, `read_file`, and `search_text` for bounded inspection;
- `edit_file` and `write_file` for SHA-guarded atomic text mutation;
- `run_command` for argv-only local execution through `CommandPolicy`;
- `complete_task` for requesting host acceptance.

`complete_task` does not declare success. At run startup, the CLI loads and
freezes a `VerificationSpec`, protects user-configured inputs and host-derived
verifier control inputs, records their manifest, and runs the configured
baseline before the first model request. A standalone `complete_task` request
causes the host Gate to run the frozen final commands. `VERIFIED` is possible
only when every command starts, avoids timeout, exits zero, protected inputs
are unchanged, and `verified_seq == mutation_seq`.

Successful file mutations and every ordinary command that actually starts
advance `mutation_seq` and invalidate prior verification. A failed final check
is returned to the model as a paired `ToolResult`; repair is limited by
`max_repair_rounds`, and repeated equivalent failures can stop as `STALLED`.
A plain final model message—even one containing the word `VERIFIED`—is always
`COMPLETED_UNVERIFIED`. The same is true when no verification commands are
configured.

Every executed model tool still goes through `ToolRegistry.execute`.
If `complete_task` is mixed with another call in one response, none of those
calls execute and every call ID receives a structured failure result.

## Requirements and installation

- Python 3.11 or newer
- Runtime dependency: `openai`
- Test dependency: `pytest`

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The default suite is deterministic and offline. It uses temporary workspaces,
`ScriptedModel`, and local bounded subprocesses; it does not require a provider
key or make a real provider/network call.

```powershell
python -m pytest -q
```

## Verification configuration

The default config path is the workspace-relative `.veriloop.toml`:

```toml
[verification]
baseline_policy = "record_only"
max_repair_rounds = 2
max_same_failure = 2
# User-declared inputs; Python/pytest verifier controls are added automatically.
protected_globs = ["tests/**"]

[[verification.commands]]
argv = ["python", "-m", "pytest", "-q"]
cwd = "."
timeout_seconds = 90
```

Baseline policies are:

- `record_only`: run and record the baseline without requiring red or green;
- `must_fail`: require at least one configured command to fail, which is useful
  for a known Red→Green task;
- `skip`: record an explicit skipped baseline and run no baseline command.

The config file itself is automatically protected. Configured commands, cwd,
timeouts, repair limits, and protected globs are validated and frozen before the
model is created. Configured Python commands also add workspace-controlled
module shadows and interpreter startup hooks to that frozen boundary. Pytest
commands additionally add its workspace configuration, `conftest.py`, and
plugin-discovery metadata candidates. This includes candidates that do not yet
exist: file tools deny their creation, and the final manifest detects creation
or change by an allowed process. Missing config or an empty command list is
valid, but it cannot produce `VERIFIED`.

## CLI

Set `OPENAI_API_KEY` only in the invoking environment. Select a model with
`OPENAI_MODEL` or `--model`; `OPENAI_BASE_URL` or `--base-url` is optional.

```powershell
veriloop run "Fix the boundary bug" --workspace D:\work\repo
```

The legacy equivalent remains supported:

```powershell
veriloop "Fix the boundary bug" --model your-model-name --workspace D:\work\repo
```

Useful options are `--config` (default `.veriloop.toml`) and `--max-steps`
(default 12). `--help` does not require an API key. Only `VERIFIED` exits 0;
all other normal run terminal states exit 1. Argument, configuration, and setup
errors exit 2.

## Evidence and replay

Each run writes host-owned evidence under
`.veriloop/runs/<run-id>/`:

- `events.jsonl`: flushed, ordered lifecycle events with increasing `seq`;
- `result.json`: final state, baseline/final evidence, freshness, protected
  changes, repair count, changed files, usage, and artifact paths;
- `patch.diff`: an optional bounded, redacted unstaged Git diff when available.

Replay needs neither an API key nor a model:

```powershell
veriloop replay D:\work\repo\.veriloop\runs\<run-id>
```

Replay validates and formats the saved `events.jsonl`. It does not restore a
session, call a model, invoke `ToolRegistry`, execute a command, apply a patch,
or modify the workspace.

Known provider secrets and obvious Authorization Bearer values are redacted
before trace previews are truncated. This is deterministic best-effort
redaction, not a general data-loss-prevention system.

## Trust boundary and limits

`WorkspaceGuard` constrains VeriLoop's own file tools. `CommandPolicy` blocks
obvious destructive, shell-host, downloader, installer, and mutating Git command
shapes. Commands use `shell=False`, an in-workspace cwd, a timeout, and a frozen
filtered child environment that excludes provider secrets.

These controls are not an OS sandbox. An allowed workspace Python script,
pytest, or other program can still attempt to read outside the workspace or
access the network. VeriLoop does not provide a container, VM, namespace,
SELinux/Landlock boundary, or Windows Job Object. POSIX timeout cleanup targets
the process group; Windows reliably targets the direct child and handles
descendants only on a best-effort basis.

There is no multi-agent execution, streaming, parallel tool execution, session
resume, automatic rollback, or automatic Git commit/push. VeriLoop is not
claimed to be production-ready, absolutely secure, or 100% successful. Run it
only in a trusted or disposable local workspace.

See [SPEC.md](docs/SPEC.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[HANDOFF.md](docs/HANDOFF.md) for the exact contract and implementation guide.
