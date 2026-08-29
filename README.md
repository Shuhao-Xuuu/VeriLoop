# VeriLoop

VeriLoop is a small local coding-agent harness built to make the core model/tool
loop easy to test and explain. The repository currently contains Milestone 1
only: provider-independent messages, a non-streaming OpenAI-compatible model
adapter, a tool registry, a synchronous agent loop, and fully offline tests.

## Requirements and installation

- Python 3.11 or newer
- Runtime dependency: `openai`
- Test dependency: `pytest`

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the deterministic test suite without an API key or network access:

```powershell
python -m pytest -q
```

## Minimal CLI

The CLI is the future production entry point. It currently has no production
tools; it can only let a configured model produce text or receive structured
unknown-tool errors.

Set `OPENAI_API_KEY` in the environment, then supply a model either through
`OPENAI_MODEL` or `--model`. `OPENAI_BASE_URL` or `--base-url` may select an
OpenAI-compatible endpoint.

```powershell
veriloop "Explain the task" --model your-model-name
```

API keys are read only from the environment. They must not be stored in the
repository.

## Current limits

Milestone 1 does not include file or search tools, command execution,
workspace/path/command safety, context compression, a Verification Gate,
mutation/verification tracking, trace/replay, streaming, or multiple agents.
These belong to later milestones. A normal final model response therefore ends
as `COMPLETED_UNVERIFIED`, not as a verified success.

See [SPEC.md](docs/SPEC.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[HANDOFF.md](docs/HANDOFF.md) for the exact behavior and implementation guide.
