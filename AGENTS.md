# Repository instructions

- Implement VeriLoop milestone by milestone. Do not add later-milestone code,
  placeholders, or speculative APIs early.
- Do not use agent frameworks or agent SDKs. Internal messages, provider
  conversion, tool orchestration, history, termination, and retries remain
  VeriLoop-owned code.
- Prefer the smallest implementation that is deterministic, testable, and easy
  to explain.
- Read API keys only from environment variables. Never place credentials in
  source, tests, configuration, logs, documentation, commits, or fixtures.
- Run the full offline test suite after changes. Tests must not call a real
  provider, network, shell, or user workspace.
- Review `git status`, the complete diff, and staged content before completing a
  milestone.
- After acceptance passes, a milestone may be committed and normally pushed to
  the existing configured remote.
- Never force-push, rewrite history, use destructive Git cleanup/reset commands,
  change remotes, or alter global Git/credential configuration.
- Milestone 2 path, SHA, atomic-write, command-policy, timeout, output, and child
  environment boundaries must not be weakened by later work.
- Milestone 3 must continue to execute every tool through `ToolRegistry`; it may
  not add direct file or subprocess access to `AgentLoop`.
- The tool execution layer must never receive provider secrets. A prompt is not
  a substitute for host-side path, command, or environment checks.
- Only the host `VerificationGate` may grant `VERIFIED`; model text, tool
  arguments, and ordinary command results are never verification authority.
- Load and freeze verification commands before the first model request. Every
  model-side workspace mutation must invalidate older verification, while Gate
  commands must not advance the model mutation sequence.
- Replay is read-only evidence inspection. It must never call a model, execute a
  tool or command, apply a patch, or otherwise modify the workspace.
- Trace events, `result.json`, and `patch.diff` must never persist provider
  secrets.
- Do not amend, replace, or otherwise rewrite the reviewed Milestone 1 or
  Milestone 2 commit history.
- Milestone 3 completion starts feature freeze. Subsequent work is limited to
  bug fixes, tests, documentation, and release work until explicitly authorized.
