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
- Do not amend, replace, or otherwise rewrite the reviewed Milestone 1 or
  Milestone 2 commit history.
