# Security Policy

TracePilot is a local AI Agent workbench for learning and experimentation. The project includes application-level safeguards for local tool execution, including workspace path checks, dangerous path pattern blocking, command timeouts, runtime output truncation, and JSONL audit logs.

## Security Scope

The sandbox in this repository is an application-level directory sandbox around `workspace/office`. It is intended to reduce accidental local file access and unsafe tool execution during development.

It is not an OS-level isolation boundary. It does not currently use containers, low-privilege users, seccomp, Windows Job Objects, or a command allowlist.

## Sensitive Files

Do not commit local secrets or runtime state:

- `.env`
- API keys and provider tokens
- `workspace/state.sqlite3*`
- `workspace/tasks.json`
- `workspace/context/`
- `workspace/memory/`
- `logs/`

Use `.env.example` as the public configuration template.

## Reporting Issues

If you find a security issue, please open a GitHub issue with:

- A short description of the behavior
- Reproduction steps
- Expected impact
- Suggested fix, if available

Avoid posting real API keys, private prompts, local file contents, or user memory data in public reports.
