# Contributing

TracePilot is a personal AI Agent learning project built around LangGraph, local memory, sandboxed tools, scheduled tasks, audit logs, and MCP tool integration.

## Local Setup

Create and activate a virtual environment before installing dependencies. Do not install dependencies into the global Python environment.

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Copy the environment template and fill in local values:

```cmd
copy .env.example .env
tracepilot config
```

## Development Checks

Run these checks before opening a pull request:

```cmd
python -m compileall tracepilot entry tests examples
python -m pytest tests -q
```

## Runtime Data

Keep runtime state out of commits. This includes `.env`, logs, SQLite checkpoint files, task queues, generated transcripts, tool result dumps, and user memory files under `workspace/`.

## Pull Request Guidelines

- Keep changes focused and easy to review.
- Add or update tests when behavior changes.
- Prefer existing project patterns over new abstractions.
- Document important behavior in `README.md` or `docs/` when it affects users.
