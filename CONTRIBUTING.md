# Contributing

Thank you for your interest in contributing to gandalf-grader!

## Development Setup

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on setting up your environment, running tests, type checking, and formatting.

```bash
git clone https://github.com/Handshake-AI-Research/gandalf-the-grader.git
cd gandalf-the-grader
```

## Pull Requests

1. Fork the repo and create a feature branch.
2. Make your changes — keep diffs focused.
3. Ensure all tests pass and linting is clean (`hatch test`, `hatch fmt --check`, `hatch run types:check`).
4. Open a PR with a clear description of what changed and why.

## Code Style

- Formatted and linted with [Ruff](https://docs.astral.sh/ruff/) via `hatch fmt`.
- Type-checked with [mypy](https://mypy-lang.org/) in strict mode.
- Line length limit: 120 characters.
