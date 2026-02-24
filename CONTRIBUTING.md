# Contributing

## Development Setup

```bash
git clone https://github.com/Handshake-AI-Research/gandalf-the-grader.git
cd gandalf-the-grader
uv sync --dev
```

## Running Tests

```bash
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/
```

## Pull Requests

1. Fork the repo and create a feature branch.
2. Make your changes — keep diffs focused.
3. Ensure all tests pass and linting is clean.
4. Open a PR with a clear description of what changed and why.

## Code Style

- Formatted and linted with [Ruff](https://docs.astral.sh/ruff/).
- Type-checked with [mypy](https://mypy-lang.org/) in strict mode.
- Line length limit: 120 characters.
