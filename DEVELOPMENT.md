# Development

Gandalf uses the [Hatch] project manager ([installation instructions][hatch-install]).

Hatch automatically manages dependencies and runs testing, type checking, and other operations in isolated [environments][hatch-environments].

[Hatch]: https://hatch.pypa.io/
[hatch-install]: https://hatch.pypa.io/latest/install/
[hatch-environments]: https://hatch.pypa.io/latest/environment/

## Testing

Run the unit tests on your local machine with:

```bash
hatch test
```

The [`test` command][hatch-test] supports options such as `-c` for measuring test coverage, `-a` for testing with a matrix of Python versions, and appending an argument like `hatch test tests/test_config.py::TestLoadConfig` for running a single test.

[hatch-test]: https://hatch.pypa.io/latest/tutorials/testing/overview/

### LLM end-to-end tests

Tests marked with `@pytest.mark.llm` call a real LLM and require `LLM_API_KEY` to be set. These are excluded from the default CI run. To run them locally:

```bash
hatch test -m llm
```

## Type checking

Run the [mypy static type checker][mypy] with:

```bash
hatch run types:check
```

[mypy]: https://mypy-lang.org/

## Formatting and linting

Run the [Ruff][ruff] formatter and linter with:

```bash
hatch fmt
```

This will automatically make [safe fixes][fix-safety] to your code. To only check without modifying files:

```bash
hatch fmt --check
```

[ruff]: https://github.com/astral-sh/ruff
[fix-safety]: https://docs.astral.sh/ruff/linter/#fix-safety

## Packaging

Build source and wheel distributions with:

```bash
hatch build
```

See [`hatch build`][hatch-build] and [`hatch publish`][hatch-publish] for more details.

[hatch-build]: https://hatch.pypa.io/latest/build/
[hatch-publish]: https://hatch.pypa.io/latest/publish/

## Continuous integration

Testing, type checking, and formatting/linting is [checked in CI](.github/workflows/ci.yml).
