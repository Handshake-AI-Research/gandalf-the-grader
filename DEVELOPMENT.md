# Development

Infinity Grader uses the [Hatch] project manager ([installation instructions][hatch-install]).

Hatch automatically manages dependencies and runs testing, type checking, and other operations in isolated [environments][hatch-environments].

[Hatch]: https://hatch.pypa.io/
[hatch-install]: https://hatch.pypa.io/latest/install/
[hatch-environments]: https://hatch.pypa.io/latest/environment/

## Testing

Run the unit tests on your local machine with:

```bash
hatch test
```

The [`test` command][hatch-test] supports options such as `-c` for measuring test coverage and `-a` for testing with a matrix of Python versions. To run a specific test, pass a pytest node ID:

```bash
hatch test tests/test_models.py::TestLoadConfig::test_parses_all_fields
```

[hatch-test]: https://hatch.pypa.io/latest/tutorials/testing/overview/

### LLM end-to-end tests

Tests marked with `@pytest.mark.llm` call real LLMs and are excluded from `hatch test` by default.

Some tests are parameterized across providers and read a provider-specific key (e.g. `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`), setting `LLM_API_KEY` for their own scope via `monkeypatch` so different providers don't conflict. Tests call `pytest.skip()` when the required key is not set. To run LLM tests:

```bash
hatch test -m llm
```

To run all tests (including LLM tests):

```bash
hatch test -m ""
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

## Updating the lockfile

We use [hatch-pinned-extra](https://github.com/edgarrmondragon/hatch-pinned-extra) to support an `infinity-grader[pinned]` extra that pins all transitive dependencies. You can upgrade the lockfile with:

```
uv lock --upgrade
```

## Packaging

Build source and wheel distributions with:

```bash
HATCH_PINNED_EXTRA_ENABLE=1 hatch build
```

See [`hatch build`][hatch-build] and [`hatch publish`][hatch-publish] for more details.

[hatch-build]: https://hatch.pypa.io/latest/build/
[hatch-publish]: https://hatch.pypa.io/latest/publish/

## Continuous integration

Testing, type checking, and formatting/linting is [checked in CI](.github/workflows/ci.yml).
