# Development

## Prerequisites

- Python 3.12+
- [Hatch](https://hatch.pypa.io/) (`pip install hatch`)

## Running tests

```bash
hatch test
```

To include LLM integration tests:

```bash
hatch test -- -m llm
```

## Type checking

```bash
hatch run types:check
```

## Formatting

```bash
hatch fmt
```

## Building

```bash
hatch build
```
