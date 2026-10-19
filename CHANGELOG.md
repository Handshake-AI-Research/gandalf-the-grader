# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0]

### Added

- Support GPT-6.1 Sol judges through OpenAI's Responses API.
- Exercise the Sol transport, tool calls, reasoning settings, and verdict parsing
  in an end-to-end test against a local HTTP server.

### Changed

- Upgrade the pinned OpenHands SDK and tools to 1.54.0.
- Adapt file-editor registration and MCP server configuration to the supported
  OpenHands SDK interfaces.
- Apply the existing OpenAI image safeguards to GPT-6.1 Sol.

### Fixed

- Constrain LiteLLM below 1.106.0 because that release requires MCP interfaces
  incompatible with the OpenHands 1.54.0 dependency stack.

## [1.0.2]

### Fixed

- Restore the `pinned` extra in published artifacts. The release workflow built
  without `HATCH_PINNED_EXTRA_ENABLE`, so `hatch-pinned-extra` skipped the hook
  and 1.0.1 shipped with no pinned dependency set. The workflow now sets the
  variable and fails the build if `Provides-Extra: pinned` is absent.

## [1.0.1]

### Added

- Route judge calls through a LiteLLM gateway, with explicit proxy mode,
  forwarded extra headers, and a child environment allowlist.
- Support remote MCP transports: `streamable-http`, `http`, and `sse`.
- Add a runnable quick start example under `examples/quickstart/`.

### Changed

- On a terminal gateway failure, `info.json` reports `reward: null` and
  `raw_score: null` rather than omitting the fields. The `gateway_error` marker
  is still present and no `reward.json` is written.

### Known issues

- The published artifact is missing the `pinned` extra. Use 1.0.2.

## [1.0.0]

### Added

- Initial open-source release of Gandalf the Grader.

[1.1.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v1.1.0
[1.0.2]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v1.0.2
[1.0.1]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v1.0.1
[1.0.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v1.0.0
