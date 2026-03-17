# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.2]

### Fixed
- Fix `PermissionError` when judge runs as a different user via sudo by isolating
  the judge's home environment.

## [0.4.1]

### Fixed
- Fix ambiguous judge prompt that caused wrong boolean encoding for negative criteria.

## [0.4.0]

### Added
- Support for negative criteria weights in rubrics.
- Expose `raw_score` in `info.json` alongside the normalized reward.

### Changed
- Normalize reward to [0, 1] range in `reward.json`.
- Rename `passed` to `met` in verdict output and all data models.
- Rename `score` to `reward` in `reward.json` for Harbor compatibility.

## [0.3.0]

### Fixed
- Fix `PermissionError` in cross-user judge runs by writing temp files to clone directory.
- Skip unreadable files during workspace clone instead of failing.

## [0.2.0]

### Added
- Nullable verdicts with retry logic for transient judge errors.
- Hard fail mode for unrecoverable judge errors.
- OpenTelemetry tracing support for judge subprocess.

## [0.1.0]

### Added
- Initial open-source release of gandalf-the-grader.
- Agent-as-judge grading framework with outer orchestrator and inner judge architecture.
- Sequential and batch evaluation modes.
- Configurable rubric-based scoring with weighted criteria.
- MCP server support for tool-augmented judging.
- TOML-based verifier configuration.
- ATIF trajectory format support.

[0.4.2]: https://github.com/Handshake-AI-Research/gandalf-the-grader/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Handshake-AI-Research/gandalf-the-grader/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v0.1.0
