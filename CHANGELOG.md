# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2025-01-XX

### Changed

- **BREAKING**: Renamed project from `gandalf-the-grader` to `infinity-grader`
- **BREAKING**: Renamed Python package from `gandalf` to `infinity_grader`
- **BREAKING**: Renamed CLI commands: `gandalf-the-grader` → `infinity-grader`, `gandalf-the-grader-judge` → `infinity-grader-judge`
- Updated repository organization from Handshake-AI-Research to Infinity-Megatron
- Updated all documentation and examples to reflect new branding

### Migration Guide

To migrate from gandalf-the-grader to infinity-grader:

1. Update installation: `uv tool install infinity-grader`
2. Update imports: `from gandalf.*` → `from infinity_grader.*`
3. Update CLI commands: `gandalf-the-grader` → `infinity-grader`

## [1.0.0]

### Added

- Initial open-source release of Gandalf the Grader.

[2.0.0]: https://github.com/Infinity-Megatron/infinity-grader/releases/tag/v2.0.0
[1.0.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v1.0.0
