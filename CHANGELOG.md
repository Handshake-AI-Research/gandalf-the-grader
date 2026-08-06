# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `workspace_is_disposable` config option. When the caller guarantees the workspace is a
  throwaway that nothing reads after grading, and that `sandbox_user` can already read and
  write it, the judge runs against it directly instead of a clone. Saves a full copy of the
  workspace, which on a large one is the difference between grading and running out of disk.
  Rejected in combination with concurrent judge sessions, which would share the workspace.

## [1.0.0]

### Added

- Initial open-source release of Gandalf the Grader.

[1.0.0]: https://github.com/Handshake-AI-Research/gandalf-the-grader/releases/tag/v1.0.0
