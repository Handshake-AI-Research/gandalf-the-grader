# Variant 6 Migration Guide

## What Changed

- **Default model prefix**: `google/gemini-2.5-flash` changed to `gemini/gemini-2.5-flash`.
- **`sandbox_user` is now optional** (default `None`). When `None`, the judge runs as the current user without `sudo`.
- **`output_dir` no longer has a default** -- it must be set explicitly in the TOML config.
- **New: `rubric` field** for inline rubric (alternative to `rubric_path`). Exactly one of `rubric` or `rubric_path` must be set.
- **New: `judge_prompt` and `judge_prompt_path`** for custom judge prompt template override. These are mutually exclusive. When set, the provided template is used instead of the built-in Jinja2 templates.

## Breaking Changes

- TOML configs that rely on `output_dir` defaulting to `/logs/verifier` **must** now set it explicitly.
- If your TOML does not set `model` explicitly and you relied on the `google/` prefix, verify that `gemini/` works with your LiteLLM setup.

## Migration

1. Add `output_dir = "/logs/verifier"` (or your preferred path) to all TOML configs.
2. Verify model prefix routing works with your LiteLLM configuration.
3. If you were setting `sandbox_user` but do not need sandboxed execution, you may now omit it.
