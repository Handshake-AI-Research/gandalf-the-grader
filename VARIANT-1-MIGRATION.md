# Variant 1 — Prompt Template Changes

## What changed

Prompt templates (`judge_batch.j2` and `judge_single.j2`) now use a two-phase structure with explicit section headers ("Phase 1: Investigation" and "Phase 2: Write verdict file") and instruct the judge agent to call the `finish` tool after writing the verdict file.

The backslash-escaping reminder line ("Remember: in JSON strings, backslashes must be escaped as \\\\") has been removed from both templates.

## Breaking changes

None — this is a prompt-only change. No API, configuration, or dependency changes.

## Migration

No code changes are needed by downstream consumers. Behavior change: the judge agent will now be instructed to call the `finish` tool after writing the verdict file. If your OpenHands runtime does not support a `finish` tool, the judge may error on the final step — verify your runtime supports it.

## Note on removed backslash-escaping instruction

The line reminding the judge LLM to escape backslashes in JSON strings has been removed. If the judge LLM produces invalid JSON escapes more often after this change, this variant is the cause.
