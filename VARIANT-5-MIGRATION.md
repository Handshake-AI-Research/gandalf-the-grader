# Variant 5 Migration Guide: Nested Verdict Output Format

## What changed

- **Single-criterion judge output format** changed from a flat structure:
  ```json
  {"met": true, "reasoning": "...", "evidence": [...], "llm_usage": {...}}
  ```
  to a nested structure:
  ```json
  {"verdict": {"met": true, "reasoning": "...", "evidence": [...]}, "llm_usage": {...}}
  ```

- **Legacy bare-array batch format** is no longer supported. The batch judge
  output must be a JSON object with `"verdicts"` and `"llm_usage"` keys. The
  `isinstance(data, list)` fallback that previously accepted a bare JSON array
  has been removed.

## Breaking changes

If you have external tooling that reads single-criterion judge output files
directly (not through the orchestrator), it must be updated:

- **Before:** `data["met"]`, `data["reasoning"]`, `data["evidence"]`
- **After:** `data["verdict"]["met"]`, `data["verdict"]["reasoning"]`, `data["verdict"]["evidence"]`

The orchestrator itself (`__main__.py`) is updated and handles the new format
transparently -- callers that consume `reward.json` or `info.json` are unaffected.

## Migration

1. Update any direct judge output parsers to read from `data["verdict"]` instead
   of the top-level keys.
2. If you relied on the legacy bare JSON array format for batch output, wrap your
   array in `{"verdicts": [...], "llm_usage": {}}`.
3. The orchestrator API (`reward.json`, `info.json`) is unchanged -- no action
   needed for consumers of those files.
