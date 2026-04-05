# Variant 4: Remove JSON Sanitization from Verdict Parsing

## What changed

Removed the `_sanitize_json` regex that fixed invalid backslash escapes in LLM-generated verdict JSON files. Verdict files are now parsed with plain `json.loads()`.

The removed code included the `_ESCAPE_RE` compiled regex pattern and the `_sanitize_json()` function, along with the `import re` statement (no longer needed).

## Breaking changes

If the judge LLM writes invalid JSON (e.g., unescaped backslashes like `\$` in Excel number formats), the verdict will now fail to parse and the criterion will be scored as `met=None` (error). Previously, the sanitizer would auto-fix these by doubling invalid backslash escapes before passing the content to `json.loads()`.

## Migration

No code changes needed. Monitor for `json.JSONDecodeError` in judge traces. If these appear, the sanitizer removal may need to be paired with the prompt template changes from Variant 1 (which also removes the "remember to escape backslashes" instruction -- the theory being that removing the instruction may paradoxically improve LLM JSON output quality).

## Risk

LOW-MEDIUM. If the judge LLM reliably produces valid JSON, this change has no effect. If it doesn't, you'll see increased error rates on criteria.
