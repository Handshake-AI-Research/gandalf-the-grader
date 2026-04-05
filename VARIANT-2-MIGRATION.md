# Variant 2 Migration Guide

## What Changed

1. **Subprocess timeout now kills entire process group via SIGKILL** instead of just the direct child via SIGTERM. Both `evaluate_criteria()` and `evaluate_all_criteria()` now use `Popen` with `start_new_session=True` and `os.killpg()` to ensure all descendant processes are terminated on timeout.

2. **LLM client now has explicit timeout and retry cap.** The `LLM()` constructor in `judge.py` now passes `timeout=120` (seconds per request) and `num_retries=3`. Previously there was no explicit timeout, and the default retry count was 5.

## Breaking Changes

None for API consumers.

**Behavioral changes:**

- Judge subprocesses that previously survived timeout (zombie grandchildren) will now be force-killed via SIGKILL to the entire process group.
- LLM calls will timeout after 120 seconds per attempt with a maximum of 3 retries instead of the previous default of 5.

## Migration

No code changes needed. If you relied on the old behavior where judge processes could outlive the timeout, that is no longer possible.

## Impact

Fixes the issue where a single failed LiteLLM connection could cause 34-minute delays via exponential backoff retry storms. The combination of a 120s per-call timeout and 3 retries (down from 5) caps the worst-case LLM call duration, while the process-group kill ensures no orphaned judge processes linger after a timeout.
