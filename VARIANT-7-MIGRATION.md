# Variant 7 Migration Guide: Deduplication of Judge Execution and Retry Logic

## What Changed

- `evaluate_criteria()` and `evaluate_all_criteria()` have been merged into a single `run_judge()` function. It detects single vs batch mode via `isinstance(judge_input, BatchJudgeInput)`.
- `_retry_individual()` and `_retry_batch()` have been merged into a single `apply_retries()` function. It uses `config.mode` to determine whether to retry one-by-one (individual) or as a batch.
- The return type is normalized to `(list[dict], dict)` for both single and batch inputs. For single-criterion calls, the verdicts list contains one element.

## Breaking Changes

If you imported or monkey-patched any of the following from `gandalf_grader.__main__`, those functions no longer exist:

- `evaluate_criteria`
- `evaluate_all_criteria`
- `_retry_individual`
- `_retry_batch`

Use `run_judge()` and `apply_retries()` instead.

## Migration

### Judge execution

Replace:
```python
# Single criterion (old)
verdict = evaluate_criteria(judge_input, sandbox_user=..., trace_path=..., timeout=...)

# Batch (old)
verdicts, llm_usage = evaluate_all_criteria(batch_input, sandbox_user=..., trace_path=..., timeout=...)
```

With:
```python
# Single criterion (new) - returns ([verdict_dict], {})
verdicts, llm_usage = run_judge(judge_input, sandbox_user=..., trace_path=..., timeout=...)
verdict = verdicts[0]

# Batch (new) - same signature, same return type
verdicts, llm_usage = run_judge(batch_input, sandbox_user=..., trace_path=..., timeout=...)
```

The unified function detects mode from the input type (`JudgeInput` vs `BatchJudgeInput`).

### Retry logic

Replace:
```python
# Old individual retry
_retry_individual(config, rubric, results, llm_usage, final_output, judge_guidance, errored)

# Old batch retry
_retry_batch(config, rubric, results, llm_usage, final_output, judge_guidance, errored)
```

With:
```python
# New unified retry (uses config.mode internally)
apply_retries(config, rubric, results, llm_usage, final_output, judge_guidance, errored)
```

### Test mocks

If your tests mock `evaluate_criteria` or `evaluate_all_criteria`, update them to mock `run_judge` instead. For individual-mode tests, update return values from flat dicts to `([dict], {})` tuples.
