# Variant 3: Prompt Templates + Subprocess Timeout Fix

## What changed

Combines prompt template changes (two-phase + finish instruction) with subprocess timeout fix (killpg) and LLM retry cap (num_retries=3, timeout=120).

- **Prompt templates** (`judge_batch.j2`, `judge_single.j2`): Judge instructions now use a two-phase structure (Phase 1: Investigation, Phase 2: Write verdict file). The "Remember: backslash escaping" reminder is removed. The final action block is replaced with a two-step instruction: write the verdict file, then call the `finish` tool.
- **Subprocess timeout** (`__main__.py`): `subprocess.run` replaced with `subprocess.Popen` using `start_new_session=True`. On timeout, the entire process group is killed via `os.killpg(os.getpgid(proc.pid), signal.SIGKILL)`, preventing orphaned child processes.
- **LLM retry cap** (`judge.py`): `LLM()` constructor now passes `timeout=120` and `num_retries=3` to bound retries and per-call duration.

## Breaking changes

None for API consumers.

- Judge is now instructed to call the `finish` tool after writing the verdict file.
- Subprocess timeouts now kill the entire process group (not just the top-level process).
- LLM retry is capped at 3 attempts with a 120-second timeout per call.

## Migration

No code changes needed for consumers of the gandalf-grader package. Verify your OpenHands runtime supports the `finish` tool.
