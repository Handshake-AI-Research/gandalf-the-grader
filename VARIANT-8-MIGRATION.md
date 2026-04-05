# Variant 8 Migration Guide: Package Rename

## Package Rename

- **Package**: `gandalf_grader` -> `gandalf`
- **Project name**: `gandalf-grader` -> `gandalf-the-grader`

## Entry Points

| Old | New |
|-----|-----|
| `gandalf-grader` | `gandalf-the-grader` |
| `gandalf-grader-judge` | `gandalf-the-grader-judge` |

## Module Restructure

| Old | New | Notes |
|-----|-----|-------|
| `gandalf_grader.__main__` | `gandalf.orchestrator` | Renamed from `__main__.py` to `orchestrator.py` |
| `gandalf_grader.config` | `gandalf.models` | Renamed from `config.py` to `models.py` |
| `gandalf_grader.trajectory` | *(removed)* | `load_trajectory_final_output` inlined into `gandalf.orchestrator` |
| `gandalf_grader.judge` | `gandalf.judge` | Package prefix changed only |
| *(new)* | `gandalf.__about__` | Contains `__version__` for dynamic versioning |

## Class Renames

| Old | New |
|-----|-----|
| `VerifierConfig` | `GraderConfig` |
| `CriteriaResult` | `CriterionResult` |

## Field Renames

| Model | Old Field | New Field |
|-------|-----------|-----------|
| `RubricItem` | `criteria: str` | `criterion: str` |
| `JudgeInput` | `criteria: str` | `criterion: str` |
| `BatchCriterion` | `criteria: str` | `criterion: str` |
| `CriterionResult` | `criteria: str` | `criterion: str` |

**Note**: `BatchJudgeInput.criteria: list[BatchCriterion]` is unchanged (it is a list field, not a single string).

## Function Renames (Leading Underscores Removed)

All module-level functions had their leading underscores removed:

| Module | Old | New |
|--------|-----|-----|
| `gandalf.orchestrator` | `_judge_env_vars` | `judge_env_vars` |
| `gandalf.orchestrator` | `_clone_workspace` | `clone_workspace` |
| `gandalf.orchestrator` | `_save_trace` | `save_trace` |
| `gandalf.orchestrator` | `_fail_all` | `fail_all` |
| `gandalf.orchestrator` | `_run_individual` | `run_individual` |
| `gandalf.orchestrator` | `_run_batch` | `run_batch` |
| `gandalf.orchestrator` | `_run_batch_concurrent` | `run_batch_concurrent` |
| `gandalf.orchestrator` | `_get_errored_indices` | `get_errored_indices` |
| `gandalf.orchestrator` | `_write_info` | `write_info` |
| `gandalf.judge` | `_render_template` | `render_template` |
| `gandalf.judge` | `_sanitize_json` | `sanitize_json` |
| `gandalf.judge` | `_read_verdict` | `read_verdict` |
| `gandalf.judge` | `_read_batch_verdict` | `read_batch_verdict` |
| `gandalf.judge` | `_make_verdict_path` | `make_verdict_path` |
| `gandalf.judge` | `_run_agent_session` | `run_agent_session` |

## Import Path Changes

```python
# Old
from gandalf_grader.config import VerifierConfig, CriteriaResult, RubricItem, JudgeInput
from gandalf_grader.__main__ import main, evaluate_criteria
from gandalf_grader.judge import run_judge, _read_verdict
from gandalf_grader.trajectory import load_trajectory_final_output

# New
from gandalf.models import GraderConfig, CriterionResult, RubricItem, JudgeInput
from gandalf.orchestrator import main, evaluate_criteria
from gandalf.judge import run_judge, read_verdict
from gandalf.orchestrator import load_trajectory_final_output
```

## Build System Changes

- Version is now dynamic, sourced from `src/gandalf/__about__.py`
- `[tool.hatch.version]` section added with `path = "src/gandalf/__about__.py"`
- Build target: `packages = ["src/gandalf"]`

## Rubric JSON Format

The `criteria` field in rubric JSON files is now `criterion`:

```json
// Old
[{"criteria": "The file exists", "weight": 1.0}]

// New
[{"criterion": "The file exists", "weight": 1.0}]
```

## Breaking Changes for Downstream Consumers

1. **All import paths changed** -- any code importing from `gandalf_grader` must update to `gandalf`
2. **Entry point binaries renamed** -- `gandalf-grader` -> `gandalf-the-grader`, `gandalf-grader-judge` -> `gandalf-the-grader-judge`
3. **Model field `criteria` renamed to `criterion`** on `RubricItem`, `JudgeInput`, `BatchCriterion`, `CriterionResult` -- rubric JSON files and any code accessing these fields must be updated
4. **Class renames** -- `VerifierConfig` -> `GraderConfig`, `CriteriaResult` -> `CriterionResult`
5. **Functions are now public** -- leading underscores removed from all module-level functions; existing mock targets like `gandalf_grader.__main__._clone_workspace` become `gandalf.orchestrator.clone_workspace`
6. **`trajectory.py` removed** -- `load_trajectory_final_output` is now in `gandalf.orchestrator`
