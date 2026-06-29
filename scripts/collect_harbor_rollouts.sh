#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-$HOME/Downloads/harbor-research-v2-Batch1-2}"
ENV_FILE="${ENV_FILE:-$HOME/Downloads/env}"
OUT="${OUT:-_run/rollouts_no_verifier}"
LOG_DIR="${LOG_DIR:-_run/rollout_logs}"
MAX_JOBS="${MAX_JOBS:-5}"
MODEL="${MODEL:-openai/gpt-5.5}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

if ! command -v harbor >/dev/null 2>&1; then
  echo "harbor CLI not found on PATH" >&2
  exit 1
fi

mkdir -p "$OUT" "$LOG_DIR"

set -a
. "$ENV_FILE"
set +a

tasks=()
while IFS= read -r task_dir; do
  [ -f "$task_dir/task.toml" ] || continue
  tasks+=("$task_dir")
done < <(find "$TASK_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)

echo "Discovered ${#tasks[@]} Harbor task(s) under $TASK_ROOT"
echo "Writing rollouts to $OUT and logs to $LOG_DIR"

is_complete_slug() {
  local slug="$1"
  [ -n "$(find "$OUT/$slug" -path '*/agent/trajectory.json' ! -path '*/artifacts/*' -print -quit 2>/dev/null)" ]
}

slug_for_task_dir() {
  local task_dir="$1"
  local env_name
  local task_name

  env_name="$(basename "$(dirname "$task_dir")")"
  task_name="$(basename "$task_dir")"
  echo "${env_name}__${task_name}"
}

run_one() {
  local task_dir="$1"
  local env_name
  local task_name
  local slug
  local out_dir
  local log_file

  env_name="$(basename "$(dirname "$task_dir")")"
  task_name="$(basename "$task_dir")"
  slug="${env_name}__${task_name}"
  out_dir="$OUT/$slug"
  log_file="$LOG_DIR/$slug.log"

  if is_complete_slug "$slug"; then
    echo "[$slug] skipped; canonical trajectory already exists"
    return 0
  fi

  echo "[$slug] starting"
  if harbor run \
    -p "$task_dir" \
    -a codex \
    -m "$MODEL" \
    -e docker \
    -n 1 \
    --disable-verification \
    --artifact /home/agent/workspace \
    --artifact /logs/agent \
    --environment-build-timeout-multiplier 8 \
    --agent-setup-timeout-multiplier 3 \
    -o "$out_dir" \
    --env-file "$ENV_FILE" \
    --yes \
    >"$log_file" 2>&1; then
    echo "[$slug] complete"
  else
    echo "[$slug] failed; see $log_file" >&2
    return 1
  fi
}

status=0
pids=()

wait_for_slot() {
  local pid
  while [ "${#pids[@]}" -ge "$MAX_JOBS" ]; do
    pid="${pids[0]}"
    if ! wait "$pid"; then
      status=1
    fi
    pids=("${pids[@]:1}")
  done
}

wait_for_all() {
  local pid
  while [ "${#pids[@]}" -gt 0 ]; do
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    pids=()
  done
}

for task_dir in "${tasks[@]}"; do
  slug="$(slug_for_task_dir "$task_dir")"
  if is_complete_slug "$slug"; then
    echo "[$slug] skipped; canonical trajectory already exists"
    continue
  fi

  run_one "$task_dir" &
  pids+=("$!")
  wait_for_slot
done

wait_for_all

exit "$status"
