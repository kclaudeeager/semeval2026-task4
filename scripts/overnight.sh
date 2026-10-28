#!/usr/bin/env bash
# The two runs still missing, about five minutes.
#   make runs
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$HOME/.venvs/task4/bin/python}"

# Re-exec under caffeinate on macOS so a closed lid does not pause the run.
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export CAFFEINATED=1
  exec caffeinate -i "$0" "$@"
fi

LOG="reports/runs_$(date +%Y%m%d_%H%M).log"
mkdir -p reports

run () {
  local label="$1"; shift
  echo "=== $label  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  if "$PY" -m task4.cli "$@" >>"$LOG" 2>&1; then
    grep "acc=" "$LOG" | tail -1 | sed 's/.*INFO] - /    /' | tee -a "$LOG"
  else
    echo "    FAILED - see $LOG" | tee -a "$LOG"
  fi
}

# 1. Clean L4 pair. L4x only exists with all noun chunks; L4n/actors is 53.50.
#    This gives L4 the same one-variable comparison L3 already has.
run "L4x roles, actor filter" \
    scorer=l4x_roles_embed data=dev scorer.params.actors_only=true

# 2. Does the combination improve when its structural component is the better
#    one? L5 was fitted over L3n; this fits it over L3x instead.
run "L5 over the embedding-aligned layers" \
    scorer=l5n_combined_nocoref data=dev \
    'scorer.params.components=[l2_e5,l3x_characters_embed,l4x_roles_embed]'

echo "=== done $(date +%H:%M:%S) ===" | tee -a "$LOG"
"$PY" scripts/compare_runs.py 2>&1 | tee -a "$LOG"
echo "full log: $LOG"
