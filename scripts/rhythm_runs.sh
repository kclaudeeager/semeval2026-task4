#!/usr/bin/env bash
# The sequential-structure layers. About four minutes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$HOME/.venvs/task4/bin/python}"
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export CAFFEINATED=1; exec caffeinate -i "$0" "$@"
fi
LOG="reports/rhythm_$(date +%Y%m%d_%H%M).log"; mkdir -p reports
run () { local l="$1"; shift
  echo "=== $l $(date +%H:%M:%S) ===" | tee -a "$LOG"
  "$PY" -m task4.cli "$@" >>"$LOG" 2>&1 && grep "acc=" "$LOG" | tail -1 | sed 's/.*INFO] - /    /' | tee -a "$LOG" || echo "    FAILED" | tee -a "$LOG"; }

run "positional, 5 phases"  scorer=l6_positional_segments data=dev
run "elastic (DTW), 5"      scorer=l6e_elastic_segments  data=dev
run "positional, 3 phases"  scorer=l6_positional_segments data=dev scorer.params.segments=3
run "positional, 8 phases"  scorer=l6_positional_segments data=dev scorer.params.segments=8
"$PY" scripts/compare_runs.py | tee -a "$LOG"
