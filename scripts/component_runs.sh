#!/usr/bin/env bash
# Phase 1: are the components real?  Phase 2: do they help?  ~8 minutes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-$HOME/.venvs/task4/bin/python}"
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export CAFFEINATED=1; exec caffeinate -i "$0" "$@"
fi
LOG="reports/components_$(date +%Y%m%d_%H%M).log"; mkdir -p reports

echo "### PHASE 1 - is the decomposition real?" | tee -a "$LOG"
"$PY" scripts/validate_components.py dev 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "### PHASE 2 - does comparing component-wise help?" | tee -a "$LOG"
for s in l7s_theme l7s_action l7s_outcome l7_components; do
  "$PY" -m task4.cli scorer=$s data=dev >>"$LOG" 2>&1 \
    && grep "acc=" "$LOG" | tail -1 | sed 's/.*INFO] - /    /' | tee -a "$LOG" \
    || echo "    $s FAILED" | tee -a "$LOG"
done
echo | tee -a "$LOG"
"$PY" -c "
import json,glob
for f in glob.glob('reports/results/dev__l7_components__*.json'):
    d=json.load(open(f)); w=d.get('fitted',{})
    print('fitted component weights:', w)
" | tee -a "$LOG"
"$PY" scripts/compare_runs.py | tee -a "$LOG"
echo "log: $LOG"
