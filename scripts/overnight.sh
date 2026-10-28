#!/usr/bin/env bash
# The remaining gaps, about ten minutes total. Not actually an overnight job.
#   bash scripts/overnight.sh
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

# 1. A clean L4 pair. L4n was run with the actor filter and L4x without it, so
#    the +3.50 between them confounds two changes. This fixes that.
run "L4n, all noun chunks (clean pair for L4x)" \
    scorer=l4n_roles_nocoref data=dev scorer.params.actors_only=false

# 2. Does a stronger encoder rescue the alignment? MiniLM gave +1.00 over
#    Jaccard; e5 is the backbone's own encoder.
run "L3x with the e5 encoder" \
    scorer=l3x_characters_embed data=dev \
    scorer.params.embed_model=intfloat/e5-base-v2 \
    'scorer.params.embed_prefix=query: '

# 3. The actor-filtered variant of L3x, to see whether the filter still costs
#    points once the comparison is semantic rather than lexical.
run "L3x with the actor filter" \
    scorer=l3x_characters_embed data=dev scorer.params.actors_only=true

echo "=== done $(date +%H:%M:%S) ===" | tee -a "$LOG"
"$PY" scripts/compare_runs.py 2>&1 | tee -a "$LOG"
echo "full log: $LOG"
