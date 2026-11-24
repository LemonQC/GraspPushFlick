#!/usr/bin/env bash
# This entry executes actual heldout robot rollouts, not stored-label accuracy.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash scripts/eval_baseline.sh [all|grasp_only|heuristic|random_valid|dqn|bc|dp] [runner options]'
  echo 'Default: physical heldout evaluation, --headless. Pass --gui for native visualization.'
  echo 'Learned policies require --dqn-checkpoint / --bc-checkpoint / --dp-checkpoint.'
  exit 0
fi
method=all
if [[ $# -gt 0 && "$1" != --* ]]; then method="$1"; shift; fi
case "$method" in all|grasp_only|heuristic|random_valid|dqn|bc|dp) ;; *) echo "Unknown baseline: $method" >&2; exit 2 ;; esac
gui=false; headless=false
for argument in "$@"; do
  [[ "$argument" == "--gui" ]] && gui=true
  [[ "$argument" == "--headless" ]] && headless=true
  if [[ "$argument" == "--mode" || "$argument" == --mode=* ]]; then
    echo 'eval_baseline.sh always runs --mode evaluate; use the smoke runner for other modes.' >&2; exit 2
  fi
done
if $gui && $headless; then echo 'Choose --gui or --headless, not both.' >&2; exit 2; fi
display_args=()
if ! $gui && ! $headless; then display_args=(--headless); fi
exec "$PROJECT_ROOT/scripts/python.sh" "$PROJECT_ROOT/scripts/run_baseline_holdout.py" --method "$method" "${display_args[@]}" "$@"
