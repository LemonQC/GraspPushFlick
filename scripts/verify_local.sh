#!/usr/bin/env bash
# CPU/install checks only; a zero exit does not establish M0 readiness.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$PROJECT_ROOT"
"$APFG_PYTHON" -m pip check
"$APFG_PYTHON" -m unittest discover -s tests -v
"$APFG_PYTHON" -m compileall -q apfg scripts
for script in scripts/*.sh; do bash -n "$script"; done
for mode in headless gui; do
  code=0
  bash "scripts/run_${mode}.sh" --test physics --steps "${APFG_VERIFY_STEPS:-1200}" --width "${APFG_VERIFY_WIDTH:-1280}" --height "${APFG_VERIFY_HEIGHT:-720}" || code=$?
  if [[ "$code" != 0 && "$code" != 2 ]]; then exit "$code"; fi
done
