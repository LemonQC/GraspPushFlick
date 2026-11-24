#!/usr/bin/env bash
# CPU by default. Existing CUDA train CLIs own the shared GPU lock.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash scripts/train_baseline.sh bc|dp|dqn [training CLI options]'
  echo 'Default: actual shared dataset manifest, unique project-local output, CPU.'
  echo 'Use --device cuda explicitly; the selected trainer coordinates the GPU lock.'
  exit 0
fi
method="$1"; shift
case "$method" in
  bc|dp) trainer="$PROJECT_ROOT/scripts/train_imitation.py"; method_args=(--method "$method") ;;
  dqn) trainer="$PROJECT_ROOT/scripts/train_dqn.py"; method_args=() ;;
  *) echo "Unknown baseline: $method" >&2; exit 2 ;;
esac
run_id="$("$APFG_PYTHON" -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8])')"
exec "$APFG_PYTHON" "$trainer" "${method_args[@]}" \
  --manifest "$PROJECT_ROOT/outputs/baselines/dataset/manifest.jsonl" \
  --output "$PROJECT_ROOT/outputs/baselines/$method/$run_id" --device cpu "$@"
