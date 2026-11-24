#!/usr/bin/env bash
# All original successes and failures are listed; playback never starts Isaac.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
exec "$APFG_PYTHON" "$PROJECT_ROOT/scripts/play_clutter_v2.py" "$@"
