#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
exec "$APFG_PYTHON" "$PROJECT_ROOT/scripts/video_playback.py" "$@"
