#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
exec python "$PROJECT_ROOT/scripts/download_ycb.py" "$@"
