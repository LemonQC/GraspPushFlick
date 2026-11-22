#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
exec "$PROJECT_ROOT/scripts/python.sh" "$PROJECT_ROOT/scripts/sr5_graspnet_demo.py" "$@"
