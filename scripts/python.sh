#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
test -x "$APFG_PYTHON" || { echo 'Run scripts/setup_isaac.sh first' >&2; exit 2; }
source "$ISAAC_PATH/setup_conda_env.sh"
export LD_PRELOAD="$ISAAC_PATH/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}"
# Only the simulator child gets a private HOME; DISPLAY/XAUTHORITY are inherited.
export HOME="$PROJECT_ROOT/.runtime_home"
exec "$APFG_PYTHON" "$@"
