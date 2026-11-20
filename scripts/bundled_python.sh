#!/usr/bin/env bash
# USD inspection works without starting Kit or using the GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
export HOME="$PROJECT_ROOT/.runtime_home"
unset CONDA_PREFIX
for usd_lib in "$ISAAC_PATH"/extscache/omni.usd.libs-*; do
  export PYTHONPATH="$usd_lib:$PYTHONPATH"
  export LD_LIBRARY_PATH="$usd_lib/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
exec "$ISAAC_PATH/python.sh" "$@"
