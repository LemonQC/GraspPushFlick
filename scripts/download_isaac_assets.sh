#!/usr/bin/env bash
# Reuse only the small required subtrees of an official, version-matched asset pack.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
if [[ ! -f "$PROJECT_ROOT/assets/isaac/Isaac/Robots/Franka/franka.usd" ]]; then
  test -n "${APFG_ISAAC_ASSET_SOURCE:-}" || { echo 'Set APFG_ISAAC_ASSET_SOURCE to the official Isaac 4.5 asset root (containing Isaac/).' >&2; exit 2; }
  for part in Isaac/Robots/Franka Isaac/Props/YCB; do
    mkdir -p "$PROJECT_ROOT/assets/isaac/$part"
    rsync -a --no-owner --no-group --partial --exclude='.thumbs' "$APFG_ISAAC_ASSET_SOURCE/$part/" "$PROJECT_ROOT/assets/isaac/$part/"
  done
fi
mkdir -p "$PROJECT_ROOT/assets/isaac/NVIDIA"
if [[ ! -f "$PROJECT_ROOT/assets/isaac/Isaac/Environments/Grid/default_environment.usd" ]]; then
  test -n "${APFG_ISAAC_ASSET_SOURCE:-}" || { echo 'Set APFG_ISAAC_ASSET_SOURCE to copy the required Grid environment.' >&2; exit 2; }
  mkdir -p "$PROJECT_ROOT/assets/isaac/Isaac/Environments/Grid"
  rsync -a --no-owner --no-group --partial --exclude='.thumbs' "$APFG_ISAAC_ASSET_SOURCE/Isaac/Environments/Grid/" "$PROJECT_ROOT/assets/isaac/Isaac/Environments/Grid/"
fi
exec "$PROJECT_ROOT/scripts/bundled_python.sh" "$PROJECT_ROOT/scripts/check_asset_integrity.py"
