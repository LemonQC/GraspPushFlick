#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$PROJECT_ROOT"
bash scripts/check_system.sh
bash scripts/audit_storage.sh --baseline
bash scripts/setup_isaac.sh
bash scripts/download_isaac_assets.sh
bash scripts/download_ycb.sh
bash scripts/bundled_python.sh scripts/convert_assets.py
bash scripts/bundled_python.sh scripts/check_asset_integrity.py
bash scripts/run_headless.sh --test physics
bash scripts/run_gui.sh --test physics
bash scripts/run_robot_smoke.sh
bash scripts/run_camera_smoke.sh
bash scripts/run_object_smoke.sh
for scene in s1 s2; do
  bash scripts/run_headless.sh --test clutter --scene "$scene"
  bash scripts/run_gui.sh --test clutter --scene "$scene"
done
bash scripts/audit_storage.sh
# Readiness is calculated from saved measurements; no unconditional success message.
python scripts/summarize_m0.py
