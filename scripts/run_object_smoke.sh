#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$PROJECT_ROOT"
# Independent processes, each with its own initial conditions and video.
"$APFG_PYTHON" - "$@" <<'PY'
import json, subprocess, sys
from pathlib import Path
root=Path.cwd()
registry=json.loads((root/'assets/manifests/object_registry.json').read_text())['objects']
required=['003_cracker_box','004_sugar_box','005_tomato_soup_can','006_mustard_bottle','007_tuna_fish_can','008_pudding_box',
          '009_gelatin_box','011_banana','021_bleach_cleanser','024_bowl','025_mug','035_power_drill','037_scissors']
available={e['asset_id'] for e in registry}
for name in required:
    asset_id='ycb:'+name
    if asset_id not in available: raise SystemExit('Missing required asset '+asset_id)
    subprocess.run([str(root/'scripts/run_headless.sh'),'--test','objects','--asset-id',asset_id,*sys.argv[1:]],check=True)
PY
