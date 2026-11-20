#!/usr/bin/env bash
# Official binary fallback for the recorded Ubuntu 20.04 host. No driver changes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$PROJECT_ROOT"
python - <<'PY'
from pathlib import Path
import shutil
assert shutil.disk_usage(Path.cwd()).free > 170*2**30, 'Need 150 GiB peak budget + 20 GiB reserve'
PY
if [[ ! -x "$ISAAC_PATH/kit/kit" ]]; then
  test -n "${APFG_ISAAC_BINARY_SOURCE:-}" || { echo 'Set APFG_ISAAC_BINARY_SOURCE to an extracted official Isaac Sim 4.5 binary distribution' >&2; exit 2; }
  rsync -a --no-owner --no-group --exclude='__pycache__' --exclude='kit/cache' --exclude='kit/data' --exclude='kit/logs' --partial "$APFG_ISAAC_BINARY_SOURCE/" "$ISAAC_PATH/"
fi
if [[ ! -x "$APFG_PYTHON" ]]; then
  HOME="$PROJECT_ROOT/.runtime_home" conda create --prefix "$PROJECT_ROOT/.conda/isaac-apfg" python=3.10 pip -y --override-channels -c conda-forge
fi
if [[ ! -d third_party/IsaacLab/.git ]]; then
  git clone --depth 1 --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi
test "$(git -C third_party/IsaacLab describe --tags --exact-match)" = v2.0.2
ln -sfn "$ISAAC_PATH" third_party/IsaacLab/_isaac_sim
"$APFG_PYTHON" -m pip install --retries 4 --timeout 60 'pip<26' wheel 'setuptools<81'
"$APFG_PYTHON" -m pip install --retries 4 --resume-retries 4 --timeout 60 --find-links "$PROJECT_ROOT/cache/wheels" -c configs/constraints.txt torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
"$APFG_PYTHON" -m pip install --retries 4 --resume-retries 4 --timeout 60 -c configs/constraints.txt -e third_party/IsaacLab/source/isaaclab -e third_party/IsaacLab/source/isaaclab_assets imageio imageio-ffmpeg scipy pytest
"$APFG_PYTHON" -m pip check
"$APFG_PYTHON" -m pip freeze > outputs/environment_check/pip_freeze.txt
git -C third_party/IsaacLab rev-parse HEAD > outputs/environment_check/isaaclab_commit.txt
