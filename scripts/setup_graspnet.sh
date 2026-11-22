#!/usr/bin/env bash
# Build inference-only upstream operators without changing Isaac/PyTorch pins.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/.conda/isaac-apfg/bin:$PATH"
export CUDA_HOME="${APFG_GRASPNET_CUDA_HOME:-/usr/local/cuda-11.3}"
test -x "$CUDA_HOME/bin/nvcc" || { echo 'Set APFG_GRASPNET_CUDA_HOME to an existing CUDA toolkit (no driver changes required).' >&2; exit 2; }
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
repo="$PROJECT_ROOT/third_party/graspnet-baseline"
if ! test -d "$repo/.git"; then
  git clone https://github.com/graspnet/graspnet-baseline.git "$repo"
fi
revision="$(git -C "$repo" rev-parse HEAD)"
test "$revision" = 280c215129f759ed8649cb4e89fc5dfee55f4f80 || { echo "Unexpected upstream revision: $revision; inspect before building." >&2; exit 2; }
"$APFG_PYTHON" -m pip install --no-deps ninja==1.11.1.3
for component in pointnet2 knn; do
  if test "$component" = pointnet2; then package=pointnet2; else package=knn_pytorch; fi
  destination="$PROJECT_ROOT/third_party/graspnet_extensions/$package"
  mkdir -p "$destination"
  if ! compgen -G "$destination/*.so" > /dev/null; then
    (cd "$repo/$component" && "$APFG_PYTHON" setup.py build_ext)
    "$APFG_PYTHON" - "$repo/$component/build" "$destination" <<'PY'
import pathlib,shutil,sys
source,destination=map(pathlib.Path,sys.argv[1:])
files=list(source.glob('lib.*/*/*.so'))
if len(files)!=1:raise SystemExit(f'Expected one real compiled operator: {files}')
shutil.copy2(files[0],destination/files[0].name)
(destination/'__init__.py').touch()
PY
  fi
done
"$APFG_PYTHON" - <<'PY'
from apfg.download import download
from pathlib import Path
import json
root=Path.cwd()
url='https://huggingface.co/dgrachev/a2_pretrained/resolve/7856e4e550052131b6ac5c0adb6564b04433ad9b/checkpoint-rs.tar'
receipt=download(url,root/'assets/models/graspnet/checkpoint-rs.tar','60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868')
receipt.update(original_model='https://github.com/graspnet/graspnet-baseline',
  official_baseline_commit='280c215129f759ed8649cb4e89fc5dfee55f4f80',
  mirror_repo='dgrachev/a2_pretrained',mirror_revision='7856e4e550052131b6ac5c0adb6564b04433ad9b',
  provenance='Public community GraspNet checkpoint-rs mirror. HF LFS SHA256 verified; official Drive returned quota exceeded; equality to official Drive checksum unverified.')
(root/'assets/manifests/graspnet_model.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
