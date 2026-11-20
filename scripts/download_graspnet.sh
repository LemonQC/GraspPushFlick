#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
if [[ -z "${APFG_GRASPNET_MODELS_URL:-}" ]]; then
  echo 'Official models.zip is 4.3 GB. Jbox transport timed out; Google Drive endpoint did not provide a direct model archive. See assets/manifests/extension_asset_audit.json. Set APFG_GRASPNET_MODELS_URL to an authorized direct models.zip URL after access is restored.' >&2
  exit 2
fi
python - <<'PY'
import os,zipfile
from pathlib import Path
from apfg.download import download
root=Path(os.environ['PROJECT_ROOT'])
archive=root/'assets/graspnet/models.zip'
download(os.environ['APFG_GRASPNET_MODELS_URL'],archive)
destination=root/'assets/graspnet/models'
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        if not (destination/info.filename).resolve().is_relative_to(destination.resolve()): raise ValueError('Unsafe archive member')
    if z.testzip() is not None: raise ValueError('ZIP CRC failed')
    z.extractall(destination)
PY
