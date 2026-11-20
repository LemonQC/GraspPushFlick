#!/usr/bin/env python3
"""Separate installation integrity from actual simulator runtime acceptance."""
import datetime as dt
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import warnings

from apfg.download import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/environment_check'
versions={}
for package in ['pip','torch','torchvision','isaaclab','isaaclab_assets','numpy','pillow','imageio','imageio-ffmpeg','warp-lang']:
    try:versions[package]=metadata.version(package)
    except metadata.PackageNotFoundError:versions[package]=None
check=subprocess.run([str(ROOT/'.conda/isaac-apfg/bin/python'),'-m','pip','check'],capture_output=True,text=True)
result={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'python':platform.python_version(),
        'packages':versions,'isaac_lab_release_tag':'v2.0.2',
        'isaac_lab_commit':subprocess.check_output(['git','-C',str(ROOT/'third_party/IsaacLab'),'rev-parse','HEAD'],text=True).strip(),
        'isaac_sim_binary_version':(ROOT/'third_party/isaacsim/VERSION').read_text().strip(),
        'pip_check':{'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr},
        'installation_status':'PASS' if check.returncode==0 and all(versions.values()) else 'FAILED',
        'simulation_startup':'NOT_RUN','binary_origin':'existing user official distribution; publisher archive checksum unavailable',
        'binary_local_hashes':{name:sha256(ROOT/'third_party/isaacsim'/name) for name in ['VERSION','kit/kit','kit/python/bin/python3']}}
with warnings.catch_warnings(record=True) as recorded:
    import torch
    result['torch_cuda_available']=torch.cuda.is_available()
    result['torch_cuda_runtime']=torch.version.cuda
    result['cuda_warnings']=[str(w.message) for w in recorded]
result['simulation_startup']='BLOCKED' if not result['torch_cuda_available'] else 'NOT_RUN'
probe=subprocess.run(['bash',str(ROOT/'scripts/python.sh'),'-c',
    'from isaaclab.app import AppLauncher; import numpy,imageio,torch,json; print(json.dumps({"numpy":numpy.__version__,"imageio":imageio.__version__,"torch":torch.__version__}))'],capture_output=True,text=True)
result['app_launcher_import']='PASS' if probe.returncode==0 else 'FAILED'
if probe.returncode==0:
    result['binary_runtime_import_versions']=json.loads(probe.stdout.splitlines()[-1])
    result['binary_runtime_note']='Official binary setup adds vendor prebundles to PYTHONPATH. Runtime NumPy/ImageIO differ from clean Conda versions. pip_check above audits the isolated Conda prefix; it does not certify vendor prebundled dependency metadata or successful Kit startup.'
else:
    result['app_launcher_import_error']=probe.stderr
    result['installation_status']='FAILED'
(OUT/'environment_versions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['installation_status']=='PASS' else 1)
