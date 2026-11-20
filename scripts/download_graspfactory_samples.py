#!/usr/bin/env python3
"""Five public Panda sample meshes only. No grasp labels or dataset bulk download."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from apfg.download import download

ROOT=Path(__file__).resolve().parents[1]
repo='AutodeskRoboticsLab/graspfactory'
url=f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1'
with urllib.request.urlopen(url,timeout=40) as response:
    tree=json.load(response)
selected=[e for e in tree['tree'] if e['path'].startswith('sample_data/panda/meshes/') and e['path'].endswith('.obj')][:5]


def fetch(entry):
    url=f'https://raw.githubusercontent.com/{repo}/{tree["sha"]}/{entry["path"]}'
    path=ROOT/'assets/graspfactory'/Path(entry['path']).name
    receipt=download(url,path)
    content=path.read_bytes()
    digest=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if digest!=entry['sha']:
        raise ValueError('Official Git blob hash mismatch: '+entry['path'])
    return dict(receipt,path=str(path.relative_to(ROOT)),git_blob_sha1=digest,source_tree_sha=tree['sha'],status='PASS')


with ThreadPoolExecutor(max_workers=3) as pool:
    objects=list(pool.map(fetch,selected))
result={'source':'https://github.com/'+repo,'license':'MIT repository; verify mesh redistribution terms before publishing assets',
        'mesh_count':len(objects),'objects':objects,'physics_validation':'NOT_RUN','labels_downloaded':False}
(ROOT/'assets/manifests/graspfactory_samples.json').write_text(json.dumps(result,indent=2)+'\n')
print('GraspFactory mesh samples:',len(objects))
