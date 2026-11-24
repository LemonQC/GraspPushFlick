"""Auditable DQN transition loading. Qualification/holdout never train Q."""
import hashlib
import json
from pathlib import Path

import numpy as np
from apfg.baselines.dqn import numeric_transition

ROOT=Path(__file__).resolve().parents[2]


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def manifest_rows(manifest):
    path=Path(manifest).resolve();path.relative_to(ROOT)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        for key in ('transition_path','episode_id','scene_id','split','object_id','seed','data_sha256'):
            if key not in row:raise ValueError('Missing dataset manifest field: '+key)
    train=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']=='holdout']
    for key in ['episode_id','scene_id','object_id','seed']:
        overlap={str(r[key]) for r in train}&{str(r[key]) for r in held}
        if overlap:raise ValueError('Training/holdout '+key+' overlap: '+str(sorted(overlap)))
    return rows


def load_transitions(manifest,split='train'):
    if split not in ['train','holdout','qualification']:raise ValueError('Unknown partition')
    selected=[row for row in manifest_rows(manifest) if row['split']==split]
    transitions=[]
    for row in selected:
        path=(ROOT/row['transition_path']).resolve();path.relative_to(ROOT)
        if file_sha256(path)!=row['data_sha256']:raise ValueError('Transition content hash mismatch: '+row['transition_path'])
        with np.load(path,allow_pickle=False) as data:transition=numeric_transition({k:data[k].copy() for k in data.files})
        transitions.append(transition)
    if not transitions:raise ValueError('No actual transitions in selected '+split+' partition')
    return transitions,selected
