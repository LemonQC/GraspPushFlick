"""Auditable numeric transition loader shared by BC and high-level diffusion.

Success labels and object/seed identifiers select and audit datasets; they are
never model features. The original RGB-D and trajectory references remain in
the manifest, not silently replaced by oracle state.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,SCHEMA_VERSION

ROOT=Path(__file__).resolve().parents[2]
NUMERIC_KEYS=('obs','candidates','mask','action','reward','next_obs','next_candidates','next_mask','done')


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def project_path(path):
    value=Path(path)
    value=(ROOT/value).resolve() if not value.is_absolute() else value.resolve()
    value.relative_to(ROOT)
    return value


def read_manifest(path):
    path=project_path(path);content=path.read_text()
    if path.suffix=='.jsonl':rows=[json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        value=json.loads(content);rows=value if isinstance(value,list) else value['transitions']
    if not all(isinstance(row,dict) for row in rows):raise ValueError('Manifest must contain transition metadata objects')
    return rows


def validate_partitions(rows):
    """Forbid common objects or seed values across train and physical holdout."""
    train=[r for r in rows if r.get('split')=='train']
    holdout=[r for r in rows if r.get('split') in ('holdout','test')]
    shared_objects={r['object_id'] for r in train}&{r['object_id'] for r in holdout}
    shared_seeds={r['seed'] for r in train}&{r['seed'] for r in holdout}
    shared_scenes={r['scene_id'] for r in train}&{r['scene_id'] for r in holdout}
    if shared_objects or shared_seeds or shared_scenes:
        raise ValueError(f'Train/holdout overlap: objects={sorted(shared_objects)}, seeds={sorted(shared_seeds)}, scenes={sorted(shared_scenes)}')


def load_transition(row):
    path=project_path(row['transition_path'])
    if row.get('data_sha256') and file_sha256(path)!=row['data_sha256']:raise ValueError('Transition content hash mismatch: '+str(row['transition_path']))
    with np.load(path,allow_pickle=False) as data:
        missing=set(NUMERIC_KEYS)-set(data.files)
        if missing:raise ValueError('Missing numeric transition fields: '+str(sorted(missing)))
        values={name:np.asarray(data[name]).copy() for name in NUMERIC_KEYS}
    for name,shape in [('obs',(OBS_DIM,)),('next_obs',(OBS_DIM,)),('candidates',(MAX_CANDIDATES,CAND_DIM)),
                       ('next_candidates',(MAX_CANDIDATES,CAND_DIM)),('mask',(MAX_CANDIDATES,)),('next_mask',(MAX_CANDIDATES,))]:
        if values[name].shape!=shape:raise ValueError(f'{name} has shape {values[name].shape}, expected {shape}')
        if not np.isfinite(values[name]).all():raise ValueError('Nonfinite policy feature: '+name)
    for name in ['mask','next_mask']:values[name]=values[name].astype(bool)
    for name in ['obs','next_obs','candidates','next_candidates']:values[name]=values[name].astype(np.float32)
    action=int(values['action'].item())
    if not 0<=action<MAX_CANDIDATES or not values['mask'][action]:raise ValueError('Demonstrated/replay action is not a legal candidate')
    values.update(action=np.int64(action),reward=np.float32(values['reward'].item()),done=np.bool_(values['done'].item()))
    if not np.isfinite(values['reward']):raise ValueError('Nonfinite reward')
    return values


class TransitionDataset(Dataset):
    def __init__(self,manifest,split='train',success_only=False,episode_ids=None):
        all_rows=read_manifest(manifest);validate_partitions(all_rows)
        self.manifest=project_path(manifest);self.split=split
        self.rows=[r for r in all_rows if r.get('split')==split and (not success_only or r.get('success') is True)
                   and (episode_ids is None or r['episode_id'] in set(episode_ids))]
        self.rows.sort(key=lambda r:(r['episode_id'],int(r['step_index']),r['transition_path']))
        if success_only and any(not r.get('demonstrator_source') for r in self.rows):raise ValueError('Successful demonstrations need an explicit source')
        self.values=[load_transition(row) for row in self.rows]
        identifiers=[{'path':r['transition_path'],'episode_id':r['episode_id'],'step_index':int(r['step_index']),
                      'sha256':file_sha256(project_path(r['transition_path']))} for r in self.rows]
        self.subset_sha256=hashlib.sha256(json.dumps(identifiers,sort_keys=True).encode()).hexdigest()
        self.by_episode={}
        for index,row in enumerate(self.rows):self.by_episode.setdefault(row['episode_id'],[]).append(index)
        for episode,indices in self.by_episode.items():
            steps=[int(self.rows[index]['step_index']) for index in indices]
            if len(steps)!=len(set(steps)):raise ValueError('Repeated step in episode: '+episode)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        value=self.values[index]
        return {name:torch.as_tensor(np.asarray(value[name])) for name in NUMERIC_KEYS}

    def summary(self):
        from collections import Counter
        return {'schema_version':SCHEMA_VERSION,'split':self.split,'transitions':len(self),'episodes':len(self.by_episode),
            'subset_sha256':self.subset_sha256,'episode_ids':sorted(self.by_episode),
            'object_ids':sorted({r['object_id'] for r in self.rows}),'seeds':sorted({int(r['seed']) for r in self.rows}),
            'action_types':dict(Counter(r['action_type'] for r in self.rows)),
            'demonstrator_sources':dict(Counter(r.get('demonstrator_source','unspecified') for r in self.rows)),
            'episode_length_histogram':dict(Counter(str(len(v)) for v in self.by_episode.values())),
            'observation_shape':[OBS_DIM],'candidate_shape':[MAX_CANDIDATES,CAND_DIM],
            'success_episodes':len({r['episode_id'] for r in self.rows if r.get('success') is True}),
            'failure_episodes':len({r['episode_id'] for r in self.rows if r.get('success') is False})}


class SuccessfulDemoDataset(TransitionDataset):
    def __init__(self,manifest,split='train',episode_ids=None):super().__init__(manifest,split,True,episode_ids)

    def sequence(self,index,horizon=2):
        indices=self.by_episode[self.rows[index]['episode_id']];start=indices.index(index);future=indices[start:start+horizon]
        targets=[self.values[i]['candidates'][self.values[i]['action']] for i in future]
        mask=np.zeros(horizon,np.float32);mask[:len(targets)]=1
        while len(targets)<horizon:targets.append(targets[-1].copy())
        return torch.from_numpy(np.stack(targets)),torch.from_numpy(mask)


def split_demo_episodes(dataset,validation_fraction=.2,seed=0):
    if not 0<=validation_fraction<1:raise ValueError('Episode validation fraction must be in [0,1)')
    episodes=np.array(sorted(dataset.by_episode),dtype=str)
    if len(episodes)<2 or validation_fraction==0:return episodes.tolist(),[]
    rng=np.random.default_rng(seed);rng.shuffle(episodes)
    count=max(1,min(len(episodes)-1,int(round(len(episodes)*validation_fraction))))
    return sorted(episodes[count:].tolist()),sorted(episodes[:count].tolist())
