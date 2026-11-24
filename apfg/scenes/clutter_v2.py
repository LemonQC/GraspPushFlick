"""Versioned local 7-body YCB clutter. Families are hypotheses, not labels."""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from apfg.scenes.m1 import registry,bounds

VERSION='clutter_scene_v2'
FAMILIES=('Direct','C1','C2','C3','Mixed')

def seal(spec):
    spec.pop('scene_content_sha256',None)
    spec['scene_content_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return spec

def validate(spec):
    if spec['schema_version']!=VERSION or spec['family'] not in FAMILIES:raise ValueError('Unknown v2 scene')
    objects=spec['objects'];entries=registry()
    if not 6<=len(objects)<=10 or sum(o['role']=='target' for o in objects)!=1:raise ValueError('Need 6..10 bodies, one target')
    for o in objects:
        lo,hi=bounds(o,entries)
        if lo[2]<.001 or lo[0]<.08 or hi[0]>.85 or lo[1]<-.45 or hi[1]>.45:raise ValueError('Source body out of open table')
        if abs(np.linalg.norm(o['orientation_xyzw'])-1)>1e-6:raise ValueError('Invalid XYZW')
    allowed={frozenset(p) for p in spec.get('support_pairs',[])}
    for i,a in enumerate(objects):
        for b in objects[i+1:]:
            lo,hi=bounds(a,entries);bl,bh=bounds(b,entries)
            if np.all(np.minimum(hi,bh)-np.maximum(lo,bl)>.0001) and frozenset([a['id'],b['id']]) not in allowed:
                raise ValueError('Initial AABB overlap '+a['id']+'/'+b['id'])
    if 'scene_content_sha256' in spec:
        clean={k:v for k,v in spec.items() if k!='scene_content_sha256'}
        if hashlib.sha256(json.dumps(clean,sort_keys=True).encode()).hexdigest()!=spec['scene_content_sha256']:raise ValueError('Scene hash changed')
    return spec

def generate(family,seed=2026112401,revision=5):
    if family not in FAMILIES:raise ValueError(family)
    entries=registry();rng=np.random.default_rng(seed);j=rng.uniform(-.001,.001,2);x,y=.42+j[0],j[1];objects=[]
    def body(name,asset,xy,r=None,z=None):
        r=Rotation.identity() if r is None else r;d=np.array(entries[asset]['geometry_dimensions_m']);h=np.abs(r.as_matrix())@d/2
        o={'id':name,'asset_id':asset,'role':'target' if name=='target' else 'distractor','position_m':[float(xy[0]),float(xy[1]),float(h[2]+.006 if z is None else z)],'orientation_xyzw':r.as_quat().tolist(),'usd_sha256':entries[asset]['usd_sha256'],'mass_kg':entries[asset]['mass_kg'],'parameter_provenance':'Existing registry; estimated physical parameters'}
        objects.append(o);return o
    lying=family in ('C2','Mixed');r=Rotation.from_euler('y',np.pi/2) if lying else Rotation.identity()
    target=body('target','ycb:004_sugar_box',(x,y),r)
    if family=='C2':target['position_m'][2]+=.083543+.006
    hx,hy=(.0881,.0471) if lying else (.0248,.0471)
    gap=.010 if family in ('C1','Mixed') else .025
    body('near_left','ycb:005_tomato_soup_can',(x-.15,y-.125) if family=='C2' else (x-hx-.034-gap,y))
    body('near_right','ycb:006_mustard_bottle',(x+hx+.0487+gap,y))
    front='ycb:021_bleach_cleanser' if family in ('C3','Mixed') else ('ycb:007_tuna_fish_can' if family=='C2' else 'ycb:010_potted_meat_can')
    fd=entries[front]['geometry_dimensions_m'];body('near_front',front,(x+.14,y-.125) if family=='C2' else (x,y-hy-fd[1]/2-.014))
    body('near_back','ycb:010_potted_meat_can' if family=='C2' else 'ycb:007_tuna_fish_can',(x,y) if family=='C2' else (x,y+hy+.0428+.015))
    body('diagonal_left','ycb:013_apple',(x-.137,y+.15) if family=='C2' else (x-.137,y+.118))
    body('diagonal_right','ycb:061_foam_brick',(x+.15,y+.145) if family=='C2' else (x+.145,y-.125))
    spec={'schema_version':VERSION,'generator_revision':revision,'family':family,'seed':int(seed),'objects':objects,'support_pairs':[['target','near_back']] if family=='C2' else [],
      'mechanism_qualification':'UNVERIFIED; physical paired development tests required','table_z_m':0.,'settle_s':4.,'physics_hz':480,'recording_fps':30,
      'target_hint':{'type':'initial_rgbd_region','world_xy_region':[x-.016,x+.016,y-.024,y+.024],'description':'Initial explicit task cue only; later identity/position from RGBD temporal association; no asset identity supplied'},
      'retrieval_region':{'target_aabb_min_y_m':.30,'command_goal_xyz_m':[.42,.40,.35],'hold_s':1.,'minimum_rise_m':.10}}
    validate(spec);return seal(spec)
