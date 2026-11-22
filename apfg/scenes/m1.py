"""Seeded M1 candidate scene families, never skill-necessity labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
FAMILIES=('Direct','C1','C2','C3','C4')
TARGET_POOL=['004_sugar_box','005_tomato_soup_can','006_mustard_bottle','007_tuna_fish_can',
             '008_pudding_box','009_gelatin_box','010_potted_meat_can','061_foam_brick']


def registry():
    return {e['asset_id']:e for e in json.loads((ROOT/'assets/manifests/object_registry.json').read_text())['objects']}


def suitability(entry,position=(.42,0,.1)):
    d=np.asarray(entry['geometry_dimensions_m']);p=np.asarray(position)
    reasons=[]
    if float(min(d))>.074:reasons.append('No bounding-box grasp dimension below 80mm minus 6mm margin')
    if float(max(d))>.24:reasons.append('Object bounding extent exceeds pilot .24m limit')
    if not .22<=p[0]<=.70 or abs(p[1])>.26:reasons.append('Outside predefined SR5 tabletop pilot workspace')
    if not (ROOT/entry['usd_path']).is_file():reasons.append('Local USD missing')
    return {'status':'UNSUITABLE' if reasons else 'CONDITIONAL','reasons':reasons,
            'minimum_geometry_extent_m':float(min(d)),'maximum_geometry_extent_m':float(max(d)),
            'mass_kg':entry['mass_kg'],'physical_parameters':entry['physical_parameter_source']['status'],
            'note':'Extent/reach screening is not grasp feasibility; actual collision, IK, contact and extraction must be measured'}


def bounds(obj,entries):
    half=np.abs(Rotation.from_quat(obj['orientation_xyzw']).as_matrix())@np.array(entries[obj['asset_id']]['geometry_dimensions_m'])/2
    p=np.array(obj['position_m']);return p-half,p+half


def validate(spec,entries=None):
    entries=registry() if entries is None else entries
    if spec.get('schema_version')!='m1_scene_v1' or spec.get('family') not in FAMILIES:raise ValueError('Invalid scene schema/family')
    if not isinstance(spec.get('seed'),int) or not 3<=len(spec.get('objects',[]))<=5:raise ValueError('Need integer seed and 3..5 real YCB bodies')
    if sum(o['role']=='target' for o in spec['objects'])!=1:raise ValueError('Exactly one designated target required')
    ids=[]
    for o in spec['objects']:
        if o['asset_id'] not in entries:raise ValueError('Unknown registry asset')
        p=np.asarray(o['position_m']);q=np.asarray(o['orientation_xyzw'])
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>1e-6:raise ValueError('Invalid metre/XYZW pose')
        lo,hi=bounds(o,entries)
        if lo[2]<.001 or lo[0]<.08 or hi[0]>.85 or lo[1]<-.45 or hi[1]>.45:raise ValueError('Initial body outside workspace or penetrating tabletop')
        ids.append(o['id'])
    if len(ids)!=len(set(ids)):raise ValueError('Body IDs must be unique')
    for i,a in enumerate(spec['objects']):
        alo,ahi=bounds(a,entries)
        for b in spec['objects'][i+1:]:
            blo,bhi=bounds(b,entries)
            if np.all(np.minimum(ahi,bhi)-np.maximum(alo,blo)>.0001):raise ValueError(f'Initial AABB overlap: {a["id"]}/{b["id"]}')
    if 'scene_content_sha256' in spec:
        payload={k:v for k,v in spec.items() if k!='scene_content_sha256'}
        if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=spec['scene_content_sha256']:raise ValueError('Scene content hash mismatch; record a new scene revision after edits')
    return spec


def generate(family,seed,target_asset=None,mix=None):
    if family not in FAMILIES:raise ValueError(f'family must be {FAMILIES}')
    entries=registry();rng=np.random.default_rng(seed)
    asset=target_asset or 'ycb:004_sugar_box'
    if asset not in entries:raise ValueError('Target asset missing')
    screen=suitability(entries[asset])
    if screen['status']=='UNSUITABLE':raise ValueError('UNSUITABLE: '+'; '.join(screen['reasons']))
    flags={'clearance':family=='C1','orientation':family=='C2','visibility':family=='C3'}
    if family=='C4':
        if mix and (set(mix)-set(flags) or any(not isinstance(v,bool) for v in mix.values())):raise ValueError('Mix flags must be named boolean mechanism options')
        flags.update(mix or {'clearance':True,'orientation':True,'visibility':True})
        if sum(flags.values())<2:raise ValueError('C4 requires at least two mechanism flags')
    x,y=.42+float(rng.uniform(-.004,.004)),float(rng.uniform(-.004,.004))
    yaw=float(rng.uniform(-.10,.10))
    r=Rotation.from_euler('z',yaw)
    if flags['orientation']:r=r*Rotation.from_euler('y',np.pi/2)
    objects=[]
    def body(name,asset_id,xy,rotation,role):
        d=np.asarray(entries[asset_id]['geometry_dimensions_m'])
        half=np.abs(rotation.as_matrix())@d/2
        return {'id':name,'asset_id':asset_id,'role':role,'position_m':[float(xy[0]),float(xy[1]),float(half[2]+.008)],
                'orientation_xyzw':rotation.as_quat().tolist(),'usd_sha256':entries[asset_id]['usd_sha256'],
                'mass_kg':entries[asset_id]['mass_kg'],'parameter_provenance':'registry: estimated, not hardware calibrated'}
    target=body('target',asset,(x,y),r,'target');objects.append(target)
    half=np.abs(r.as_matrix())@np.asarray(entries[asset]['geometry_dimensions_m'])/2
    if flags['clearance']:
        can='ycb:005_tomato_soup_can';dx=entries[can]['geometry_dimensions_m'][0]/2
        for sign,name in [(-1,'left_blocker'),(1,'right_blocker')]:
            objects.append(body(name,can,(x+sign*(half[0]+dx+.007),y),Rotation.identity(),'distractor'))
        # A visible front obstacle leaves genuine alternate approaches possible;
        # whether direct grasp is blocked is NOT assumed by this family label.
        front='ycb:036_wood_block' if family=='C4' else 'ycb:003_cracker_box'
        dy=entries[front]['geometry_dimensions_m'][1]/2
        objects.append(body('front_blocker',front,(x,y-half[1]-dy-.012),Rotation.identity(),'distractor'))
    elif flags['visibility']:
        front='ycb:003_cracker_box';dy=entries[front]['geometry_dimensions_m'][1]/2
        objects.append(body('visual_occluder',front,(x,y-half[1]-dy-.016),Rotation.identity(),'distractor'))
        objects.append(body('far_left','ycb:010_potted_meat_can',(.19,.25),Rotation.identity(),'distractor'))
        objects.append(body('far_right','ycb:005_tomato_soup_can',(.70,.22),Rotation.identity(),'distractor'))
    else:
        objects.append(body('far_left','ycb:010_potted_meat_can',(.19,.25),Rotation.identity(),'distractor'))
        objects.append(body('far_right','ycb:005_tomato_soup_can',(.70,.22),Rotation.identity(),'distractor'))
        objects.append(body('far_front','ycb:007_tuna_fish_can',(.20,-.27),Rotation.identity(),'distractor'))
    spec={'schema_version':'m1_scene_v1','generator_revision':2,'family':family,'seed':int(seed),'objects':objects,'target_suitability':screen,
          'mechanism_flags':flags,'mechanism_qualification':'UNVERIFIED: candidate layout, not a necessity/success label',
          'table_z_m':0.,'observation_view':'front','target_hint':{'type':'visible_color_prompt','asset_id':asset,
              'description':'Designated yellow sugar package; visible RGB cue, not oracle instance mask',
              'color_rule':{'red_min':90,'green_min':80,'blue_to_min_rg_max':.72,'r_g_min':.75,'r_g_max':1.35}},
          'retrieval_region':{'target_aabb_min_y_m':.30,'command_goal_xyz_m':[.42,.40,.35],'hold_s':1.,'minimum_rise_m':.10},
          'physics_hz':480,'recording_fps':30,'settle_s':4.,'forbidden':'No per-family action labels, rollout resets, fabricated supports or policy GT'}
    if asset!='ycb:004_sugar_box':
        spec['target_hint']={'type':'workspace_prompt','asset_id':asset,'world_xy_region':[x-.035,x+.035,y-.035,y+.035],
            'description':'Explicit supplied task region; may include visible occluders, not perfect target segmentation'}
    validate(spec,entries)
    spec['scene_content_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return spec
