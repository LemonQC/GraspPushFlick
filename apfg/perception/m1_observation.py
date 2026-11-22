"""Explicit policy/diagnostic observation boundary, independent of Isaac objects."""
from dataclasses import dataclass,field
from enum import Enum
import numpy as np


class InformationMode(str,Enum):
    RGBD_POLICY='RGBD_POLICY'
    ORACLE_DIAGNOSTIC='ORACLE_DIAGNOSTIC'


@dataclass
class PolicyObservation:
    rgbd:dict
    mask:dict
    proprio:dict
    task_hint:dict
    grasp_candidates:list=field(default_factory=list)
    history:list=field(default_factory=list)
    mode:str='RGBD_POLICY'


@dataclass
class DiagnosticObservation:
    visible:PolicyObservation
    oracle_state:dict
    mode:str='ORACLE_DIAGNOSTIC'


def visible_mask(rgb,depth,K,T_world_optical,hint):
    """Fair, explicit visible task cue; never reads a segmentation/GT accessor."""
    rgb=np.asarray(rgb);depth=np.asarray(depth)
    valid=np.isfinite(depth)&(depth>.005)&(depth<2.)
    if hint['type']=='visible_color_prompt':
        rule=hint['color_rule'];r,g,b=rgb[...,:3].astype(float).transpose(2,0,1)
        ratio=r/np.maximum(g,1.)
        return valid&(r>=rule['red_min'])&(g>=rule['green_min'])&(b<rule['blue_to_min_rg_max']*np.minimum(r,g))&(ratio>=rule.get('r_g_min',0.))&(ratio<=rule.get('r_g_max',float('inf')))
    if hint['type']=='workspace_prompt':
        y,x=np.indices(depth.shape);z=np.nan_to_num(depth,nan=0,posinf=0)
        optical=np.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z],axis=-1)
        world=optical@T_world_optical[:3,:3].T+T_world_optical[:3,3]
        a,b,c,d=hint['world_xy_region']
        return valid&(world[...,0]>=a)&(world[...,0]<=b)&(world[...,1]>=c)&(world[...,1]<=d)&(world[...,2]>.005)
    raise ValueError('Explicit visible color/region hint required; oracle masks are not a policy fallback')


def build_observation(sensor_provider,mode,hint,history,oracle_provider=None):
    mode=InformationMode(mode)
    rgbd,proprio=sensor_provider()  # only sensor/proprio access is allowed here
    masks={n:visible_mask(c['rgb'],c['depth_m'],c['K'],c['T_world_optical'],hint) for n,c in rgbd.items()}
    visible=PolicyObservation(rgbd,masks,proprio,dict(hint),history=list(history))
    if mode==InformationMode.RGBD_POLICY:return visible
    if oracle_provider is None:raise ValueError('Diagnostic mode requires an explicit oracle provider')
    return DiagnosticObservation(visible,oracle_provider())
