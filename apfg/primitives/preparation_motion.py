"""Bounded RGB-D candidate checks for existing physical preparation primitives."""
import numpy as np
from scipy.spatial.transform import Rotation
from apfg.primitives.m1_actions import view_tcp
from apfg.primitives.swept import swept_check

REVISION='observed_preparation_v1'


def plan_preparation(model,start,kind,parameters,scene_points,target_points):
    p=parameters;speed=None
    if kind=='ACTIVE':
        xyz,q=view_tcp(model,p['camera_xyz_m'],p['look_at_xyz_m']);goals=[xyz];phases=['camera_move'];contact_start=2
        opening=.08
    else:
        point=np.asarray(p['contact_point_m'],float)
        d=np.asarray(p['direction'] if kind=='PUSH' else np.cross(p['pivot_axis'],[0,0,1]),float)
        d/=np.linalg.norm(d);z=np.array([0.,0.,-1.]);q=Rotation.from_matrix(np.column_stack([d,np.cross(z,d),z])).as_quat()
        length=float(p['distance_m'] if kind=='PUSH' else p['stroke_m']);speed=p['speed_m_s']
        pre=point-.080*d;end=point+(length-.012)*d;hover=pre.copy();hover[2]=max(.35,pre[2]+.18)
        goals=[hover]+[hover+u*(pre-hover) for u in np.linspace(.25,1,4)]
        phases=['hover']+['approach']*4;contact_start=len(goals)
        n=max(2,int(np.ceil(np.linalg.norm(end-pre)/.015)))
        goals.extend(pre+u*(end-pre) for u in np.linspace(0,1,n+1)[1:]);phases+=['contact']*n
        goals.append(end+[0,0,.15]);phases.append('withdraw');opening=0.
    path=[np.asarray(start,float)]
    try:
        for goal in goals:path.append(model.ik(goal,q,path[-1]))
    except ValueError as exc:return None,{'status':'FAILED','reason':'IK','detail':str(exc)}
    # The observed target may be contacted in the stroke, but is never assumed
    # attached/carried. Everything else, including the real camera/plate/table,
    # remains in the source geometry check. No object GT is consulted.
    checked=swept_check(model,path,scene_points,target_points,np.eye(4),carry_from_segment=contact_start,
                        carry_target=False,opening=opening)
    if checked['status']!='PASS':return None,checked
    return {'revision':REVISION,'kind':kind,'parameters':parameters,'quat_xyzw':np.asarray(q).tolist(),
            'goals':[np.asarray(g).tolist() for g in goals],'phases':phases,'joint_path':[v.tolist() for v in path],
            'speed_m_s':speed,'swept':checked},checked


def execute_preparation(controller,plan):
    if np.max(abs(controller.joints()-np.asarray(plan['joint_path'][0])))>.025:raise ValueError('PREPARATION_PLAN_STALE')
    kind=plan['kind'];prefix='active' if kind=='ACTIVE' else kind.lower()
    if kind!='ACTIVE':controller.set_opening(0.,phase=prefix+'_close')
    for index,(goal,phase) in enumerate(zip(plan['goals'],plan['phases'])):
        controller.move_tcp(goal,plan['quat_xyzw'],phase=prefix+'_'+phase,
                            speed_m_s=plan['speed_m_s'] if phase=='contact' else None,
                            joint_goal=plan['joint_path'][index+1])
    controller.phase=prefix+('_hold' if kind=='ACTIVE' else '_settle');controller.hold(.5 if kind=='ACTIVE' else 3.)
    return {'prepared_candidate_revision':REVISION,'kind':kind}
