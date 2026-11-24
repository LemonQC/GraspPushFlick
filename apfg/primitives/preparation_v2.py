"""Once-checked real strokes and edge-lift trials; no shadow physics."""
import numpy as np
from scipy.spatial.transform import Rotation
from apfg.primitives.preparation_motion import plan_preparation,execute_preparation
from apfg.primitives.swept import swept_check

REVISION='observed_preparation_v2'

def plan(model,start,kind,parameters,scene,subject,mode='stroke'):
    if mode=='pinch_roll':
        from apfg.primitives.pinch_reorientation import plan as pinch_plan
        return pinch_plan(model,start,parameters,scene,subject)
    if mode=='stroke':
        p,r=plan_preparation(model,start,kind,parameters,scene,subject)
        if p is not None:p.update(revision=REVISION,mode=mode)
        return p,r
    # A tilted closed finger contacts an exposed lower side and follows an
    # upward arc. This is a physical levering trial, never a guaranteed flip.
    d=np.asarray(parameters['direction'],float);d/=np.linalg.norm(d);p=np.asarray(parameters['contact_point_m']);theta=np.radians(40)
    x=d*np.cos(theta)+np.array([0,0,1])*np.sin(theta);z=d*np.sin(theta)-np.array([0,0,1])*np.cos(theta)
    q=Rotation.from_matrix(np.column_stack([x,np.cross(z,x),z])).as_quat()
    touch=p-.012*x-.005*z;pre=touch-.08*d;hover=pre+[0,0,.23]
    goals=[hover,pre+[0,0,.10],pre];phases=['hover','approach','approach'];contact_start=3
    goals.extend(pre+u*(touch-pre) for u in np.linspace(.2,1,5));phases+=['contact']*5
    for u in np.linspace(.125,1,8):
        goals.append(touch+.065*np.sin(u*np.pi/2)*d+np.array([0,0,.11*(1-np.cos(u*np.pi/2))]));phases.append('contact')
    goals.append(goals[-1]-d*.06+[0,0,.06]);phases.append('withdraw')
    path=[np.asarray(start)]
    try:
        for goal in goals:path.append(model.ik(goal,q,path[-1]))
    except ValueError as exc:return None,{'reason':'IK','detail':str(exc)}
    checked=swept_check(model,path,scene,subject,np.eye(4),carry_from_segment=contact_start,carry_target=False,opening=0.)
    if checked['status']!='PASS':return None,checked
    return {'revision':REVISION,'mode':mode,'kind':kind,'parameters':parameters,'quat_xyzw':q.tolist(),'goals':[g.tolist() for g in goals],'phases':phases,'joint_path':[j.tolist() for j in path],'speed_m_s':.025,'swept':checked},checked

def execute_plan(controller,plan):
    if plan.get('mode')=='pinch_roll':
        from apfg.primitives.pinch_reorientation import execute
        return execute(controller,plan)
    return execute_preparation(controller,plan)
