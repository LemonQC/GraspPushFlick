"""Parameterized physical actions. No scene-family-to-skill policy lives here."""
from dataclasses import dataclass,field
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation


class ActionType(str,Enum):
    ACTIVE='ACTIVE'
    PUSH='PUSH'
    FLIP='FLIP'
    GRASP='GRASP'


@dataclass
class Action:
    type:ActionType
    parameters:dict=field(default_factory=dict)

    def __post_init__(self):
        self.type=ActionType(self.type)
        allowed={ActionType.ACTIVE:{'camera_xyz_m','look_at_xyz_m'},
            ActionType.PUSH:{'contact_point_m','direction','distance_m','speed_m_s'},
            ActionType.FLIP:{'contact_point_m','pivot_axis','stroke_m','speed_m_s'},
            ActionType.GRASP:{'candidate','approach_m','lift_m','extraction_xyz_m'}}[self.type]
        unknown=set(self.parameters)-allowed
        if unknown:raise ValueError(f'Unknown action parameters: {sorted(unknown)}')


def vector(x):
    a=np.asarray(x,float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Expected finite XYZ vector')
    return a


def view_tcp(model,camera_xyz,look_at):
    eye=vector(camera_xyz);z=vector(look_at)-eye
    if np.linalg.norm(z)<.05:raise ValueError('Camera look-at distance must exceed 5cm')
    z/=np.linalg.norm(z);x=np.array([1.,0,0]);x-=x@z*z
    if np.linalg.norm(x)<.05:raise ValueError('Degenerate camera roll convention')
    x/=np.linalg.norm(x);y=np.cross(z,x);r=np.column_stack([x,y,z])
    source=model.fk(np.zeros(6));offset=(np.linalg.inv(source['ral_tcp'])@source['ral_optical'])[:3,3]
    return eye-r@offset,Rotation.from_matrix(r).as_quat()


def execute_view(controller,model,camera_xyz_m,look_at_xyz_m):
    xyz,q=view_tcp(model,camera_xyz_m,look_at_xyz_m)
    result=controller.move_tcp(xyz,q,phase='active_camera_move');controller.phase='active_hold';controller.hold(.5)
    return result


def execute_stroke(controller,contact_point_m,direction,distance_m,speed_m_s=.04,phase='push'):
    p=vector(contact_point_m);d=vector(direction)
    if abs(d[2])>1e-6 or np.linalg.norm(d)<1e-5:raise ValueError('Stroke direction must be horizontal/nonzero')
    if not .02<=distance_m<=.28 or not .003<=speed_m_s<=.12:raise ValueError('Stroke distance/speed outside fixed pilot bounds')
    d/=np.linalg.norm(d);z=np.array([0.,0,-1.]);y=np.cross(z,d);q=Rotation.from_matrix(np.column_stack([d,y,z])).as_quat()
    # Source closed finger pad is 12mm from TCP along horizontal closing X.
    # Housing extends 57.5mm beyond TCP along this direction. The precontact
    # pose must clear it as well as the 12mm finger nose; never disable it.
    pre=p-.080*d;end=p+(distance_m-.012)*d
    controller.set_opening(0.,phase=phase+'_close')
    hover=pre.copy();hover[2]=max(.35,pre[2]+.18)
    controller.move_tcp(hover,q,phase=phase+'_hover')
    for u in np.linspace(.25,1.,4):
        controller.move_tcp(hover+u*(pre-hover),q,phase=phase+'_approach')
    n=max(2,int(np.ceil(np.linalg.norm(end-pre)/.015)))
    for u in np.linspace(0,1,n+1)[1:]:
        controller.move_tcp(pre+u*(end-pre),q,phase=phase+'_contact',speed_m_s=speed_m_s)
    controller.move_tcp(end+[0,0,.15],q,phase=phase+'_withdraw')
    controller.phase=phase+'_settle';controller.hold(3.)
    return {'commanded_stroke_m':float(distance_m),'nominal_segment_speed_m_s':float(speed_m_s),
            'force_control':'source joint position drives/effort limits; not measured force-control calibration'}


def execute_flip(controller,contact_point_m,pivot_axis,stroke_m=.18,speed_m_s=.035):
    axis=vector(pivot_axis)
    if np.linalg.norm(axis)<1e-8 or np.linalg.norm(axis[:2])<.8*np.linalg.norm(axis):raise ValueError('Flip pivot axis must have a substantial horizontal component; yaw-only requests rejected')
    direction=np.cross(axis,[0.,0.,1.]);direction[2]=0.;direction/=np.linalg.norm(direction)
    result=execute_stroke(controller,contact_point_m,direction,stroke_m,speed_m_s,phase='flip')
    result['requested_pivot_axis']=axis.tolist()
    return result
