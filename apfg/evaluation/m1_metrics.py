"""Measured physical outcome checks. This module is evaluation-only."""
import numpy as np
from scipy.spatial.transform import Rotation


def support_state(matrix):
    t=np.asarray(matrix);up=t[:3,:3].T@np.array([0.,0.,1.])
    axis=int(np.argmax(abs(up)))
    return {'dominant_local_up_axis':axis,'sign':1 if up[axis]>=0 else -1,'local_up_vector':up.tolist(),
            'interpretation':'dominant body support-axis proxy; not an exact contact patch for irregular meshes'}


def flip_outcome(before,after,stable,contact):
    a,b=np.asarray(before),np.asarray(after)
    change=Rotation.from_matrix(b[:3,:3]@a[:3,:3].T)
    vector=change.as_rotvec();angle=float(np.linalg.norm(vector));horizontal=float(np.linalg.norm(vector[:2]))
    start,end=support_state(a),support_state(b)
    non_yaw=float(np.arccos(np.clip((a[:3,:3].T@[0,0,1])@(b[:3,:3].T@[0,0,1]),-1,1)))
    passed=bool(non_yaw>=np.radians(35) and horizontal>=np.radians(25) and stable and contact)
    return {'status':'PASS' if passed else 'FAILED','non_yaw_support_orientation_change_deg':float(np.degrees(non_yaw)),
            'rotation_angle_deg':float(np.degrees(angle)),'horizontal_rotation_vector_component_deg':float(np.degrees(horizontal)),
            'initial_support':start,'final_support':end,'stable_new_support':bool(stable),'measured_target_contact':bool(contact),
            'criterion':'>=35deg body-local gravity change, >=25deg horizontal rotation component, measured contact and stable end state; pure yaw rejected'}


def pose_stability(rows,ids,seconds=1.):
    if not rows:return {'status':'FAILED','reason':'No actual poses'}
    tail=[r for r in rows if r['time_s']>=rows[-1]['time_s']-seconds]
    if len(tail)<2 or tail[-1]['time_s']-tail[0]['time_s']<.9*seconds:return {'status':'FAILED','reason':'Insufficient measured stability window'}
    translation=rotation=0.
    per_body={}
    for name in ids:
        poses=[np.asarray(r['m1_objects'][name]['T_world']) for r in tail]
        drift=float(max(np.linalg.norm(t[:3,3]-poses[-1][:3,3]) for t in poses))
        angles=[(Rotation.from_matrix(t[:3,:3]).inv()*Rotation.from_matrix(poses[-1][:3,:3])).magnitude() for t in poses]
        turn=float(max(angles));translation=max(translation,drift);rotation=max(rotation,turn)
        per_body[name]={'position_drift_m':drift,'orientation_drift_rad':turn,'stable':drift<.002 and turn<.01}
    return {'status':'PASS' if translation<.002 and rotation<.01 else 'FAILED','coverage_s':tail[-1]['time_s']-tail[0]['time_s'],
            'max_position_drift_m':translation,'max_orientation_drift_rad':rotation,'bodies':per_body,'source':'actual successive rigid poses; no sleeping-velocity shortcut'}
