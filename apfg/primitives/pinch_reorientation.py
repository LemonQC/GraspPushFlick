"""Observed-edge pinch, real frictional lift/roll/place/release.

A separate physical reorientation candidate for an exposed overhang; never
attaches, resets or prescribes an object transform. Failure remains failure.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from apfg.primitives.swept import swept_check

REVISION='observed_pinch_axis_roll_v2_1'

def propose(points):
    if len(points)<30:return []
    lo,hi=np.quantile(points,.02,axis=0),np.quantile(points,.98,axis=0);extent=hi-lo
    # Vertical jaw opening must clear the observed thickness. A real support
    # must have left an observed underside gap; do not infer a phantom support.
    axis=int(np.argmax(extent[:2]));length=extent[axis]
    if length<.10:return []
    proposals=[]
    for sign in [1.,-1.]:
        d=np.zeros(3);d[axis]=sign;x=np.array([0.,0.,1.]);z=d;y=np.cross(z,x);q=Rotation.from_matrix(np.column_stack([x,y,z])).as_quat()
        edge=points[points[:,axis]<(lo[axis]+.028)] if sign>0 else points[points[:,axis]>(hi[axis]-.028)]
        if len(edge)<20:continue
        elo,ehi=np.quantile(edge,.03,axis=0),np.quantile(edge,.97,axis=0)
        if not .025<elo[2] or not .025<ehi[2]-elo[2]<.072:continue
        tcp=(lo+hi)/2;tcp[2]=(elo[2]+ehi[2])/2;tcp[axis]=(lo[axis]-.0015 if sign>0 else hi[axis]+.0015)
        # Source joint 6 rotates around TCP approach without moving TCP.
        # With the initial jaw axis up, +90 also brings the real camera above
        # the hand. This physically rolls the object onto a different face.
        roll_axis=d
        proposals.append({'tcp_xyz_m':tcp.tolist(),'approach_direction':d.tolist(),'quat_xyzw':q.tolist(),'roll_axis':roll_axis.tolist(),'roll_degrees':90.,'release_tcp_z_m':float(max(.095,elo[2]+extent[1-axis]/2+.015)),'speed_m_s':.025,'observed_edge_band_min_z_m':float(elo[2]-.004)})
    return proposals

def plan(model,start,parameters,scene,target):
    p=parameters;tcp=np.asarray(p['tcp_xyz_m']);d=np.asarray(p['approach_direction']);r=Rotation.from_quat(p['quat_xyzw']);roll=np.asarray(p['roll_axis']);pre=tcp-.08*d;hover=pre.copy();hover[2]=.32;lift=tcp.copy();lift[2]=.34
    lift[2]=.27
    goals=[hover,pre,tcp,lift];quats=[r.as_quat()]*4;phases=['hover','approach','approach','lift']
    path=[np.asarray(start)]
    try:
        # Anchor the low contact branch first. Forward seeding from a high
        # hover can select the limit-valid branch with wrist link 4 below the
        # table; the full source check still decides acceptance.
        touch_joint=model.ik(tcp,r.as_quat(),start)
        pre_joint=model.ik(pre,r.as_quat(),touch_joint)
        hover_joint=model.ik(hover,r.as_quat(),pre_joint)
        path.extend([hover_joint,pre_joint,touch_joint,model.ik(lift,r.as_quat(),touch_joint)])
        lifted=path[-1].copy()
        for u in np.linspace(1/6,1,6):
            joint=lifted.copy();joint[5]+=np.radians(p['roll_degrees'])*u
            if np.any(joint<model.lower) or np.any(joint>model.upper):raise ValueError('WRIST_ROLL_JOINT_LIMIT')
            pose=model.fk(joint)['ral_tcp'];goals.append(pose[:3,3]);quats.append(Rotation.from_matrix(pose[:3,:3]).as_quat());phases.append('reorient');path.append(joint)
        placed=goals[-1].copy();placed[2]=p['release_tcp_z_m'];raised=goals[-1].copy()
        for u in np.linspace(.25,1,4):
            goal=raised+u*(placed-raised);q=quats[-1];path.append(model.ik(goal,q,path[-1]));goals.append(goal);quats.append(q);phases.append('place')
        withdrawn=placed-.08*d;path.append(model.ik(withdrawn,quats[-1],path[-1]));goals.append(withdrawn);quats.append(quats[-1]);phases.append('withdraw')
    except ValueError as exc:return None,{'reason':'IK','detail':str(exc),'solved_waypoints':len(path)-1}
    t=np.eye(4);t[:3,:3]=r.as_matrix();t[:3,3]=tcp
    main=swept_check(model,path[:-1],scene,target,t,carry_from_segment=3,carry_target=True,opening=.08)
    if main['status']!='PASS':return None,main
    # The jaws open BEFORE withdrawal. Do not transport the released object
    # with the hand during that source check. This is a kinematic visible-cloud
    # prediction only: the simulator object is never assigned this transform.
    placed_frame=model.fk(path[-2])['ral_tcp'];relative=(target-t[:3,3])@t[:3,:3]
    released=relative@placed_frame[:3,:3].T+placed_frame[:3,3]
    other=scene[cKDTree(target).query(scene)[0]>.010] if len(target) and len(scene) else scene
    predicted=np.concatenate([other,released])
    withdrawal=swept_check(model,path[-2:],predicted,np.empty((0,3)),np.eye(4),carry_from_segment=0,carry_target=False,opening=.08)
    if withdrawal['status']!='PASS':return None,dict(withdrawal,scope='post-release source check; released target remains static in the prediction')
    checked={'status':'PASS','main_held_cloud':main,'post_release':withdrawal,'scope':'One bounded path partitioned at physical jaw release; no per-frame checks or shadow simulation'}
    return {'revision':REVISION,'mode':'pinch_roll','kind':'FLIP','parameters':parameters,'goals':[v.tolist() for v in goals],'quats_xyzw':[q.tolist() for q in quats],'phases':phases,'joint_path':[q.tolist() for q in path],'swept':checked},checked

def execute(controller,plan):
    if np.max(abs(controller.joints()-np.asarray(plan['joint_path'][0])))>.025:raise ValueError('REORIENTATION_PLAN_STALE')
    controller.set_opening(.08,phase='flip_pinch_open')
    for i,(goal,q,phase) in enumerate(zip(plan['goals'],plan['quats_xyzw'],plan['phases'])):
        if phase=='lift':controller.set_opening(0.,phase='flip_pinch_close')
        if phase=='withdraw':controller.set_opening(.08,phase='flip_pinch_release')
        controller.move_tcp(goal,q,phase='flip_'+phase,joint_goal=plan['joint_path'][i+1],speed_m_s=.025 if phase in ('approach','place') else None)
    controller.phase='flip_settle';controller.hold(3.)
    return {'revision':REVISION,'artificial_attachment':False,'source_jaw_force_and_limits':'unchanged'}
