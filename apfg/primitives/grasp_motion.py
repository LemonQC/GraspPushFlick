"""Traceable parallel-jaw roll choice and bounded source-assembly path checks.

Raw GraspNet poses are never changed. A separate execution pose may exchange
the fingers by a pi roll about TCP +Z. Camera and plate are NOT assumed
symmetric; each proposed execution branch passes its own geometry checks.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from apfg.perception.graspnet_candidates import observed_collision
from apfg.primitives.swept import swept_check

REVISION='parallel_jaw_execution_v1'


def execution_variants(candidate,current_tcp):
    raw=Rotation.from_quat(candidate['quat_xyzw']);current=Rotation.from_matrix(np.asarray(current_tcp)[:3,:3])
    variants=[]
    for degrees in [0,180]:
        r=raw*Rotation.from_rotvec([0,0,np.radians(degrees)])
        variants.append({'xyz_m':list(candidate['xyz_m']),'quat_xyzw':r.as_quat().tolist(),
            'jaw_roll_about_approach_deg':degrees,'raw_prediction_index':candidate['index'],
            'rotation_from_current_deg':float(np.degrees((current.inv()*r).magnitude())),
            'interpretation':'Raw mapping' if degrees==0 else 'Explicit left/right finger exchange; full assembly rechecked'})
    return sorted(variants,key=lambda v:v['rotation_from_current_deg'])


def plan_grasp(model,start_joints,candidate,scene_points,target_points,approach_m=.10,lift_m=.15,extraction_xyz_m=None):
    if not .02<=approach_m<=.20 or not .05<=lift_m<=.30:raise ValueError('Grasp approach/lift outside controller bounds')
    current=model.fk(start_joints)['ral_tcp'];attempts=[]
    for pose in execution_variants(candidate,current):
        record={'execution_pose':pose}
        collision=observed_collision(pose,scene_points,model)
        if collision['collision']:
            attempts.append(dict(record,status='FAILED',reason='tool_observed_cloud_or_table',detail=collision));continue
        xyz=np.asarray(pose['xyz_m']);quat=pose['quat_xyzw'];axis=Rotation.from_quat(quat).apply([0,0,1])
        pre=xyz-approach_m*axis;hover=pre.copy();hover[2]=max(.35,pre[2]+.20)
        # Include the exact approach waypoints that will be executed, rather
        # than checking a coarse route and independently solving it again.
        goals=[hover,pre]+[xyz-(1-u)*approach_m*axis for u in np.linspace(.2,1,5)]+[xyz+[0,0,lift_m]]
        phases=['hover','pre']+['approach']*5+['lift']
        if extraction_xyz_m is not None:
            extract=np.asarray(extraction_xyz_m,float)
            if extract.shape!=(3,) or not np.isfinite(extract).all():raise ValueError('Finite extraction XYZ required')
            goals.append(extract);phases.append('extract')
        path=[np.asarray(start_joints,float)]
        try:
            for goal in goals:path.append(model.ik(goal,quat,path[-1]))
        except ValueError as exc:
            attempts.append(dict(record,status='FAILED',reason='IK',detail=str(exc)));continue
        tcp=np.eye(4);tcp[:3,:3]=Rotation.from_quat(quat).as_matrix();tcp[:3,3]=xyz
        swept=swept_check(model,path,scene_points,target_points,tcp,carry_from_segment=7)
        if swept['status']!='PASS':
            attempts.append(dict(record,status='FAILED',reason='full_arm_swept_check',detail=swept));continue
        attempts.append(dict(record,status='PASS',detail=swept))
        differences=np.diff(path,axis=0)
        return {'controller_revision':REVISION,'raw_prediction_index':candidate['index'],
            'raw_tcp_pose':{'xyz_m':candidate['xyz_m'],'quat_xyzw':candidate['quat_xyzw']},'execution_pose':pose,
            'parameters':{'approach_m':float(approach_m),'lift_m':float(lift_m),
                          'extraction_xyz_m':None if extraction_xyz_m is None else np.asarray(extraction_xyz_m,float).tolist()},
            'goals':[p.tolist() for p in goals],'phases':phases,'joint_path':[p.tolist() for p in path],
            'hover_max_joint_travel_deg':float(np.degrees(np.max(abs(differences[0])))),
            'total_absolute_joint_travel_deg':float(np.degrees(abs(differences).sum())),
            'swept':swept,'variant_attempts':attempts},attempts
    return None,attempts


def execute_grasp_plan(controller,plan,prefix='grasp',hold_s=1.):
    if np.max(abs(controller.joints()-np.asarray(plan['joint_path'][0])))>.025:
        raise ValueError('GRASP_PLAN_STALE: reobserve and plan from actual joints')
    quat=plan['execution_pose']['quat_xyzw']
    controller.commands.append({'command':'grasp_execution_plan','phase':prefix+'_plan',
        'raw_prediction_index':plan['raw_prediction_index'],'controller_revision':plan['controller_revision'],
        'execution_pose':plan['execution_pose'],'raw_tcp_pose':plan['raw_tcp_pose']})
    controller.set_opening(.08,phase=prefix+'_open')
    for index,(goal,phase) in enumerate(zip(plan['goals'],plan['phases'])):
        if phase=='lift':controller.set_opening(0.,phase=prefix+'_close')
        controller.move_tcp(goal,quat,phase=prefix+'_'+phase,joint_goal=plan['joint_path'][index+1])
    controller.phase=prefix+('_extracted_hold' if 'extract' in plan['phases'] else '_hold');controller.hold(hold_s)
    return {'executed_prediction_index':plan['raw_prediction_index'],'execution_pose':plan['execution_pose'],
            'controller_revision':plan['controller_revision']}
