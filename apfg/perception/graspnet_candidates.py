"""GraspNet score selection and source-geometry TCP alignment, without GT input."""
import numpy as np
from scipy.spatial.transform import Rotation
from apfg.perception.graspnet_pose import prediction_to_tcp


def physical_tip_offset(model):
    """Distance from ral_tcp to the source fingers' common tip along TCP +Z."""
    f=model.fk(np.zeros(6),opening=0.)
    tcp=f['ral_tcp'];finger=model.links['ral_left_finger']
    from apfg.robots.sr5 import origin
    box=np.fromstring(finger.find('collision/geometry/box').get('size'),sep=' ')
    center=origin(finger.find('collision'))[:3,3]
    tip=f['ral_left_finger']@np.r_[center+[0,0,box[2]/2],1.]
    return float((np.linalg.inv(tcp)@tip)[2])


def aligned_tcp(row,transform,model):
    """Align physical finger tips with GraspNet's predicted depth plane.

Raw GraspNet translation is not the physical TCP. Its fingers end at local
approach coordinate depth. This fixed source-geometry mapping keeps rotations
and raw predictions unchanged, without object-specific depth tuning.
"""
    tip=physical_tip_offset(model)
    pose=prediction_to_tcp(row,transform,tcp_offset_m=float(row[3])-tip)
    pose.update(mapping_reference='physical finger tip = raw GraspNet translation + predicted depth * approach',
                physical_tip_from_tcp_m=tip,predicted_depth_m=float(row[3]))
    return pose


def ranked_candidates(rows,transform,model,limit=30):
    candidates=[];rejected={'invalid_or_width':0,'duplicate':0}
    for index in np.argsort(-rows[:,0],kind='stable'):
        row=rows[index]
        try:pose=aligned_tcp(row,transform,model)
        except ValueError:
            rejected['invalid_or_width']+=1;continue
        if row[0]<=0:continue
        r=Rotation.from_quat(pose['quat_xyzw'])
        duplicate=False
        for old in candidates:
            if np.linalg.norm(np.array(old['xyz_m'])-pose['xyz_m'])<.015 and (Rotation.from_quat(old['quat_xyzw']).inv()*r).magnitude()<np.radians(25):
                duplicate=True;break
        if duplicate:rejected['duplicate']+=1;continue
        approach=r.apply([0.,0.,1.])
        tilt=float(np.degrees(np.arccos(np.clip(approach@[0,0,-1],-1,1))))
        candidates.append(dict(index=int(index),raw_row=row.tolist(),tilt_from_world_down_deg=tilt,**pose))
        if len(candidates)>=limit:break
    return candidates,rejected


def source_tool_boxes(model):
    """Source collision bounds in TCP coordinates, with both jaws fully open."""
    from apfg.robots.sr5 import origin
    from itertools import product
    frames=model.fk(np.zeros(6),opening=.08);tcp_inv=np.linalg.inv(frames['ral_tcp'])
    parts=[]
    for name in ['ral_camera','ral_gripper','ral_left_finger','ral_right_finger','ral_plate']:
        elem=model.links[name].find('collision');g=elem.find('geometry')
        if g.find('box') is not None:
            dims=np.fromstring(g.find('box').get('size'),sep=' ')
            corners=np.array(list(product(*[[-v/2,v/2] for v in dims])))
        else:
            data=(model.urdf.parent/g.find('mesh').get('filename')).read_bytes()
            tri=np.frombuffer(data,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]),offset=84)
            vertices=tri['vertices'].reshape(-1,3)
            corners=np.array(list(product(*zip(vertices.min(axis=0),vertices.max(axis=0)))))
        t=tcp_inv@frames[name]@origin(elem)
        transformed=corners@t[:3,:3].T+t[:3,3]
        parts.append((name,transformed.min(axis=0),transformed.max(axis=0),transformed))
    return parts


def observed_collision(candidate,world_points,model,voxel_size=.005):
    """One observed-cloud filter using actual source PGIA finger/body extents.

This single-view filter is not a physical preview or full-arm collision planner.
It cannot see occluded geometry. All executed contacts remain real PhysX.
"""
    points=np.asarray(world_points)
    if len(points):
        _,indices=np.unique(np.floor(points/voxel_size).astype(np.int64),axis=0,return_index=True)
        points=points[indices]
    r=Rotation.from_quat(candidate['quat_xyzw']).as_matrix()
    p=(points-np.asarray(candidate['xyz_m']))@r
    collided=np.zeros(len(p),bool);details=[];minimum_z=float('inf')
    for name,lo,hi,corners in source_tool_boxes(model):
        lo=lo.copy();lo[2]-=.10
        inside=np.all((p>lo+.001)&(p<hi-.001),axis=1)
        collided|=inside
        pose=corners@r.T+candidate['xyz_m']
        pre=pose-.10*r[:,2]
        part_z=float(min(pose[:,2].min(),pre[:,2].min()));minimum_z=min(minimum_z,part_z)
        details.append({'part':name,'observed_collision_points':int(inside.sum()),'approach_minimum_world_z_m':part_z})
    count=int(collided.sum())
    return {'collision':count>2 or minimum_z<.002,'observed_collision_points':count,'known_table_clearance_m':minimum_z,
            'parts':details,'scope':'one source-tool bounds check along 100mm approach, observed RGBD cloud and known table plane only; camera/plate retained; no object GT/full-arm planning/physics preview'}
