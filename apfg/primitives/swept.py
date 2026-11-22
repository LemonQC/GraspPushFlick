"""One bounded kinematic full-arm/held-cloud check before grasp execution.

Uses source URDF collision meshes and measured scene points, not a shadow
simulation. It is conservative and cannot certify unobserved geometry.
"""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation
from apfg.robots.sr5 import origin


def source_shapes(model):
    if hasattr(model,'_m1_collision_shapes'):return model._m1_collision_shapes
    shapes=[]
    for name,link in model.links.items():
        if name in ('world','xMateSR5_base'):continue
        for collision in link.findall('collision'):
            g=collision.find('geometry')
            if g.find('box') is not None:
                dims=np.fromstring(g.find('box').get('size'),sep=' ');vertices=np.array(list(product(*[[-v/2,v/2] for v in dims])))
            elif g.find('mesh') is not None:
                mesh=g.find('mesh');data=(model.urdf.parent/mesh.get('filename')).read_bytes()
                tri=np.frombuffer(data,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]),offset=84)
                vertices=tri['vertices'].reshape(-1,3).astype(float)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            else:continue
            vertices=np.unique(vertices,axis=0);hull=ConvexHull(vertices)
            shapes.append({'link':name,'origin':origin(collision),'vertices':vertices[hull.vertices],
                           'planes':hull.equations,'lo':vertices.min(axis=0),'hi':vertices.max(axis=0)})
    model._m1_collision_shapes=shapes
    return shapes


def remove_observed_robot(points,model,joints,margin=.004):
    """Proprioception/source-model ego mask; no environmental GT is read."""
    points=np.asarray(points);keep=np.ones(len(points),bool);frames=model.fk(joints,opening=.08)
    for shape in source_shapes(model):
        transform=frames[shape['link']]@shape['origin']
        local=(points-transform[:3,3])@transform[:3,:3]
        broad=np.all((local>=shape['lo']-margin)&(local<=shape['hi']+margin),axis=1)&keep
        ids=np.flatnonzero(broad)
        if len(ids):
            eq=shape['planes'];inside=np.all(local[ids]@eq[:,:3].T+eq[:,3]<=margin,axis=1)
            keep[ids[inside]]=False
    return points[keep]


def swept_check(model,joint_path,scene_points,target_points,grasp_tcp,carry_from_segment=3,samples=7):
    scene=np.asarray(scene_points,float);target=np.asarray(target_points,float)
    if len(scene):
        _,idx=np.unique(np.floor(scene/.006).astype(np.int64),axis=0,return_index=True);scene=scene[idx]
    shapes=source_shapes(model);reports=[];sample_count=0
    non_target=scene
    if len(target) and len(scene):non_target=scene[cKDTree(target).query(scene)[0]>.010]
    for segment,(start,end) in enumerate(zip(joint_path,joint_path[1:])):
        points=scene if segment<carry_from_segment else non_target
        for u in np.linspace(0,1,samples):
            q=np.asarray(start)+u*(np.asarray(end)-start);frames=model.fk(q,opening=.08);sample_count+=1
            for shape in shapes:
                transform=frames[shape['link']]@shape['origin'];vertices=shape['vertices']@transform[:3,:3].T+transform[:3,3]
                if vertices[:,2].min()<-.004:
                    return {'status':'FAILED','reason':'known_table_intersection','link':shape['link'],'segment':segment,'sample':float(u),'checked_samples':sample_count}
                if not len(points):continue
                local=(points-transform[:3,3])@transform[:3,:3]
                inside=np.all((local>shape['lo']+.002)&(local<shape['hi']-.002),axis=1)
                if inside.sum()>2:
                    equations=shape['planes'];collisions=np.all(local[inside]@equations[:,:3].T+equations[:,3]<-.001,axis=1)
                    if collisions.sum()>2:
                        return {'status':'FAILED','reason':'observed_cloud_intersection','link':shape['link'],'segment':segment,'sample':float(u),'points':int(collisions.sum()),'checked_samples':sample_count}
            if segment>=carry_from_segment and len(target):
                rel=(target-grasp_tcp[:3,3])@grasp_tcp[:3,:3]
                moved=rel@frames['ral_tcp'][:3,:3].T+frames['ral_tcp'][:3,3]
                if moved[:,2].min()<-.004:return {'status':'FAILED','reason':'held_observed_target_table_intersection','segment':segment}
                if len(non_target) and np.count_nonzero(cKDTree(non_target).query(moved)[0]<.004)>3:
                    return {'status':'FAILED','reason':'held_observed_target_obstacle_intersection','segment':segment}
    return {'status':'PASS','checked_samples':sample_count,'source_collision_links':len(shapes),
            'scope':'source convex collision hulls, fixed table, measured RGBD scene and transported visible target cloud; unobserved geometry not certified; no per-step scans or physics preview'}
