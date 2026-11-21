"""Compare the imported USD with the authoritative URDF; no model rewriting."""
import numpy as np
from scipy.spatial.transform import Rotation


def audit_import(stage, model, link_paths):
    from pxr import UsdPhysics, UsdUtils
    from apfg.robots.sr5 import ARM, JAWS, origin
    joint_details=[]
    for p in stage.Traverse():
        name=p.GetName()
        if name not in ARM+JAWS or not p.IsA(UsdPhysics.Joint):continue
        j=UsdPhysics.Joint(p);src=model.joints[name]
        angular=name in ARM
        api=UsdPhysics.RevoluteJoint(p) if angular else UsdPhysics.PrismaticJoint(p)
        bounds=np.array([api.GetLowerLimitAttr().Get(),api.GetUpperLimitAttr().Get()])
        if angular:bounds=np.radians(bounds)
        expected=np.array([float(src.find('limit').get(k)) for k in ['lower','upper']])
        axis=np.eye(3)['XYZ'.index(api.GetAxisAttr().Get())]
        q=j.GetLocalRot0Attr().Get()
        parent_axis=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).apply(axis)
        expected_axis=origin(src)[:3,:3]@np.fromstring(src.find('axis').get('xyz'),sep=' ')
        joint_details.append({'joint':name,'limit_error_rad_or_m':float(np.max(np.abs(bounds-expected))),
                              'parent_axis_error':float(np.linalg.norm(parent_axis-expected_axis))})
    mass_details=[]
    for name,elem in model.links.items():
        inertial=elem.find('inertial')
        if inertial is None:continue
        mass=UsdPhysics.MassAPI(stage.GetPrimAtPath(link_paths[name]))
        m=float(mass.GetMassAttr().Get())
        c=np.array(mass.GetCenterOfMassAttr().Get())
        d=np.array(mass.GetDiagonalInertiaAttr().Get())
        q=mass.GetPrincipalAxesAttr().Get()
        r=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
        actual=r@np.diag(d)@r.T
        i=inertial.find('inertia')
        expected=np.array([[float(i.get('ixx')),float(i.get('ixy')),float(i.get('ixz'))],
                           [float(i.get('ixy')),float(i.get('iyy')),float(i.get('iyz'))],
                           [float(i.get('ixz')),float(i.get('iyz')),float(i.get('izz'))]])
        it=origin(inertial)
        expected=it[:3,:3]@expected@it[:3,:3].T
        mass_details.append({'link':name,'mass_error_kg':abs(m-float(inertial.find('mass').get('value'))),
                             'com_error_m':float(np.linalg.norm(c-it[:3,3])),
                             'inertia_matrix_max_error_kg_m2':float(np.max(np.abs(actual-expected)))})
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(stage.GetRootLayer().identifier)
    paths=[str(l.realPath) for l in layers]+list(map(str,assets))
    from apfg.robots.sr5 import ROOT
    bundled={str(x):str(ROOT/'third_party/isaacsim/kit/mdl/core/Base'/str(x)) for x in unresolved
             if str(x).endswith('.mdl') and '/' not in str(x)}
    from pathlib import Path
    missing=[str(x) for x in unresolved if str(x) not in bundled or not Path(bundled[str(x)]).is_file()]
    portable=not missing and all(p.startswith(str(ROOT)+'/') for p in paths if p)
    checks={'source_joint_limits':len(joint_details)==8 and all(x['limit_error_rad_or_m']<1e-6 for x in joint_details),
            'source_joint_axes':all(x['parent_axis_error']<1e-6 for x in joint_details),
            'source_mass_com_inertia':all(x['mass_error_kg']<1e-6 and x['com_error_m']<1e-6 and x['inertia_matrix_max_error_kg_m2']<1e-6 for x in mass_details),
            'project_local_dependencies':portable}
    return {'status':'PASS' if all(checks.values()) else 'FAILED','checks':checks,'joints':joint_details,'inertials':mass_details,
            'dependency_count':len(paths),'unresolved':missing,'bundled_mdl':{k:str(Path(v).relative_to(ROOT)) for k,v in bundled.items()}}
