"""Read-only kinematics and integrity checks for the supplied SR5 assembly.

URDF is the sole source of axes, limits and fixed transforms. All public poses
use metres, radians, right-handed world Z up, quaternion XYZW.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'assets/imported_sr5/isaac_sr5_export'
ARM = [f'joint{i}' for i in range(1, 7)]
JAWS = ['ral_left_jaw', 'ral_right_jaw']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin(element):
    o = element.find('origin')
    t = np.eye(4)
    if o is not None:
        t[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def nearest_revolute_equivalent(q,seed,lower,upper):
    """Choose limit-valid 2pi representatives without changing the FK pose."""
    q,seed,lower,upper=map(lambda a:np.asarray(a,float),(q,seed,lower,upper))
    if not (q.shape==seed.shape==lower.shape==upper.shape) or not np.isfinite(np.r_[q,seed,lower,upper]).all():
        raise ValueError('Finite joint vectors with matching shapes required')
    result=q.copy()
    for i,value in enumerate(q):
        first=int(np.ceil((lower[i]-value)/(2*np.pi)-1e-10))
        last=int(np.floor((upper[i]-value)/(2*np.pi)+1e-10))
        if first>last:raise ValueError('No equivalent joint angle within source limits')
        k=int(np.clip(np.rint((seed[i]-value)/(2*np.pi)),first,last))
        result[i]=np.clip(value+2*np.pi*k,lower[i],upper[i])
    return result


class SR5Model:
    def __init__(self, snapshot=SNAPSHOT):
        self.snapshot = Path(snapshot).resolve()
        self.urdf = self.snapshot / 'robot/assembled_portable.urdf'
        self.xml = ET.parse(self.urdf).getroot()
        self.config = json.loads((self.snapshot/'sr5_tool_current.json').read_text())
        self.assembly = json.loads((self.snapshot/'assembly_current.json').read_text())
        self.export = json.loads((self.snapshot/'export_report.json').read_text())
        self.links = {x.get('name'): x for x in self.xml.findall('link')}
        self.joints = {x.get('name'): x for x in self.xml.findall('joint')}
        self.lower = np.array([float(self.joints[n].find('limit').get('lower')) for n in ARM])
        self.upper = np.array([float(self.joints[n].find('limit').get('upper')) for n in ARM])
        self.efforts = np.array([float(self.joints[n].find('limit').get('effort')) for n in ARM])
        self.base = np.asarray(self.config['base_xyz'], float)

    def fk(self, q, opening=.08, base=None):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Six finite joint angles required')
        values = dict(zip(ARM, q))
        values.update(dict.fromkeys(JAWS, opening/2))
        transforms = {'world': np.eye(4)}
        transforms['world'][:3, 3] = self.base if base is None else base
        remaining = dict(self.joints)
        while remaining:
            progress = False
            for name, j in list(remaining.items()):
                parent, child = j.find('parent').get('link'), j.find('child').get('link')
                if parent not in transforms:
                    continue
                t = origin(j)
                axis_element = j.find('axis')
                axis = np.array([1., 0., 0.]) if axis_element is None else np.fromstring(axis_element.get('xyz'), sep=' ')
                motion = np.eye(4)
                if j.get('type') in ('revolute', 'continuous'):
                    motion[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
                elif j.get('type') == 'prismatic':
                    motion[:3, 3] = axis*values[name]
                transforms[child] = transforms[parent] @ t @ motion
                del remaining[name]
                progress = True
            if not progress:
                raise ValueError('URDF has unresolved/disconnected links')
        return transforms

    def ik(self, xyz, quat_xyzw, seed, base=None):
        xyz, quat, seed = np.asarray(xyz, float), np.asarray(quat_xyzw, float), np.asarray(seed, float)
        if xyz.shape != (3,) or quat.shape != (4,) or seed.shape != (6,) or not np.isfinite(np.r_[xyz, quat, seed]).all() or np.linalg.norm(quat) < 1e-8:
            raise ValueError('Invalid finite TCP pose or joint seed')
        desired = Rotation.from_quat(quat).as_matrix()
        def residual(q):
            t = self.fk(q, base=base)['ral_tcp']
            return np.r_[t[:3, 3]-xyz, .25*Rotation.from_matrix(desired @ t[:3, :3].T).as_rotvec()]
        result = least_squares(residual, np.clip(seed, self.lower+1e-8, self.upper-1e-8), bounds=(self.lower, self.upper),
                               max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        err = residual(result.x)
        if np.linalg.norm(err[:3]) > .001 or np.linalg.norm(err[3:])/.25 > .01:
            raise ValueError(f'IK_UNREACHABLE: position={np.linalg.norm(err[:3]):.6f}m orientation={np.linalg.norm(err[3:])/.25:.6f}rad')
        return nearest_revolute_equivalent(result.x,seed,self.lower,self.upper)

    def audit(self):
        checks, meshes = {}, []
        checks['urdf_sha256'] = digest(self.urdf) == self.export['robot_urdf_sha256']
        for data in self.export['meshes'].values():
            p = self.urdf.parent/data['path']
            meshes.append({'path':str(p.relative_to(ROOT)), 'sha256':digest(p) if p.is_file() else None,
                           'expected_sha256':data['sha256'], 'bytes':p.stat().st_size if p.is_file() else 0})
        checks['mesh_hashes'] = all(x['sha256']==x['expected_sha256'] for x in meshes)
        checks['relative_mesh_paths'] = all(not Path(m.get('filename')).is_absolute() and (self.urdf.parent/m.get('filename')).is_file() for m in self.xml.iter('mesh'))
        checks['six_vendor_revolute'] = [n for n,j in self.joints.items() if j.get('type')=='revolute'] == ARM == self.assembly['arm_joints']
        checks['two_jaw_prismatic'] = [n for n,j in self.joints.items() if j.get('type')=='prismatic'] == JAWS
        checks['jaw_limits'] = all(float(self.joints[n].find('limit').get('lower'))==0 and float(self.joints[n].find('limit').get('upper'))==.04 for n in JAWS)
        checks['config_consistent'] = self.config == self.assembly['tool_config']
        checks['flange'] = self.joints['ral_plate_fixed'].find('parent').get('link') == self.assembly['flange']
        f = self.fk(np.zeros(6))
        jaw = f['ral_gripper'][:3, :3] @ np.array([1., 0., 0.])
        camera = f['ral_camera'][:3, :3] @ np.array([0., 1., 0.])
        cosine = abs(float(jaw@camera))
        checks['jaw_parallel_camera_long_edge'] = cosine > .999999 and self.export['jaw_camera_alignment_status']=='PARALLEL'
        plate_mesh = self.links['ral_plate'].find('visual/geometry/mesh')
        plate_path = self.urdf.parent/plate_mesh.get('filename')
        data = plate_path.read_bytes()
        tri = np.frombuffer(data, dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]), offset=84)
        vertices = tri['vertices'].reshape(-1,3)
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        checks['plate_meter_scale'] = bool(np.allclose(bounds[1]-bounds[0], [.14898632,.007,.10999612], atol=1e-5)) and plate_mesh.get('scale','1 1 1')=='1 1 1'
        checks['plate_hash'] = digest(plate_path)==self.assembly['plate_sha256']
        return {'status':'PASS' if all(checks.values()) else 'FAILED', 'checks':checks,
                'source_urdf':str(self.urdf.relative_to(ROOT)), 'source_urdf_sha256':digest(self.urdf),
                'meshes':list({m['path']:m for m in meshes}.values()), 'mesh_reference_count':len(meshes),
                'jaw_camera_long_axis_abs_cosine':cosine, 'plate_source_bounds_m':bounds.tolist(),
                'flange':self.assembly['flange'], 'arm_joints':ARM, 'jaw_joints':JAWS,
                'assumption_status':{'arm':'SOURCE_SNAPSHOT: vendor geometry/inertia/axes retained; not real-world recalibrated',
                    'plate':'REAL_CAD_APPROX_MOUNT: source mesh-local transform retained once',
                    'gripper':'APPROX: source box housing and fingers, not manufacturer CAD',
                    'camera':'APPROX: source box housing and uncalibrated optical extrinsics, 58deg vertical FOV',
                    'tcp':'APPROX: source fixed frame, not measured hardware TCP',
                    'drives_friction':'ISAAC_SIM_TUNED_NOT_HARDWARE_CALIBRATION'}}
