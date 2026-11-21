#!/usr/bin/env python3
"""Import the exact portable SR5 URDF with Isaac Sim 4.5's bundled importer."""
import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apfg.robots.sr5 import SR5Model, ARM, JAWS, digest
from apfg.simulation.launcher import add_launcher_args, launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(p)
    args = p.parse_args()
    run = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    root = ROOT/'outputs/isaac_sr5_hardware'
    out = root/'imports'/run
    out.mkdir(parents=True)
    model = SR5Model()
    audit = model.audit()
    app = None
    try:
        if audit['status']!='PASS':
            raise ValueError('Source assembly integrity/alignment failed')
        app = launch(args)
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.asset.importer.urdf')
        app.update()
        ok, config = omni.kit.commands.execute('URDFCreateImportConfig')
        if not ok:
            raise RuntimeError('URDFCreateImportConfig failed')
        config.set_merge_fixed_joints(False)
        config.set_import_inertia_tensor(True)
        config.set_fix_base(True)
        config.set_self_collision(True)
        config.set_convex_decomp(False)
        config.set_make_default_prim(True)
        config.set_parse_mimic(False)
        config.set_distance_scale(1.)
        config.set_default_drive_type(1)
        config.set_default_drive_strength(20000.)
        config.set_default_position_drive_damping(500.)
        usd = out/'robot_sr5_pgia_d435i.usd'
        # 4.5 importer uses STL basenames as USD identifiers without sanitizing
        # leading digits. Change ONLY dependency filenames in an import copy.
        # The authoritative source, axes, origins, inertias and scales stay intact.
        source_tree = ET.parse(model.urdf)
        input_dir = out/'import_input'
        (input_dir/'meshes').mkdir(parents=True)
        path_map = {}
        for mesh in source_tree.getroot().iter('mesh'):
            name = mesh.get('filename')
            target = 'meshes/m_'+Path(name).name
            if name not in path_map:
                shutil.copy2(model.urdf.parent/name, input_dir/target)
                path_map[name] = target
            mesh.set('filename', target)
        import_urdf = input_dir/'assembled_import_identifiers.urdf'
        source_tree.write(import_urdf, encoding='utf-8', xml_declaration=True)
        audit['importer_filename_workaround'] = {'reason':'4.5 importer rejects digit-leading hashed STL basenames as USD prim identifiers',
            'filename_only_map':path_map, 'original_urdf_unchanged':digest(model.urdf)==model.export['robot_urdf_sha256'],
            'import_copy_sha256':digest(import_urdf)}
        ok, result = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(import_urdf), import_config=config, dest_path=str(usd))
        if not ok or not usd.is_file():
            raise RuntimeError(f'URDF import failed: {ok} {result}')
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
        stage = Usd.Stage.Open(str(usd))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        paths = {}
        for prim in stage.Traverse():
            if prim.GetName() in model.links:
                paths[prim.GetName()] = str(prim.GetPath())
            if prim.GetName() in ARM+JAWS and prim.IsA(UsdPhysics.Joint):
                angular = prim.GetName() in ARM
                drive = UsdPhysics.DriveAPI.Apply(prim, 'angular' if angular else 'linear')
                drive.CreateTypeAttr('force')
                drive.CreateStiffnessAttr(20000. if angular else 8000.)
                drive.CreateDampingAttr(500. if angular else 100.)
                drive.CreateMaxForceAttr(float(model.efforts[ARM.index(prim.GetName())]) if angular else model.config['finger_force_n'])
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr(0.)
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
        checks = {'single_articulation':len(roots)==1, 'eight_dofs':len(joints)==8,
                  'all_source_links_retained':set(model.links)<=set(paths),
                  'names':{p.GetName() for p in joints}==set(ARM+JAWS)}
        # Keep self and external collision enabled. PhysX joint-neighbour
        # collision defaults apply; no arbitrary robot/object exclusions.
        sensor = UsdGeom.Camera.Define(stage, paths['ral_optical']+'/D435iSensor')
        sensor.AddOrientOp().Set(Gf.Quatf(0.,1.,0.,0.))  # ROS optical -> native USD camera
        sensor.CreateFocalLengthAttr(20.)
        import math
        vertical=40.*math.tan(math.radians(model.config['camera_fov_vertical_deg'])/2)
        sensor.CreateVerticalApertureAttr(vertical)
        sensor.CreateHorizontalApertureAttr(vertical*640/480)
        sensor.CreateClippingRangeAttr(Gf.Vec2f(.005,20.))
        sensor.GetPrim().SetCustomDataByKey('calibration_status','APPROX: source 58 degree vertical FOV, not RealSense hardware calibration')
        stage.GetRootLayer().customLayerData = {'source_urdf_sha256':digest(model.urdf), 'hardware_geometry':'SR5 source + APPROX PGIA/D435i, real plate CAD', 'self_collision':'enabled; joint-neighbour PhysX defaults; external collisions retained'}
        stage.GetRootLayer().Save()
        from apfg.robots.sr5_usd import audit_import
        audit['source_to_usd']=audit_import(stage,model,paths)
        checks['source_to_usd']=audit['source_to_usd']['status']=='PASS'
        audit.update(usd_path=str(usd.relative_to(ROOT)), usd_sha256=digest(usd), link_paths=paths,
                     articulation_roots=roots, importer_checks=checks,
                     importer={'extension':'isaacsim.asset.importer.urdf 2.3.10','fixed_base':True,'merge_fixed_joints':False,'self_collision':True,'convex_decomp':False,'collision_mesh':'source mesh convex hull; box primitives unchanged','inertia':'source tensor imported',
                               'arm_kp':20000,'arm_kd':500,'jaw_kp':8000,'jaw_kd':100,'jaw_force_limit_n':model.config['finger_force_n'],
                               'joint_friction':0,'note':'Isaac drive tuning differs from PyBullet damping; source URDF unchanged'},
                     status='PASS' if all(checks.values()) else 'FAILED')
        for name in ['robot_sr5_pgia_d435i.usd']:
            link = root/name
            if link.is_symlink(): link.unlink()
            if not link.exists(): link.symlink_to(os.path.relpath(usd, root))
        snapshot = out/'source_config';snapshot.mkdir()
        for name in ['assembly_current.json','sr5_tool_current.json','export_report.json']:
            (snapshot/name).write_bytes((model.snapshot/name).read_bytes())
    except Exception as exc:
        audit.update(status='FAILED', reason=str(exc))
        (out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        (out/'assembly_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        (root/'assembly_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({'status':audit['status'],'output':str(out),'reason':audit.get('reason')}),flush=True)
        if app: app.close()
    return 0 if audit['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
