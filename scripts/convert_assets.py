#!/usr/bin/env python3
"""Offline OBJ -> meter-scale USD with textured visuals and mesh collision.

Run using scripts/bundled_python.sh. Estimated inertial parameters are explicit;
this conversion does not certify contact physics or claim physical validation.
"""
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from apfg.download import sha256

ROOT = Path(__file__).resolve().parents[1]


def parse_obj(path):
    vertices, uvs, triangles, texture_indices = [], [], [], []
    material_library = None
    with path.open(errors='strict') as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append(tuple(map(float, fields[1:4])))
            elif fields[0] == 'vt':
                uvs.append(tuple(map(float, fields[1:3])))
            elif fields[0] == 'mtllib':
                material_library = path.parent / ' '.join(fields[1:])
            elif fields[0] == 'f':
                face = [x.split('/') for x in fields[1:]]
                for i in range(1, len(face)-1):
                    for index in (0, i, i+1):
                        v = int(face[index][0])
                        triangles.append(v-1 if v > 0 else len(vertices)+v)
                        t = face[index][1] if len(face[index]) > 1 else ''
                        texture_indices.append(int(t)-1 if t else None)
    if not vertices or not triangles:
        raise ValueError(f'No triangle geometry: {path}')
    texture = None
    if material_library and material_library.exists():
        for line in material_library.read_text().splitlines():
            if line.startswith('map_Kd '):
                texture = material_library.parent / line.split(maxsplit=1)[1].strip()
                break
    return vertices, triangles, uvs, texture_indices, texture


def convert(entry):
    source = ROOT / entry['mesh_path']
    name = entry['asset_id'].split(':', 1)[1]
    vertices, indices, uv, uv_indices, texture = parse_obj(source)
    lo = [min(v[i] for v in vertices) for i in range(3)]
    hi = [max(v[i] for v in vertices) for i in range(3)]
    dims = [hi[i]-lo[i] for i in range(3)]
    if not .001 < max(dims) < 2.0:
        raise ValueError(f'Implausible real-world dimensions {name}: {dims}')
    center = [(hi[i]+lo[i])/2 for i in range(3)]
    output = ROOT / 'assets/converted_usd' / (name + '.usda')
    mass = max(.02, min(2.0, math.prod(dims)*500*.25))
    inertia = [mass * (dims[(i+1)%3]**2 + dims[(i+2)%3]**2)/12 for i in range(3)]
    if output.exists():
        stage = Usd.Stage.Open(str(output))
        old_hash = stage.GetRootLayer().customLayerData.get('source_sha256')
        if old_hash == sha256(source):
            cached = json.loads(output.with_suffix('.json').read_text())
            if cached['mesh_path'] == entry['mesh_path'] and cached['source'] == entry['source']:
                return cached
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Xform.Define(stage, '/Object')
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    m = UsdPhysics.MassAPI.Apply(body.GetPrim())
    m.CreateMassAttr(mass)
    m.CreateCenterOfMassAttr(Gf.Vec3f(0))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1))
    mesh = UsdGeom.Mesh.Define(stage, '/Object/Mesh')
    mesh.CreatePointsAttr([Gf.Vec3f(*[v[i]-center[i] for i in range(3)]) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3]*(len(indices)//3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexDecomposition')
    if uv and all(i is not None for i in uv_indices):
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying')
        st.Set([Gf.Vec2f(*uv[i]) for i in uv_indices])
    mat = UsdShade.Material.Define(stage, '/Object/Material')
    shader = UsdShade.Shader.Define(stage, '/Object/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    if texture and texture.exists():
        tex = UsdShade.Shader.Define(stage, '/Object/Material/Texture')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(os.path.relpath(texture, output.parent)))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        reader = UsdShade.Shader.Define(stage, '/Object/Material/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics_mat.CreateStaticFrictionAttr(.6)
    physics_mat.CreateDynamicFrictionAttr(.5)
    physics_mat.CreateRestitutionAttr(.02)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(mat, materialPurpose='physics')
    stage.GetRootLayer().customLayerData = {'source_sha256': sha256(source), 'physics_provenance': 'estimated; requires independent drop/contact validation'}
    stage.GetRootLayer().Export(str(output))
    record = dict(entry, usd_path=str(output.relative_to(ROOT)), usd_sha256=sha256(output),
                  mesh_sha256=sha256(source), geometry_dimensions_m=dims, scale=1.0,
                  mesh_to_body_translation_m=[-c for c in center], mass_kg=mass,
                  center_of_mass_m=[0., 0., 0.], diagonal_inertia_kg_m2=inertia,
                  principal_axes_wxyz=[1., 0., 0., 0.], static_friction=.6, dynamic_friction=.5,
                  restitution=.02, collision_approximation='convexDecomposition',
                  physical_parameter_source={'status': 'estimated', 'mass': 'bounding volume * fill fraction .25 * density 500 kg/m3; clamped .02..2 kg',
                                             'inertia': 'uniform bounding cuboid approximation; not measured',
                                             'com': 'bounding extent midpoint; not measured', 'friction_restitution': 'M0 defaults; not measured'},
                  collision_review='NOT_RUN: cup/bowl cavities and thin tools require inspection and contact tests',
                  conversion_status='PASS', physics_validation='NOT_RUN')
    output.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    path = ROOT / 'assets/manifests/ycb_downloads.json'
    entries = json.loads(path.read_text()) if path.exists() else []
    entries = [e for e in entries if e['status'] == 'PASS']
    canonical = {e['asset_id'] for e in entries}
    for mesh in sorted((ROOT / 'assets/ycb/local_reuse/YCB').glob('*/textured_simple.obj')):
        asset_id = 'ycb:' + mesh.parent.name
        if asset_id not in canonical:
            entries.append({'asset_id': asset_id, 'dataset': 'YCB', 'source': 'local YCB.zip repack',
                            'url': 'https://ycb-benchmarks.s3.amazonaws.com/index.html',
                            'license': 'UNVERIFIED local repack; canonical YCB CC-BY-4.0',
                            'mesh_path': str(mesh.relative_to(ROOT)), 'provenance_status': 'needs_canonical_comparison'})
    registry = []
    for entry in entries:
        registry.append(convert(entry))
        print(entry['asset_id'], 'converted', flush=True)
    (ROOT / 'assets/manifests/object_registry.json').write_text(json.dumps({'schema_version': 1, 'objects': registry}, indent=2)+'\n')


if __name__ == '__main__':
    main()
