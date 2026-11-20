#!/usr/bin/env python3
"""Resolve USD layers/textures locally without opening Kit or contacting a server."""
import json
from pathlib import Path
from pxr import Sdf, Usd, UsdUtils

from apfg.download import sha256

ROOT = Path(__file__).resolve().parents[1]


def check(path):
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetDefaultPrim():
        return {'path': str(path.relative_to(ROOT)), 'status': 'FAILED', 'reason': 'missing stage/default prim'}
    layers, assets, missing = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    # Kit supplies these built-in MDL modules from the copied binary distribution.
    builtins = {m: ROOT/'third_party/isaacsim/kit/mdl/core/Base'/m for m in set(missing) if m.endswith('.mdl') and '/' not in m}
    builtin_ok = {m: str(p.relative_to(ROOT)) for m, p in builtins.items() if p.is_file()}
    unresolved = sorted(set(missing) - set(builtin_ok))
    files = [Path(l.realPath) for l in layers if l.realPath] + [Path(p) for p in assets]
    external = [str(p) for p in files if not p.resolve().is_relative_to(ROOT)]
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
            'status': 'PASS' if not unresolved and not external else 'FAILED',
            'usd_layers': [str(Path(l.realPath).relative_to(ROOT)) for l in layers if l.realPath and Path(l.realPath).is_relative_to(ROOT)],
            'dependency_count': len(files), 'unresolved': unresolved, 'external': external,
            'bundled_mdl': builtin_ok, 'scope': 'static USD dependency audit; MDL imports use Kit search path; actual offline rendering NOT_RUN'}


def main():
    targets = [ROOT/'assets/isaac/Isaac/Robots/Franka/franka.usd']
    targets += sorted((ROOT/'assets/isaac/Isaac/Props/YCB/Axis_Aligned').glob('*.usd'))
    targets += sorted((ROOT/'assets/converted_usd').glob('*.usda'))
    results = [check(p) if p.exists() else {'path': str(p.relative_to(ROOT)), 'status': 'FAILED', 'reason': 'missing file'} for p in targets]
    data = {'status': 'PASS' if results and all(x['status']=='PASS' for x in results) else 'FAILED',
            'static_dependency_check': results, 'offline_rendering': 'NOT_RUN'}
    out = ROOT/'outputs/environment_check/offline_asset_check.json'
    out.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'status': data['status'], 'assets': len(results), 'failed': [x['path'] for x in results if x['status']!='PASS']}))
    if data['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
