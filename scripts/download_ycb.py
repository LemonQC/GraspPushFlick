#!/usr/bin/env python3
"""Download only official textured object meshes, never RGBD training scenes."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

from apfg.download import download, extract_tar

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = '''002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can
006_mustard_bottle 007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can
011_banana 019_pitcher_base 021_bleach_cleanser 024_bowl 025_mug 035_power_drill
036_wood_block 037_scissors 040_large_marker 042_adjustable_wrench 044_flat_screwdriver
048_hammer 051_large_clamp 052_extra_large_clamp 061_foam_brick 065-c_cups 072-a_toy_airplane
012_strawberry 013_apple 014_lemon 015_peach'''.split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    index_url = 'https://ycb-benchmarks.s3.amazonaws.com/index.html'
    with urllib.request.urlopen(index_url, timeout=40) as r:
        page = r.read().decode()
    def fetch(name):
        rel = f'data/google/{name}_google_16k.tgz'
        entry = {'asset_id': f'ycb:{name}', 'dataset': 'YCB', 'source': index_url,
                 'license': 'CC-BY-4.0', 'scale': 1.0, 'units': 'meters', 'status': 'NOT_RUN'}
        try:
            if rel not in page:
                # Some objects have no Google scan. Use original Berkeley meshes.
                rel = f'data/berkeley/{name}/{name}_berkeley_meshes.tgz'
            if rel not in page:
                raise RuntimeError('No official model link for requested object')
            url = 'https://ycb-benchmarks.s3.amazonaws.com/' + rel
            archive = ROOT / 'assets/ycb/archives' / Path(rel).name
            receipt = download(url, archive)
            target = ROOT / 'assets/ycb/models'
            stamp = ROOT / 'assets/ycb' / (name + '.verified.json')
            if not stamp.exists() or json.loads(stamp.read_text()).get('sha256') != receipt['sha256']:
                extract_tar(archive, target)
                stamp.write_text(json.dumps(receipt, indent=2))
            meshes = sorted((target / name).rglob('*.obj'))
            if not meshes:
                raise RuntimeError('Archive contains no OBJ mesh')
            preferred = sorted(meshes, key=lambda p: ('textured' not in p.name, len(str(p))))[0]
            entry.update(receipt, mesh_path=str(preferred.relative_to(ROOT)), status='PASS')
        except Exception as e:
            entry.update(status='FAILED', error=str(e))
        print(name, entry['status'], entry.get('error', ''), flush=True)
        return entry
    objects = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fetch, name) for name in OBJECTS[:args.limit]]):
            objects.append(future.result())
            objects.sort(key=lambda x: x['asset_id'])
            (ROOT / 'assets/manifests/ycb_downloads.json').write_text(json.dumps(objects, indent=2) + '\n')
    if sum(x['status'] == 'PASS' for x in objects) < min(12, args.limit):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
