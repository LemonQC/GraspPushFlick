#!/usr/bin/env python3
"""Measure local disk use and sample actual write locations, without deleting data."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/environment_check'


def snapshot():
    original_home = Path(os.environ.get('APFG_ORIGINAL_HOME', str(Path.home())))
    paths = [original_home/p for p in ['.cache/ov', '.cache/nvidia', '.nv', '.nvidia-omniverse', '.local/share/ov', '.config/ov', '.conda']]
    paths.append(Path('/tmp'))
    data = {}
    for root in paths:
        for directory, _, files in os.walk(root):
            for name in files:
                p = Path(directory)/name
                try:
                    s = p.stat()
                    if s.st_uid == os.getuid():
                        data[str(p)] = [s.st_size, s.st_mtime_ns]
                except OSError:
                    continue
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    current = snapshot()
    baseline = OUT/'outside_storage_baseline.json'
    if args.baseline:
        baseline.write_text(json.dumps(current))
        print('Outside write baseline captured; it does not cover earlier operations.')
        return
    previous = json.loads(baseline.read_text()) if baseline.exists() else None
    changed = [p for p, meta in current.items() if previous is not None and previous.get(p) != meta]
    sizes = {}
    for folder in ['.conda', '.runtime_home', 'cache', 'tmp', 'third_party', 'assets', 'outputs']:
        r = subprocess.run(['du','-sb',str(ROOT/folder)],capture_output=True,text=True)
        sizes[folder] = int(r.stdout.split()[0]) if r.returncode==0 else None
    free = shutil.disk_usage(ROOT)
    allocated=subprocess.run(['du','-sB1',str(ROOT)],capture_output=True,text=True,check=True)
    data = {'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'project_usage_bytes': sizes,
            'project_allocated_bytes':int(allocated.stdout.split()[0]),
            'free_bytes': free.free, 'free_inodes': os.statvfs(ROOT).f_favail,
            'outside_scope_changed_files': changed, 'outside_scope_changed_bytes': sum(current[p][0] for p in changed),
            'scope': 'Observed changes in current-user files under known external cache paths and /tmp since saved baseline; concurrent user activity cannot be attributed. Not a whole-filesystem syscall audit.',
            'status': 'PASS' if previous is not None and not changed and free.free > 20*2**30 else 'NOT_RUN' if previous is None else 'FAILED'}
    (OUT/'storage_audit.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'status': data['status'], 'project_gib': round(sum(x or 0 for x in sizes.values())/2**30,2),
                      'free_gib': round(free.free/2**30,2),'outside_changed_count': len(changed)}))


if __name__ == '__main__':
    main()
