#!/usr/bin/env python3
"""Find the latest real v2 recording; VP8 playback supports the remote desktop."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',nargs='?',choices=['push','flip','active','dense'],default='push');p.add_argument('--list',action='store_true');p.add_argument('--camera',choices=['third','fixed','wrist'],default='third');p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-id',help='Select an explicitly named run, including failed attempts');a=p.parse_args()
    path=ROOT/'evidence/clutter_v2/video_index.json'
    if not path.exists():p.error('Run scripts/collect_clutter_v2_evidence.py after a completed campaign')
    rows=json.loads(path.read_text())
    if a.list:
        for r in rows:print(r['run_id'],r['episode'],r['family'],r['requested'],r['status'],'retrieval='+r['retrieval'])
        return 0
    want={'push':'PUSH','flip':'FLIP','active':'ACTIVE','dense':'INSPECT'}[a.kind]
    matched=[r for r in rows if r['requested']==want and a.camera in r['videos'] and (not a.run_id or r['run_id']==a.run_id)]
    if not matched:p.error('No actual '+want+' attempt video')
    r=matched[-1];print('Actual recorded attempt:',r['run_id'],r['episode'],r['family'],r['status'],'retrieval='+r['retrieval'],flush=True)
    command=['bash',str(ROOT/'scripts/play_video.sh'),str(ROOT/r['videos'][a.camera])]
    if a.prepare_only:command.append('--prepare-only')
    return subprocess.call(command)

if __name__=='__main__':raise SystemExit(main())
