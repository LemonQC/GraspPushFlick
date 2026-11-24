#!/usr/bin/env python3
"""Audit actual baseline transitions and the complete physical episode ledger."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apfg.baselines.data_audit import audit_dataset,project_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/baselines/dataset/manifest.jsonl')
    p.add_argument('--episodes-manifest',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/baselines/dataset/audit.json')
    p.add_argument('--publish-evidence',action='store_true');args=p.parse_args()
    result=audit_dataset(args.manifest,args.episodes_manifest)
    output=project_path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(result,indent=2)+'\n'
    archive=output.parent/'audits'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]+'.json')
    archive.parent.mkdir(parents=True,exist_ok=True);archive.write_text(text);output.write_text(text)
    if args.publish_evidence:
        public=ROOT/'evidence/baselines/dataset_audit.json';public.parent.mkdir(parents=True,exist_ok=True);public.write_text(text)
    print(json.dumps({'status':result['status'],'data_readiness':result['data_readiness'],'counts':result['counts'],
        'errors':result['errors'],'warnings':result['warnings'],'output':str(output.relative_to(ROOT))}),flush=True)
    return 1 if result['status']=='FAILED' else 0


if __name__=='__main__':raise SystemExit(main())
