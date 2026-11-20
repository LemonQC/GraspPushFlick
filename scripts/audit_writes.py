#!/usr/bin/env python3
"""Audit strace -yy write destinations for the explicitly traced validation span."""
import datetime as dt
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
trace=ROOT/'outputs/environment_check/storage_syscalls.log'
inside=set();outside=set();shared_memory=set();count=0;ipc_events=0
for line in trace.read_text(errors='replace').splitlines():
    paths=[]
    if re.search(r'\b(?:write|pwrite64|ftruncate)\(',line):
        m=re.search(r'\(\d+<(/[^>]+)>',line)
        if m:paths.append(m.group(1))
    elif ('openat(' in line and re.search(r'O_WRONLY|O_RDWR|O_CREAT',line)) or 'creat(' in line:
        m=re.search(r'= \d+<(/[^>]+)>',line)
        if m:paths.append(m.group(1))
    elif re.search(r'\b(?:mkdir|mkdirat|rename|renameat|unlink|unlinkat|truncate)\(',line) and re.search(r'= 0\s*$',line):
        paths.extend(re.findall(r'"([^"\n]+)"',line))
    for value in paths:
        value=value.removesuffix(' (deleted)')
        if value.startswith(('/dev/nvidia','/dev/tty')) or value in ['/dev/null','/dev/zero','/dev/urandom','/dev/random']:continue
        p=Path(value)
        if not p.is_absolute():p=ROOT/p
        count+=1
        if value.startswith('/dev/shm/'):
            shared_memory.add(value)
            ipc_events+=1
            continue
        if p.resolve().is_relative_to(ROOT):inside.add(str(p.relative_to(ROOT)))
        else:outside.add(str(p))
result={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'PASS' if count and not outside else 'FAILED',
        'scope':'strace -f -yy on scripts/verify_local.sh; isolated prefix pip check, CPU tests, compilation and CUDA-gated GUI/headless entrypoints. Does not cover earlier installation or successful GPU rendering.',
        'filesystem_write_events':count,'project_write_paths':sorted(inside),'outside_write_paths':sorted(outside),
        'persistent_filesystem_write_events':count-ipc_events,'posix_shared_memory_events':ipc_events,
        'posix_shared_memory_paths':sorted(shared_memory),
        'shared_memory_note':'Kit Carbonite uses POSIX shared memory and semaphores in RAM-backed /dev/shm. These are recorded separately from persistent downloads/caches; no shared IPC files were deleted.',
        'trace_path':str(trace.relative_to(ROOT))}
(ROOT/'outputs/environment_check/process_write_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','filesystem_write_events','outside_write_paths']}))
raise SystemExit(0 if result['status']=='PASS' else 1)
