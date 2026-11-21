#!/usr/bin/env python3
"""Audit strace -yy write destinations for the explicitly traced validation span."""
import datetime as dt
import json
from pathlib import Path
import re
import argparse
import stat

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--trace',default='outputs/environment_check/storage_syscalls.log')
args=parser.parse_args()
trace=ROOT/args.trace
inside=set();outside=set();shared_memory=set();thread_names=set();devices=set();count=0;ipc_events=0
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
        value=value.split('<',1)[0]
        try:
            if stat.S_ISCHR(Path(value).stat().st_mode):
                devices.add(value)
                continue
        except OSError:pass
        if re.fullmatch(r'/proc/\d+/task/\d+/comm',value):
            thread_names.add(value)
            continue
        if re.fullmatch(r'/run/user/\d+/omni\.telemetry\.transmitter\.exit',value) and Path(value).exists() and Path(value).stat().st_size==0:
            shared_memory.add(value)
            ipc_events+=1
            continue
        p=Path(value)
        if not p.is_absolute():p=ROOT/p
        count+=1
        if value.startswith(('/dev/shm/','/memfd:')):
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
        'kernel_thread_name_paths':sorted(thread_names),'character_devices':sorted(devices),
        'shared_memory_note':'RAM-backed POSIX shared memory, memfd and zero-byte runtime control markers are recorded separately. Kernel thread naming and character-device IO are also separate from persistent disk writes. No shared IPC files were deleted.',
        'trace_path':str(trace.relative_to(ROOT))}
(ROOT/'outputs/environment_check/process_write_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','filesystem_write_events','outside_write_paths']}))
raise SystemExit(0 if result['status']=='PASS' else 1)
