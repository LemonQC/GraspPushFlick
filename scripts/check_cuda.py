#!/usr/bin/env python3
"""Probe the system driver without importing PyTorch or starting a renderer."""
import ctypes
import datetime as dt
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
result={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
try:
    driver=ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
    result['cuInit_returncode']=driver.cuInit(0)
    count=ctypes.c_int()
    result['cuDeviceGetCount_returncode']=driver.cuDeviceGetCount(ctypes.byref(count))
    result['device_count']=count.value
except OSError as e:
    result['load_error']=str(e)
try:
    fd=os.open('/dev/nvidia-uvm',os.O_RDWR)
    os.close(fd)
    result['uvm_open_status']='PASS'
except OSError as e:
    result.update(uvm_open_status='FAILED',uvm_errno=e.errno,uvm_error=e.strerror)
result['status']='PASS' if result.get('cuInit_returncode')==0 and result.get('device_count',0)>0 else 'FAILED'
(ROOT/'outputs/environment_check/cuda_driver_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(0 if result['status']=='PASS' else 2)
