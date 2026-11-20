"""Read-only host checks. Raw host details stay in ignored local outputs."""
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/environment_check"


def command(args, timeout=40):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"command": args, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": args, "returncode": None, "error": str(e)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    free = shutil.disk_usage(ROOT)
    stat = os.statvfs(ROOT)
    checks = {}
    for name, args in {
        "uname": ["uname", "-a"], "os": ["lsb_release", "-a"], "glibc": ["ldd", "--version"],
        "gpu": ["nvidia-smi"], "gpu_csv": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "memory": ["free", "-h"], "disk": ["df", "-h", str(ROOT)], "inodes": ["df", "-i", str(ROOT)],
        "conda_info": ["conda", "info", "--json"], "conda_envs": ["conda", "env", "list", "--json"],
        "x11": ["xdpyinfo"],
    }.items():
        checks[name] = command(args)
    for name, args in {"vulkan": ["vulkaninfo", "--summary"], "opengl": ["glxinfo", "-B"]}.items():
        checks[name] = command(args) if shutil.which(args[0]) else {"status": "NOT_RUN", "reason": "diagnostic tool unavailable"}
    glibc = libc.gnu_get_libc_version().decode()
    data = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "project_root": str(ROOT),
        "platform": platform.platform(), "glibc": glibc,
        "session": {k: os.environ.get(k) for k in ("DISPLAY", "XDG_SESSION_TYPE", "XAUTHORITY")},
        "storage": {"free_bytes": free.free, "total_bytes": free.total, "free_inodes": stat.f_favail,
                    "safety_reserve_bytes": 20 * 2**30, "provisional_peak_budget_bytes": 150 * 2**30,
                    "status": "PASS" if free.free > 170 * 2**30 else "BLOCKED"},
        "requested_stack": {"status": "BLOCKED" if tuple(map(int, glibc.split('.'))) < (2, 35) else "NOT_RUN",
                            "reason": "Isaac Sim 5.1 pip requires GLIBC >=2.35; driver and OS must also meet official requirements"},
        "checks": checks,
    }
    (OUT / "system_preflight.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    (OUT / "gpu_check.json").write_text(json.dumps({"status": "PASS" if checks["gpu"]["returncode"] == 0 else "FAILED",
          "inventory": checks["gpu_csv"], "rtx_rendering": "NOT_RUN", "note": "nvidia-smi alone does not prove rendering"}, indent=2) + "\n")
    print(json.dumps({"glibc": glibc, "gpu": checks["gpu_csv"].get("stdout"), "free_gib": round(free.free / 2**30, 1),
                      "requested_stack": data["requested_stack"]["status"], "report": str(OUT / 'system_preflight.json')}))


if __name__ == "__main__":
    main()
