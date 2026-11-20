# Reproducibility

Use the repository root as PROJECT_ROOT. Source `scripts/project_env.sh` before commands; this redirects package caches and temporary files without changing the parent HOME. Installation and simulator children use a private project HOME when appropriate. Never set DISPLAY to an assumed value; launch GUI from the existing desktop session.

The requested 5.1/2.3.1 stack is incompatible with this host. The recorded fallback uses a copied Isaac Sim 4.5 binary distribution, Isaac Lab tag v2.0.2 at `b5fa0eb031a2413c182eeb54fa3a9295e8fd867c`, Python 3.10, torch 2.5.1/cu118, torchvision 0.20.1, and PhysX. Exact installed versions, the binary VERSION string, pip freeze, commands and raw logs remain under `outputs/environment_check/`. Consult `configs/constraints.txt` and `configs/versions.json`.

Run `scripts/check_system.sh`, `python scripts/check_cuda.py`, then `scripts/setup_isaac.sh`. A reused official binary/asset distribution can be supplied with APFG_ISAAC_BINARY_SOURCE / APFG_ISAAC_ASSET_SOURCE. Copying does not establish the original publisher checksum; local hashes and provenance are recorded separately.

Download YCB using `scripts/download_ycb.sh`; convert with `scripts/bundled_python.sh scripts/convert_assets.py`; audit dependencies with `scripts/bundled_python.sh scripts/check_asset_integrity.py`. Official URLs, licenses, hashes, dimensions, mesh transforms, collision method and estimated physics are in `assets/manifests/`. Do not redistribute large third-party assets in Git.

GUI: `bash scripts/run_gui.sh --test clutter --scene s1 --seed 20261120 --interactive`. Change s1 to s2 for the dropped pile. Headless: `bash scripts/run_headless.sh --test clutter --scene s1 --seed 20261120`. Both use the same AppLauncher, renderer-enabled camera configuration, simulation and recording code. Diagnostic options include --test physics/robot/camera/objects, --asset-id, --steps, --width, --height and --fps. Default output is 1280×720 at 30 FPS, physics 120 Hz. Smaller resolutions may be used explicitly.

Each run gets a UTC timestamp plus random suffix. Sensor videos, measured simulation/wall times, joint/EE/object states, contact forces, RGB-D, calibration, segmentation and target camera-frame point cloud stay local under `outputs/environment_check/runs/<run_id>/`. A final-frame multiview PNG is derived from the actual camera frames. Videos are decoded fully before acceptance; nominal FPS is distinct from actual wall capture throughput.

Current CUDA failure blocks real recording. There are no successful videos/keyframes to review. `evidence/` contains sanitized compact status and artifact hashes only; it does not imply that full videos can be downloaded from GitHub. `PASS`, `FAILED`, `BLOCKED`, and `NOT_RUN` must be read independently. `summarize_m0.py` exits 2 until all required gates have actual PASS results. `run_m0.sh` stops at failed prerequisites and cannot print unconditional READY.

CPU-only checks: source the environment, run `python -m unittest discover -s tests -v`, `python -m compileall -q apfg scripts`, and `bash -n` on shell scripts. These checks do not exercise the simulator. Run the full M0 after the system CUDA/UVM state is repaired; all unexecuted simulation paths remain unvalidated until then.
