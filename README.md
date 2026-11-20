# GraspPushFlick / TRO-APFG

面向杂乱三维环境的目标取出研究：Active 感知、Push 推动、Flip 物体真实三维姿态改变、6-DoF Grasp 与完整取出。

当前 M0 **BLOCKED，尚未 READY**。已准备项目内安装环境、30 种官方 YCB 模型、USD 转换/注册表及共享 GUI/Headless 诊断代码。动态验证受系统 CUDA/UVM 故障阻塞：`cuInit(0)=999`，打开 `/dev/nvidia-uvm` 返回 `EIO`。没有通过物理验证的物体，也没有真实仿真视频或成功关键帧。

用户确认实际 GPU 为 RTX 3090。当前 Ubuntu 20.04 / GLIBC 2.31 / 驱动 535.230.02 不符合原定 Isaac Sim 5.1 pip 方案；安装替代组合：本机复用 Isaac Sim 4.5 二进制、Isaac Lab v2.0.2、Python 3.10、PyTorch 2.5.1/cu118、TorchVision 0.20.1。实际完整版本及检查见本地 `outputs/environment_check/`，精简证据见 `evidence/`。

```bash
cd /data/GraspPushFlicks
bash scripts/check_system.sh
source scripts/project_env.sh
python scripts/check_cuda.py

# CUDA 恢复后；两种模式共享场景、控制、相机和视频代码。
bash scripts/run_gui.sh --test clutter --scene s1 --interactive
bash scripts/run_gui.sh --test clutter --scene s2 --interactive
bash scripts/run_headless.sh --test physics
bash scripts/run_robot_smoke.sh
bash scripts/run_camera_smoke.sh
bash scripts/run_object_smoke.sh
bash scripts/run_m0.sh
```

GUI 从已有桌面会话继承 `DISPLAY` 和 `XAUTHORITY`，通过向日葵观察；不硬编码显示编号。GUI 可旋转视角、暂停和正常退出；场景通过启动参数切换。默认保存 1280×720、30 FPS 的真实相机帧，生成第三人称、固定俯视和腕部视角，保留深度、标定、分割、点云和时间戳。每次运行写入独立 `outputs/environment_check/runs/<run_id>/`；当前驱动故障时入口会以退出码 2 记录 BLOCKED，不生成虚假视频。

所有大文件位于项目内部，环境为 `.conda/isaac-apfg`。`project_env.sh` 不修改调用者 HOME、桌面认证或 `.bashrc`；仅仿真子进程使用 `.runtime_home`。安装及下载可重复执行，已校验文件自动跳过。已下载的原始资产与完整视频不提交 Git。

详细说明：[环境](docs/ENVIRONMENT_SETUP_ZH.md)、[资产](docs/ASSET_MANIFEST_ZH.md)、[M0 报告](docs/M0_REPORT_ZH.md)、[复现](docs/REPRODUCIBILITY.md)、[研究范围](docs/BASELINE_SCOPE_ZH.md)。
