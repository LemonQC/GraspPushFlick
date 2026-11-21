# 环境安装与实际验证

项目 `/data/GraspPushFlicks`；实际 GPU RTX 3090 24 GiB、驱动 535.230.02、Ubuntu 20.04.6、GLIBC 2.31。继承用户实际 X11 桌面的 DISPLAY 和 XAUTHORITY，没有修改显示管理器、访问控制、用户 .bashrc 或向日葵会话。

原定 Isaac Sim 5.1 pip 安装要求 GLIBC ≥2.35，系统矩阵亦要求更新的 OS/驱动，故本机未强装该版本。参考 [官方安装说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) 和 [要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。

实际使用本机已有官方 Isaac Sim 4.5 二进制副本，完整 VERSION 为 `4.5.0-rc.36+release.19112.f59b3005.gl`。Isaac Lab release v2.0.2 固定 commit `b5fa0eb031a2413c182eeb54fa3a9295e8fd867c`，其 Python 包版本为 isaaclab 0.34.9。Python 3.10.22、torch 2.5.1+cu118、torchvision 0.20.1+cu118。该组合来自 [v2.0.2 源码的依赖定义](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2)，并已在本机实际执行完整核心 M0 验收；不宣称原定 5.1/2.3.1 已验证。

Conda 使用项目绝对 prefix `.conda/isaac-apfg`。Conda/pip/UV/HuggingFace/Torch/Triton/XDG/CUDA/GL/OptiX 缓存与临时下载均指向项目。OptiX 须另外设置 OPTIX_CACHE_PATH，依据 [官方 API 说明](https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html)；初次真实 GPU 审计发现旧默认缓存路径后已修正，没有删除用户既有共享缓存。Kit portable-root 位于 `cache/omniverse/portable`，只有仿真子进程使用 `.runtime_home`。

安装前可用空间约 3.1 TiB，保守峰值预算 150 GiB，另留 20 GiB。复用约 13 GiB Sim 二进制和需要的官方资产子树，没有下载完整 Isaac 资产库。默认 Anaconda 频道遇到 ToS 后改用 conda-forge，没有替用户接受额外条款。PyTorch 使用官方 cu118 wheel。全部组件安装在项目内，指纹一致且 pip check 通过时跳过重复安装。

```bash
source scripts/project_env.sh
APFG_ISAAC_BINARY_SOURCE=/path/to/official/binary bash scripts/setup_isaac.sh
APFG_ISAAC_ASSET_SOURCE=/path/to/Isaac/4.5 bash scripts/download_isaac_assets.sh
```

独立 prefix 的 pip check 通过，AppLauncher 和实际 Sim 启动通过。官方二进制 setup 会优先加载部分预打包 NumPy 1.26.0 / ImageIO 2.22.2，而独立环境安装为 1.26.4 / 2.38.0；两种上下文分别记录。预打包目录不是完整独立 pip 环境，其元数据缺失/冲突保留在本地日志，没有为消除输出而覆盖 Kit 的 USD ABI。

项目从固定官方 experience 派生最小 M0 配置，只启用 core/assets，排除 tasks、mimic、RL 扩展，保留 PhysX 和真正的相机渲染。未额外安装 ROS、强化学习框架、CLIP、DINOv2 或模型权重。派生配置保留 BSD 许可证。

历史 CUDA 故障曾表现为系统 cuInit=999 和 /dev/nvidia-uvm 打开 EIO；UTC 2026-11-21 的复查为 cuInit=0、GPU 数=1，随后渲染和物理均实际通过。项目没有执行驱动维护或重启，恢复原因尚未确认。不能仅由 nvidia-smi 成功推出计算或渲染正常，实际证据见各 run。

真实渲染期间的 strace 写入审计已执行。持久写入路径、RAM 共享内存/memfd、零字节运行控制标记、字符设备和内核线程命名分别报告。共享桌面缓存的并发变化不归因给本项目；该跟踪不覆盖全部历史安装操作。原始系统/SDK 日志仅本地保存，不上传 GitHub。
