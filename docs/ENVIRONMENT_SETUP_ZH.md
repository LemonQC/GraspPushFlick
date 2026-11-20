# 环境安装与当前限制

项目路径：`/data/GraspPushFlicks`。当前 GPU 是用户确认的 RTX 3090 24 GiB；Ubuntu 20.04.6、GLIBC 2.31、NVIDIA 驱动 535.230.02。检测到有效 X11 会话，继承其显示和认证变量；工作期间没有修改远程桌面、驱动、显示管理器或用户启动脚本。

原定 Isaac Sim 5.1.0 pip 安装要求 GLIBC ≥2.35，官方系统矩阵列出 Ubuntu 22.04/24.04 和更新的驱动。因此没有强行安装 5.1。参考 [5.1 安装文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) 和 [5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。

实际选择本机已有 Isaac Sim 4.5 二进制，复制到 `third_party/isaacsim`；配套 Isaac Lab v2.0.2，commit `b5fa0eb031a2413c182eeb54fa3a9295e8fd867c`，复制已有官方 Git 仓库并 checkout 固定 tag。没有修改原安装。完整二进制版本字符串在本地版本报告，不把它称为 5.1。

[Lab v2.0.2 官方源码](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2) 的安装配置对应 Sim 4.5、Python 3.10、Torch 2.5.1/cu118。使用二进制路线避开旧系统的 pip manylinux GLIBC 门槛。该版本官方安装说明记录其测试平台是 Ubuntu 22.04、驱动 535.129；本机 Ubuntu 20.04 的兼容性必须以实际运行证明，目前尚不能确认。没有将 README 的 Linux 平台链接当作完整验证证明。

```bash
source scripts/project_env.sh
APFG_ISAAC_BINARY_SOURCE=/path/to/extracted/official/isaacsim bash scripts/setup_isaac.sh
APFG_ISAAC_ASSET_SOURCE=/path/to/Isaac/4.5 bash scripts/download_isaac_assets.sh
```

Conda 使用 `.conda/isaac-apfg` 绝对 prefix；Conda 包缓存、pip/UV/Torch/HuggingFace/Triton/XDG/NVIDIA 缓存、临时文件均指向项目目录。Anaconda 默认频道遇到未接受 ToS 后改用 conda-forge，没有替用户接受额外服务条款。Kit 使用该版本支持的 `--portable-root <project>/cache/omniverse/portable`。仿真子进程的独立 HOME 不影响父进程或桌面认证。

安装前磁盘超过 3 TiB 空闲，inode 充足；从本地复用约 13 GiB Sim 二进制，PyTorch 主 wheel 838.3 MB、cuDNN wheel 663.9 MB，其余 CUDA 运行库按官方依赖安装。保守峰值预算 150 GiB，并另留 20 GiB。没有下载完整 Isaac 资产库，也没有安装 ROS、VLA、CLIP、DINOv2 或 RL 训练框架。

安装网络存在代理链路缓慢/停顿。先检测了官方端点的直接 HTTPS 连通性，再对安装子进程移除代理变量；没有关闭 SSL 校验或修改系统代理配置。已下载完整的 wheel 保留在项目缓存；部分下载保留 `.part` 或项目 tmp 中的文件。详情见 `setup_isaac*.log` 和下载日志。

当前系统级阻塞：直接调用系统 `libcuda.so.1` 返回 `cuInit(0)=999`；`/dev/nvidia-uvm` 存在且权限允许，但 `open(O_RDWR)` 返回 errno 5 / EIO。`strace` 确认失败发生在驱动设备打开阶段；清除 LD_LIBRARY_PATH 后仍失败。用户态库、内核模块均报告 535.230.02，设备 major 与 `/proc/devices` 一致，尚不能据此确定根因。Headless Kit 日志记录 `Failed to query CUDA device count` 和 `No device could be created`。

需要服务器管理员在维护窗口检查并恢复 NVIDIA UVM/CUDA 状态；是否重载模块或重启应由管理员根据桌面占用决定。项目没有自动执行这些操作。恢复后先运行 `python scripts/check_cuda.py`，再执行完整 M0。仅 `nvidia-smi` 成功不能证明 CUDA 或 RTX 渲染正常。
