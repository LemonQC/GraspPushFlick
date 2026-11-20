# M0 实际执行报告

日期：2026-11-20。项目：`/data/GraspPushFlicks`。当前总体状态：**BLOCKED，APFG_M0_READY=false**。

用户确认 GPU 从 4090 换成了 RTX 3090。实际检测为 RTX 3090 24 GiB、驱动 535.230.02、Ubuntu 20.04.6、GLIBC 2.31。约 3.1 TiB 可用空间，inode 充足。原定 5.1/2.3.1 pip 方案不满足当前系统要求，未强行安装；没有升级驱动或更改图形会话。

实际选用兼容性验证路线为本机 Isaac Sim 4.5 二进制 + Isaac Lab v2.0.2（`b5fa0eb031a2413c182eeb54fa3a9295e8fd867c`）+ Python 3.10 + PyTorch 2.5.1/cu118 + TorchVision 0.20.1，均位于项目内部。该路线不代表原定 5.1 通过验收。独立 Conda prefix 的 `pip check` 实际通过（退出码 0），AppLauncher 导入通过，重复安装会跳过已验证组件。实际 Python 为 3.10.22，Sim VERSION 为 `4.5.0-rc.36+release.19112.f59b3005.gl`。安装完整性与仿真启动分开报告。

| 项目 | 结果 | 实际证据 |
|---|---|---|
| 系统、磁盘、桌面会话读取 | PASS | `system_preflight.json`；有效 X11 会话，未修改认证 |
| 项目内 Conda、二进制复制 | PASS | `.conda/isaac-apfg`、`third_party/isaacsim` |
| 目标 5.1 pip 组合 | BLOCKED | GLIBC/OS/驱动版本不满足官方要求 |
| 独立 Conda 依赖检查 / AppLauncher 导入 | PASS | `environment_versions.json`、`setup_idempotence.log` |
| 官方 YCB 下载 | PASS | 30 个逐物体压缩包，长度/归档完整性/本地 SHA256 |
| YCB USD 转换 | PASS | 30 个真实网格，尺度与参数注册表 |
| USD 静态依赖 | PASS | 52 条目：Franka、21 个原配 YCB USD、30 个转换结果 |
| CUDA 计算初始化 | FAILED | `cuInit(0)=999`，`/dev/nvidia-uvm` 打开返回 errno 5/EIO |
| Headless Test A | BLOCKED | 实际 Kit 启动报无法创建 GPU；物理步数 0 |
| GUI Test A | BLOCKED | 实际启动在扩展注册同步时超时；CUDA 前置检查独立失败 |
| Test B 七轴/夹爪 | BLOCKED | 未进入实际控制，不能声称运动或开合通过 |
| Test C 固定/腕部 RGB-D | BLOCKED | 未获得有效渲染，不能声称外参同步或点云通过 |
| Test D 至少 12 个物体物理测试 | BLOCKED | 独立通过数量 0；资产数不等于物理通过数 |
| Test E S1/S2 杂乱场景 | BLOCKED | 代码已准备，未实际生成稳定场景 |
| 真实视频和关键帧 | NOT_RUN | 无有效仿真帧，未生成占位视频或伪截图 |
| 项目子进程写入跟踪 | PASS | 验证区间 86 次持久文件写入均位于项目内；另有 40 次 RAM 中的 POSIX IPC；非完整安装历史审计 |
| GraspFactory 样例 | PASS | 5 个官方 Panda OBJ，Git blob 哈希一致；未物理验证 |
| DexGraspNet 2.0 | NOT_RUN | 已核查官方接口，仅发现 bulk meshdata 包，未整库下载 |
| GraspNet 模型 | BLOCKED | 官方 Jbox 超时，浏览接口未取得直接模型包；未下载标签/场景 |

原始启动记录为 `headless_probe.log`、`gui_probe.log`，对应 Kit 日志保存在 `cache/omniverse/portable/logs/Kit/Isaac-Sim/4.5/`。`cuinit.strace` 明确记录 `/dev/nvidia-uvm` 返回 EIO。清空 LD_LIBRARY_PATH 后调用系统 libcuda 同样失败；用户态库、内核模块版本一致，设备 major 一致，根因尚未确定。没有执行模块重载、GPU reset、重启或任何驱动修改。

统一运行入口采用该固定版本的 AppLauncher，两种模式均开启相机，Headless 额外添加 `--headless`。当前入口遇到 CUDA 失败会保存唯一 run ID、命令、seed、版本/注册表哈希并退出，防止把失败包装为成功。早期阻塞运行 ID 包括 `20261120T090937Z_babac9c1`（Headless）和 `20261120T090937Z_f546c447`（GUI）；它们没有运动轨迹或视频。后续运行清单见 `m0_summary.json`。

CPU 验证已执行：3 个测试通过（归档路径穿越拒绝、软链接拒绝、仅静态成功不能生成 READY）；Python 编译和全部 shell 语法检查通过；资产转换及递归依赖检查已实际运行。以上均不替代 GUI/Headless 物理测试。尚未经过实际运行的仿真代码可能需要在 CUDA 恢复后修正。

所有完整视频的预定本地位置为 `outputs/environment_check/runs/<run_id>/videos/`；当前目录无有效视频。GitHub 只同步代码、文档、元数据和脱敏轻量证据，不上传环境、第三方源码、真实网格、USD、完整视频、原始系统日志或凭据。

下一步先由管理员在不破坏现有远程桌面的维护窗口恢复系统 UVM/CUDA，再执行 `python scripts/check_cuda.py`，随后依次运行物理、机器人、相机、13 种独立物体和 S1/S2 GUI/Headless 测试。没有必要在此之前训练模型或设计最终策略。

最终本地占用快照：项目实际分配约 28.75 GiB，可用约 3154.2 GiB。共享桌面期间观察到外部 GLCache 的 6 个小文件变化，约 610261 字节，不能归因给本项目；没有删除这些文件。对项目验证子进程的 strace 跟踪未发现外部持久文件写入；另记录 40 次 `/dev/shm` 中 SDK 的 POSIX 共享内存/信号量操作（内存文件系统，非安装包或磁盘缓存），没有删除这些共享 IPC。该证据只覆盖明确记录的验证区间，不能冒充全程系统调用审计。

二进制运行路径的 NumPy/ImageIO 实际为 1.26.0/2.22.2，独立 Conda 安装版本为 1.26.4/2.38.0。将 Kit 的 vendor prebundle 当作独立 pip 环境检查时仍有缺失/冲突元数据（本地 `vendor_prebundle_pip_check.log`）；没有为消除该输出而安装可能冲突的 usd-core。独立 prefix 检查通过不等于预打包运行依赖已经全部验证，CUDA 恢复后须实际复测。

首个里程碑已通过 GitHub Git data API 非强制更新 main，并用 git ls-remote 核对 commit `194bd71941a2806aeb7fab5d350572433eece6ec`。最终软件版本和证据补充提交以本报告所在的仓库 commit 为准。
