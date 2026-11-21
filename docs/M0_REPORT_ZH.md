# M0 实际执行报告

复测时间：UTC 2026-11-21（服务器当地 2026-11-20 晚）。项目 `/data/GraspPushFlicks`。当前所有仿真、资产和存储核心验收项 PASS；GitHub 新版本同步核对完成后才标记 APFG_M0_READY。最终机器可读状态见 `evidence/m0_status.json`。

实际组合：RTX 3090 24 GiB、NVIDIA 535.230.02、Ubuntu 20.04.6 / GLIBC 2.31；Isaac Sim `4.5.0-rc.36+release.19112.f59b3005.gl`；Isaac Lab release v2.0.2，commit `b5fa0eb031a2413c182eeb54fa3a9295e8fd867c`（Python 包名 isaaclab 0.34.9）；Python 3.10.22、torch 2.5.1+cu118、torchvision 0.20.1+cu118。这属于本机二进制兼容验证组合，不代表原定 5.1/2.3.1 通过验收。

此前系统 libcuda 返回 999、UVM 设备打开返回 EIO，曾真实阻塞运行；本次复查已恢复为 cuInit=0、设备数=1。没有由本项目升级或重载驱动、重启服务器或修改桌面。恢复原因尚未确认。

| 验收项 | 状态 |
|---|---|
| headless_physics | PASS |
| gui_physics | PASS |
| franka_control_and_gripper | PASS |
| fixed_and_wrist_rgbd | PASS |
| twelve_independent_ycb_drops | PASS |
| s1_headless | PASS |
| s2_headless | PASS |
| s1_gui | PASS |
| s2_gui | PASS |
| local_usd_dependencies | PASS |
| storage_audit | PASS |
| gpu_compute_and_rendering | PASS |
| installation | PASS |
| github_sync | NOT_RUN |

| 当前正式运行 | Run ID | 结果 | 实测物理步 |
|---|---|---|---|
| physics/headless/s1 | `20261121T020022Z_ebcaba07` | PASS | 240 |
| physics/gui/s1 | `20261121T020213Z_01355162` | PASS | 240 |
| camera/headless/s1 | `20261121T020853Z_2ba98b0c` | PASS | 1200 |
| robot/headless/s1 | `20261121T015621Z_1cc39931` | PASS | 1200 |
| clutter/headless/s2 | `20261121T020042Z_3e2b62ff` | PASS | 4800 |
| clutter/gui/s1 | `20261121T015853Z_2a521009` | PASS | 4800 |
| clutter/headless/s1 | `20261121T015941Z_411fc517` | PASS | 4800 |
| clutter/gui/s2 | `20261121T020050Z_2eb22380` | PASS | 4800 |

30 种官方 YCB 已下载并转换，53 个 USD 的静态依赖通过检查，30 条转换对象的尺度与网格/转换哈希也通过。13 种物体分别在独立进程中实际落地、接触并稳定，覆盖包装盒、瓶体、圆柱罐、香蕉、杯/碗、钻机/剪刀。每种保存初始/最终位姿、遥测、RGB-D 和可解码 MP4。独立运行 ID 见 `object_smoke.json`，未把同一 YCB 的 Isaac 扫描副本计为新物体。

Franka 七轴和双指夹爪均通过真实控制器运动；初始化后没有重置关节伪造运动。固定和腕部相机实际输出 RGB-D、实例分割、标定和目标点云；腕部相机挂在 panda_hand 下，随真实关节运动改变外参，并检查工作台确实进入视野。第三人称视角能够看到完整机器人和桌面。

S1 为六个真实物体从非重叠初始状态掉落到桌面的密集场景。S2 为六物体在物理开口围框内自由掉落形成的三维堆叠，围框内尺寸 0.26×0.26 m、墙高 0.30 m，是明确记录的静态碰撞支撑。没有在 rollout 中摆放成功姿态；验收同时检查真实刚体位姿、至少一个物体高于桌面支撑，以及 PhysX 原始回调中的物体间接触对。GUI/Headless 使用相同生成、参数和评测代码。

物理 120 Hz、相机/视频 30 FPS。每次渲染调用推进四个真实物理步，使用 SDK 的实际步计数和时间核对：240 步对应约 2 秒，1200 步约 10 秒，4800 步约 40 秒。视频从最早可用的真实帧开始，缺帧索引显式保存，不跳过物体掉落阶段。每个 MP4 完整解码检查帧数、FPS、时长。GUI 场景实测默认 1280×720，部分 Headless 诊断使用显式配置的 640×480。

稳定性用最后约一秒的实测位姿变化导出速度，阈值 0.02 m/s、0.1 rad/s，窗口位移 <2 mm、转角 <0.01 rad，并验证接触。SDK 原始速度另存，不把睡眠刚体残留的速度元数据直接当成持续运动。七个 CPU 测试通过，包含静止/平移/转动/不足时长的判据检查，以及归档安全和不得误报 READY 的回归检查。

本地资产渲染验证 run `20261121T020853Z_2ba98b0c` 开启 `--offline`，关闭 Kit 注册表查询，Franka、六个 YCB 及地面均从项目路径加载。53 个 USD 的递归引用/纹理和 bundled MDL 均可在项目内解析。没有切断服务器或桌面的网络，不宣称做过操作系统级断网测试。

项目实际分配约 32.19 GiB，剩余约 3150.7 GiB。真实 GPU 验证的写入跟踪通过：持久数据路径在项目内；字符设备、内核线程命名、RAM 中的 POSIX/memfd/零字节运行控制标记另行记录。初次 GPU 审计发现 SDK 默认 OptiX 缓存会触及既有 `/var/tmp/OptixCache_<user>`，现用官方 OPTIX_CACHE_PATH 重定向至项目；原共享缓存未删除。最终跟踪不是全程安装历史审计。strace 下可出现可选遥测启动器报错，正式未跟踪运行的物理/相机通过证据独立保存。

关键本地文件：`franka_scene.png`、`ycb_tabletop_scene.png`、`ycb_pile_scene.png`、固定/腕部 RGB-D，以及 `robot_motion.mp4`、`object_physics.mp4`、`clutter_preview.mp4`、`wrist_camera.mp4`。它们链接至 `outputs/environment_check/runs/<run_id>/` 的原始不可变运行文件，来源和哈希见 `artifact_aliases.json`。完整视频仅在服务器本地。公开少量实际关键帧见 `evidence/previews/`，其编码和源图哈希均记录。

仍有限制：所有新生成对象的质量、质心、惯量、摩擦等是 estimated；杯碗凹腔和薄工具的专项接触精度尚未标定。Kit 预打包目录的 pip 元数据仍有独立缺失/冲突，独立 Conda prefix 的 pip check 通过，所需实际工作流已验证。GraspNet 首次接口尝试受阻，DexGraspNet 未下载整库，GraspFactory 仅五个网格样例且未物理验证；扩展数据不计入主验收。没有训练模型、执行完整推翻抓策略或提出性能/创新结论。

后续建议是测量质量/摩擦并核查复杂碰撞，建立 M1 可复用动作接口和更广的场景种子；不将本轮 M0 工程验证当成最终研究算法。所有失败运行与旧阻塞记录保留，最新正式运行决定当前验收状态。
