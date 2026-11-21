# SR5 / PGIA / D435i Isaac 硬件仿真

本次按用户明确要求扩展至 SR5 硬件动作验证，不实现新推抓算法。唯一装配源是 `assets/imported_sr5/isaac_sr5_export/robot/assembled_portable.urdf`，原文件不修改。`assembly_current.json`、`sr5_tool_current.json`、`export_report.json` 用于一致性和哈希核验，不从旧版参考 JSON 重新拼装。

机器人为源快照中的真实 SR5 六轴网格和运动链，转接板来自真实米制 CAD。PGIA-140-80-W 壳体、指尖和 D435i 壳体沿用源 URDF 的长方体近似；安装外参、TCP、夹持驱动力和摩擦没有实机标定。本资产用于仿真调试，不宣称实物精确复刻。没有用 Panda 或 UR5 替代 SR5。

## 运行

使用项目内已验证的 Isaac Sim 4.5 / Isaac Lab v2.0.2 / Python 3.10 / RTX 3090；不需要 ROS 或原 PyBullet 服务器。所有新增资产、依赖缓存、日志和视频保存在当前项目目录。首次或装配源更新后导入：

```bash
cd /data/GraspPushFlicks
bash scripts/python.sh scripts/import_sr5.py --headless
bash scripts/run_sr5.sh --headless --test all
bash scripts/run_sr5.sh --gui --test all
```

从已有向日葵桌面终端运行 GUI，继承实际 DISPLAY/XAUTHORITY。添加 `--interactive` 会在完成录像及测试后继续显示场景，关闭窗口退出。GUI 与 Headless 使用同一资产、物理、控制和记录代码。默认三个真实相机、640×480、30 FPS，支持 `--width 1280 --height 720`。本轮实测为 PhysX CPU / MBP / TGS、物理 480 Hz，RTX 3090 负责渲染；不按实时时间 sleep，也不将 GPU 渲染称作 GPU 刚体动力学。

单项检查：`--test joints`、`gripper`、`camera`、`grasp`、`push`。抓取使用已知居中的 40 mm、100 g 方块，检查双指真实接触、至少 100 mm 抬升及持续保持；推动检查真实接触和物体位移。失败保留视频和原因。这里的动作是已知坐标的硬件夹持/推动 smoke，不是感知抓取或完整取出任务。

## 直接输入坐标

```bash
bash scripts/run_sr5.sh --gui --test move \
  --xyz 0.45 0.04 0.32 --quat-xyzw 1 0 0 0 --frame world
bash scripts/run_sr5.sh --headless --test commands \
  --command-file configs/sr5_coordinate_example.json
```

位置单位 m、关节及角度 rad；世界系右手 Z-up。`world` 位置以场景原点为准；`base` 位置相对于源 `base_xyz=[-0.16,0,0]`，两系方向相同。`--quat-xyzw` 是世界系工具姿态，无论位置 frame 是什么均如此。`[1,0,0,0]` 表示工具 +Z 朝下、+X 沿世界 +X。姿态定义在源 `ral_tcp`，不是法兰，也不是相机。只有 xyz 时默认保持控制器当前姿态；CLI 示例提供明确姿态。

Python 中在已初始化的场景使用 `scene.ctrl`：

```python
scene.ctrl.move_tcp([0.45, 0.04, 0.32], [1, 0, 0, 0], frame="world")
scene.ctrl.move_joints([q1, q2, q3, q4, q5, q6])
scene.ctrl.set_opening(0.080)  # 总净开口，不是每侧 80mm
scene.ctrl.grasp([0.4, 0, 0.027], [1, 0, 0, 0], lift_m=0.15)
scene.ctrl.push([0.32, 0, 0.027], [0.51, 0, 0.027])
```

`set_opening` 为每侧下发 opening/2；每指范围 0–0.04 m。六轴目标遵守源关节限位，默认关节指令速度不超过 0.15 rad/s。每个 TCP 目标仅做一次 URDF IK，并执行平滑的关节位置驱动轨迹；不是带障碍规划器。不可达、最终跟踪误差大或限位外请求明确失败。推动沿用户给定直线生成有限运动路点，不做逐帧安全扫描、影子物理或接触预演。初始化之后不调用关节/物体状态重置，也不添加人工抓取连接约束。

## 装配、相机和导入差异

源 `ral_gripper_fixed` 的 yaw 为 +π/2，夹爪开合 X 轴与相机壳体长边 Y 轴平行。转接板 149×110×7 mm 的 STL 保持原米制和源 mesh-local 旋转，仅应用一次。SR5 网格保留源 0.001 比例。单个 articulation，保留所有固定框架、6 revolute + 2 prismatic。

Isaac 4.5 导入器直接把 STL 文件名用作 USD 标识符，数字开头的哈希名会失败。因此导入器副本只将网格文件名加 `m_` 前缀，记录名称映射；不更改任何几何、质量、轴、限位或原装配文件。导入审计还将 USD 中主轴惯量还原为完整惯量矩阵后与 URDF 比较。固定关节不合并，自碰撞开启，保留 PhysX 关节邻接碰撞默认规则；机器人、夹爪、桌面、物体之间的外部碰撞始终保留。厂商网格采用导入器的凸包碰撞近似，源 box 碰撞不变。Isaac PD 和关节摩擦设定与 PyBullet 不等价，参数记录在 `assembly_audit.json`。

源 URDF 的空 `world` link 会生成世界固定约束；在初始化中同时设置固定锚点与 articulation 根位姿，保持 `base_xyz`。默认肘上初态避免肘下 IK 分支穿过桌面，不通过关闭桌面碰撞处理碰撞失败。

当前 Kit 配置关闭 Fabric 渲染变换更新，继承的 Fabric 状态可造成物理状态运动而画面保留绑定姿态。硬件 runner 显式关闭 Fabric，每个采样周期先推进 16 个 1/480 s 真实物理步，再调用 PhysX `update_transformations` 发布实测状态，然后渲染一帧。这样也消除了默认混合步进路径中刚唤醒物体的 USD 更新滞后。逐帧记录 raw USD 渲染框架与物理框架并核对，避免静态画面误通过。没有根据指令另外摆放可视网格。

`ral_optical/D435iSensor` 包含在机器人 USD 中。源 optical 采用 +Z 前、+X 右、+Y 下，通过 Camera 的显式 `camera_axes='ros'` 转为 USD 原生 -Z 前、+Y 上。仿真 58° 垂直 FOV 来自源开发配置，非实测 RealSense 标定。保存 RGB、m 制深度、uint16 mm PNG、K、T_world_optical 和分割；反投影与实际方块表面距离独立验证，GT 只用于评价。IMU 未实现，不影响本轮 RGB-D。

## GraspNet 接口边界

`apfg/perception/graspnet_pose.py` 使用规范矩阵，将 GraspNet 的 +X 接近、+Y 开合映射到 TCP +Z 接近、+X 开合。输入需为外部真实的 17 列预测和匹配的光学矩阵，不做专用位姿补偿或旋转投影修正。缺少模型权重及推理 worker 时，H6 推理验收标记 NOT_RUN。

```bash
bash scripts/run_sr5.sh --headless --test prediction --predictions path/to/predictions.npz
```

NPZ 包含 `filtered` (N×17) 和 `T_world_optical` (4×4)，并需匹配当前观察姿态；选择 `--prediction-index`，不自动搜索/重排候选。完成预测位姿指令不自动等于物理抓取成功，输入预测来源与抓取结果须独立评估。模型训练不在本轮范围。

## 证据与复现

每次运行保存在唯一 `outputs/isaac_sr5_hardware/runs/<run_id>/`，包括原始相机 MP4、逐帧遥测/指令、物体/末端位姿、接触回调、RGB-D 与标定、测试结果及场景 USD。导入文件保存在独立 `imports/<run_id>/`；顶层 `robot_sr5_pgia_d435i.usd` 是相对引用当前成功导入的真正 USD 入口层，保留原层的配置依赖锚点，不使用会破坏相对引用的文件软链接。`assembly_audit.json` 记录源哈希、单位、近似项和完整验证。完整模型、视频及源快照仅留服务器；GitHub 仅保存代码、文档、轻量结果和真实预览。

当前实际验收见 `evidence/sr5_hardware/summary.json` 与 `outputs/isaac_sr5_hardware/README_RUN.md`。失败和旧渲染问题的运行不会删除；仅最终相同配置的重复运行计入验收。

API 依据项目内 Isaac 4.5 源码和其自带 `standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py`，不照搬 6.x API。可参照 NVIDIA 的 [URDF 导入器说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html) 及 [6.x 导入 API 差异](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline.html)。
