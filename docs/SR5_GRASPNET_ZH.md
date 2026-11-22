# SR5 的真实 GraspNet 6-DoF 抓取与位姿可视化

此前 `run_sr5.sh --test grasp` 是固定姿态、已知坐标的硬件夹持测试，并未运行 GraspNet。新的入口 `run_sr5_graspnet.sh` 才是“真实腕部 RGB-D → 上游 GraspNet 推理 → 三维候选 → 原始预测的 TCP 映射 → SR5 物理抓取”。六自由度包含位置与完整空间姿态，不限制为竖直俯抓或只调整 yaw。

## 直接查看

```bash
cd /data/GraspPushFlicks
# 预测并停留在抓取候选画面，适合先检查姿态
bash scripts/run_sr5_graspnet.sh --gui --preview-only --interactive --joint-speed .3

# 预测后执行一次选中位姿的物理抓取，并保留结果画面
bash scripts/run_sr5_graspnet.sh --gui --interactive --joint-speed .3
```

默认加载真实 YCB 糖盒和 SR5 / PGIA / D435i 装配。GUI 内有候选夹爪轮廓、绿色选中位姿、蓝色接近箭头、红/绿/蓝 XYZ 轴，以及原生信息窗口，显示分数、世界系 TCP xyz、四元数 xyzw、预测开度及相对向下方向的倾斜角。其他候选也保留各自空间方向。绿色目标和坐标轴位于**观察时刻的预测位置**，物体抬起后不会人为跟随物体。

若要选择另一网络候选，添加 `--candidate-rank 2`；rank 按本次模型分数与几何去重结果确定，不同观测不保证同一 rank 是相同姿态。`--top-k 20` 控制显示数量，`--asset-id ycb:005_tomato_soup_can` 可更换本地注册表物体。换物体、相机或候选不等于已经验证成功，失败会留下独立结果和视频。`--headless` 使用相同传感器、模型、选择和物理代码。

GUI 完成后保持结果画面供旋转检查，关闭窗口退出。默认只推理并执行一次，不是逐帧持续检测系统。

## 实际信息流

1. 相机通过真实关节运动到固定的工作区斜视点。使用源 `ral_optical` 外参和实际 RGB-D，深度为光轴 Z 的米制值。
2. 将深度反投影为点云，以固定工作台坐标边界裁剪；输入网络的是测量点及光学标定，**没有物体真值位姿、实例分割或抓取标签**。高于桌面 6 mm 的工作区点作为模型输入，完整可见工作区用于一次性几何筛选。
3. 独立干净 Python 进程加载上游 GraspNet baseline 的真实权重，`is_training=False`，300 views、12 angles、4 depths；从测量点抽取 20000 点并执行真实 CUDA 网络与 `pred_decode`。
4. 完整保存原始 N×17 预测，按分数、PGIA 80 mm 总开度和近邻位姿去重生成显示候选，不给竖直姿态额外加分，也不将网络旋转替换为朝下。
5. 最多对前八个候选进行一次来源明确的末端外形/接近走廊检查与必要 IK。使用观察点云、已知桌面平面和源 URDF 中相机/夹爪/转接板外形边界，不使用隐藏物体状态、影子物理、逐帧安全扫描或多次抓取预演。此筛选不保证遮挡几何和整个机械臂无碰撞，实际碰撞仍由 PhysX 处理。
6. 仅执行选中候选：开爪、抬高手腕调整姿态、到预抓点、固定方向接近、合爪、保持完整预测姿态抬升。失败不会自动改深度或旋转重抓。物体真值只用于**执行后**抬升/双指接触评估。

## GraspNet 原点与实际 TCP

GraspNet 旋转的 +X 是接近方向，+Y 是开合方向；源 SR5 `ral_tcp` 的 +Z 是接近、+X 是开合。旋转映射使用已给定的 `GRASP_FROM_TCP` 矩阵。网络 translation 不是该物理 TCP 的坐标，不能直接盲拷。

上游 GraspNet 的虚拟手指端平面位于 `translation + depth * approach`。源 PGIA 指尖末端在 `ral_tcp` 的 +Z 方向 **16.5 mm** 处。因此统一使用：

```
TCP_world = T_world_optical * raw_translation
          + (raw_predicted_depth - 0.0165 m) * world_approach
```

这是从源装配和预测语义推导的固定几何映射，随网络 depth 字段变化；不修改 raw 预测，也不做物体专用深度修正。测试验证物理指尖端平面与模型预测端平面重合。源手指长 55 mm，而上游示意夹爪长 60 mm；真实 PGIA/CAD及实机 TCP 仍需实测校准。

## 模型来源与安装复现

本机已安装并实际使用：项目内 PyTorch 2.5.1+cu118、RTX 3090、CUDA 11.3 编译器构建的上游 pointnet2 与 knn 算子。没有修改系统驱动或原 Isaac/PyTorch 版本。原始上游 commit 为 `280c215129f759ed8649cb4e89fc5dfee55f4f80`。

```bash
bash scripts/setup_graspnet.sh
```

安装只准备推理所需代码、两个算子和约 12 MB 的权重，不训练模型，也不下载镜像仓库中的其他模型。所有文件保存在项目下；源权重与二进制算子不进 Git。

[上游 GraspNet baseline](https://github.com/graspnet/graspnet-baseline) 提供的 RealSense 与 Kinect Google Drive 链接本次均返回 quota exceeded。实际使用 [公开 checkpoint-rs 镜像](https://huggingface.co/dgrachev/a2_pretrained/blob/7856e4e550052131b6ac5c0adb6564b04433ad9b/checkpoint-rs.tar)，已校验其 HF LFS SHA256 `60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868`，并以 `weights_only=True` 加载、严格匹配全部上游网络参数。**镜像文件与不可访问的官方 Drive 文件是否哈希相同，没有独立核实**；不能将镜像校验写成官方原件认证。来源及固定 revision 见 `assets/manifests/graspnet_model.json`。上游代码/模型的非商业使用条件仍适用。

## 证据和边界

每次运行使用唯一 `outputs/sr5_graspnet/<run_id>/`，保存真实 RGB-D、K/T、点云、原始网络数组、模型分数、候选映射、筛选理由、控制指令、物体和机械臂遥测、真实接触、三路 MP4、关键帧以及成功/失败结果。没有创建人工抓取约束或 rollout 中物体位姿重置。

实际验收见 `evidence/sr5_graspnet/summary.json` 和 `docs/SR5_GRASPNET_REPORT_ZH.md`。这是对单物体 6-DoF 感知抓取链路的工程验证，不是一般物体成功率、杂乱提取评测或新算法贡献。相机/PGIA 外形和外参仍采用原硬件快照的明确近似。
