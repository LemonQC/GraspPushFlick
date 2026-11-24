# BC / 高层 Diffusion Policy 数据与实现审计

本文件说明实现和可复现入口，实际训练次数、曲线与物理结果必须以本轮 `outputs/baselines/` 和汇总报告为准。单元测试的编码小样不属于真实示范或科研实验。

## 实现身份与官方来源

| 路线 | 实现身份 | 当前动作/观测 |
|---|---|---|
| BC | FROM_SCRATCH | 对同一合法候选池做条件分类；32维真实RGB-D/本体派生特征+每候选24维 |
| 高层DP | ADAPTED_OFFICIAL_NETWORK + FROM_SCRATCH_SCHEDULER | 官方FiLM条件1D U-Net，预测2个24维高层动作token，每步只映射/执行第一个，再刷新观测 |
| 原生DP | NOT_RUN | 尚未重建连续末端/夹爪动作chunk与高频多模态示范接口 |
| 原生DP3 | NOT_RUN | 尚无同频点云+本体状态→连续动作的原生示范/执行管线 |

优先审查了 [Diffusion Policy官方仓库](https://github.com/real-stanford/diffusion_policy) 和其 [低维策略实现](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)。固定源码commit为 `5ba07ac6661db573af695b419a7947ecb704690f`。本轮实际使用其中的 `ConditionalUnet1D` 及两个小型网络组件；MIT许可证允许再分发，完整许可证、原始/本地文件SHA256和修改说明在 `apfg/baselines/_official_dp/`。改动仅包内导入、来源说明和尾随空白，未自行重写并冒称官方网络。

官方低维策略的时间窗、连续动作形式与高层技能池不同。我们保留噪声预测、条件FiLM网络和逐步反向采样；改变观测编码、网络宽度、时间窗与动作表示，移除Hydra/Zarr/diffusers依赖。没有下载预训练权重，没有安装依赖，也没有运行官方原生任务；不得把本结果写成“官方DP复现”。网络从随机初始化学习真实项目示范。官方默认调度/网络参数可见 [固定版本配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)。

本轮使用小型[32,64]通道、32维时间嵌入、AdamW，未使用原工作空间的EMA训练流程；具体超参数/预算在实际config.json中，不能据本轮200次更新的链路测试评价官方DP性能。真实训练与物理失败计数见Baseline报告，单元测试的随机输入不计入其中。

本地DDPM调度器独立实现：平方余弦累计噪声日程、epsilon预测、固定小后验方差、干净动作预测裁剪。32个前向/反向扩散步；训练动作按优化训练子集的min/max中心与半极差归一化，常量维的scale下界.1。余弦日程来自 [Improved DDPM](https://arxiv.org/abs/2102.09672) 的日程形式；调度器不声称为未改动的diffusers类。损失是真实噪声MSE，按有效动作时间窗mask计算，禁止用分类器结果替换扩散预测。

## 统一数据和边界

`apfg/baselines/dataset.py`读取真实episode的transition NPZ与JSONL清单；只向模型提供 `obs/candidates/mask` 等白名单数值。物体、seed、scene ID、最终成功标签与示范来源只用于划分/审计，不输入策略网络。每个NPZ必须包含当前/下一观察、候选、mask、实际动作索引、奖励与terminal。加载时检查形状、有限值、动作合法性和SHA256。

BC与DP均取同一清单中 `split=train && success=true` 且来源明确的记录；两者输出相同的 `successful_demo_subset_sha256`。失败数据保留给Q学习/诊断，不能悄悄改成正例。训练与物理holdout的对象、seed和scene ID重合会被拒绝；同一episode不跨优化训练/验证子集。优化验证只在训练对象内部按完整episode划分，不能当作物理泛化评测。

高层动作序列仅从同一成功episode的后续transition组成，不能跨episode拼接；不足2步时复制末动作以满足网络形状，但补齐部分不计损失。完整RGB-D、K/T、点云、关节/TCP与原视频由环境记录提供确定性引用，本模型本轮使用其派生特征，不使用图像CNN或原生点云PointNet。它与其他Track A方法共用候选池和信息支持。

BC输出对每个合法候选的logit。DP将实际去噪得到的第一个24维token映射到相同池内归一化欧氏距离最近的合法候选。禁止输出mask外动作；空候选池返回−1，交由统一环境终止/分类失败。两者都不读取family，也没有C1→Push等规则标签。

## 实际训练入口与产物

```bash
source scripts/project_env.sh
# 输出目录必须是新的唯一运行目录；MANIFEST由真实物理episode生成
"$APFG_PYTHON" scripts/train_imitation.py --method bc --manifest MANIFEST.jsonl --output outputs/baselines/train/bc_UNIQUE --epochs 200 --device cpu
"$APFG_PYTHON" scripts/train_imitation.py --method dp --manifest MANIFEST.jsonl --output outputs/baselines/train/dp_UNIQUE --epochs 200 --device cpu
```

两者使用固定训练seed与相同的成功集合/episode划分。默认CPU单线程，避免与Isaac抢GPU；若显式使用CUDA，训练CLI申请与其他Baseline共用的项目锁。真实示范不足时，loader拒绝空正例训练；少量有效示范仅允许过拟合/链路smoke，结果本身不授权大规模训练。

产物包含原数据清单哈希、成功子集哈希、episode IDs/长度直方图/动作多样性/维度、逐epoch真实loss CSV、模型、优化器与随机数状态，以及保存前后预测逐值一致性检查。BC/DP模型API分别为 `BCPolicy.load(path).act(state)`、`DiffusionHighLevelPolicy.load(path).act(state,seed=...)`。训练结果中的候选标签一致率只是离线一致率，不能写成物理取出率；物理结果由统一环境对同一holdout场景独立运行。

有限训练资格仍需检查真实成功episode数量、来源多样性、是否出现有效的View/Push/Flip后取出、E1机制门槛和实际holdout结果。若示范主要为直接抓取，本轮只能说明数据/训练接口可用，不能把缺少动作支持归咎于BC或DP的科学能力。

## DP3适配判断

审查 [DP3官方仓库](https://github.com/YanjieZe/3D-Diffusion-Policy)，固定审计commit为 `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`。其 [策略代码](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py) 使用点云编码和本体状态条件，输出连续动作序列，依赖diffusers/PyTorch3D等。输入形式与D435i点云方向相容，但动作形式不等于本项目的四类宏技能池；每个宏动作结束才保存的单个点云不足以直接训练原生DP3。

因此原生DP3保持NOT_RUN：需要高频同步点云/本体/连续EE与夹爪示范、对应执行时序、独立兼容性验证以及原生动作Track B协议。当前不伪装为DP3结果，也不修改原环境来追装这些依赖。
