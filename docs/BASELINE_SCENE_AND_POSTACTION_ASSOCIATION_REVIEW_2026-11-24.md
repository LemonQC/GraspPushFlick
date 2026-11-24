# 2026-11-24 GraspPushFlick 最新 Baseline 复核：场景稀疏与动作后目标关联

> 审查对象：main commit `110f25e63022a2322b2045c2bdf77cde3b20c5ff`；本报告为源代码/已上传实验记录审查，不是新的服务器仿真或图片逐帧实测。本次提交不应覆写既有 E1 v1/v2/v3 或训练证据。

## 1. 已确认的新工作与证据边界

- 并行基线源码已落实：Grasp Only、Heuristic、Random Valid、候选条件 Double-DQN、BC、高层扩散策略。高层 DP 仅复用官方条件 U-Net 部件，使用本地调度及宏技能候选映射，**不是原生 DP / DP3**；GOMP/GAPG/MS-MEM/Uni-PNP/SPIN 仅来源审查，**未复现**。
- E1 v3 4个 family × 4个动作方案 =16 次同实例试验执行完整且配对审计通过：Direct [PASS,FAILED,FAILED,PASS]；C1 [PASS,FAILED,FAILED,FAILED]；C2 [FAILED,FAILED,FAILED,FAILED]；C3 [PASS,FAILED,FAILED,PASS]，顺序为 Direct Grasp / Push→Grasp / Flip→Grasp / View→Grasp。**没有 Direct 失败但准备操作帮助成功的实例**；所以机制资格 NO_GO，另48次未运行，不能直接大规模训练。
- 12 次演示采样：6成功/6失败、18条转移；BC/DP成功样本6个episode、仅8条转移（6 GRASP +2 ACTIVE，无成功Push或Flip→Grasp）。BC/DP各200次参数更新。DQN仅4个在线episode、6转移和160次更新。
- 两个固定Direct heldout物理实例，每个方法各2次：GraspOnly1/2、Heuristic1/2、Random0/2、DQN2/2、BC1/2、高层DP0/2。样本数极低，没有任何能支持 DQN > DP 之统计结论；DQN汤罐的Active→Grasp PASS仅说明该链路可成功。
- 完整结果与实现身份：`docs/BASELINE_PARALLEL_REPORT_ZH.md`、`docs/BASELINE_IMPLEMENTATION_MATRIX.md`、`docs/IMITATION_BASELINES_ZH.md`、`evidence/baselines/summary.json`。

## 2. 为什么场景稀疏、偏离研究需求（代码级事实）

`apfg/scenes/m1.py` 的 `validate` 限定 **3–5个**物体，而当前 `generate()` 的 Direct/C1/C2/C3/C4 **事实上各生成4件**（1目标+3障碍）。并非 6–10 件混杂目标的真正紧密clutter。主要场景布局：

- Direct/C2: 非目标在 `(.19,.25), (.70,.22), (.20,-.27)`m，而目标约 `(.42,0)`m；远端摆件离目标明显较远。C2只把目标初始旋转绕 Y 90°，**没有经物理任务证明形成不可直接抓但可翻后抓的支撑关系**。
- C1: 目标左右各1汤罐，按初始AABB边界外留7mm间隙，前方1cracker box。但抓取空间是否真正阻塞，必须取**落定后**全臂/夹爪抓取结果，不能只靠物体位置命名。E1中Direct已经成功。
- C3: 只有一件cracker box近目标形成候选遮挡，其余两件较远；当前策略同时获得固定和腕部两路 RGB-D，固定相机近似俯视，可能提前看见本应由Active才能看到的信息。E1中Direct已成功。不能把某一相机的遮挡当成两路综合观测盲区。
- C4仍是4件及固定组合配置，不是随机可解、多技能耦合的真正混杂场景。
- 目前本项目定义的 C1/C2/C3 是**generator family/假定困难**，不是物理证实的必要动作标签；不可用family作监督动作标签。

**结论**：必须重构场景，而不是通过训练奖励强迫系统在稀疏场景依次执行 Push、Flip、Active。不能简单随机堆到十几件或深箱中造成不可达；应为 SR5+80mm夹爪构建“开口桌面/浅托盘、围绕指定目标有相互接触/近接触关系的混杂场景”，先从6–8件逐步扩展到8–10件，通过真实物理落定、可达、接触、覆盖率和可解性筛选。必要时先修机械臂完整轨迹和凸碰撞参数，不为提高难度而制造穿模/假挡板。

## 3. “动作后的关联”到底指什么（已经发现，但没有解决）

真正含义是 **目标对象在执行Push/Flip/Active后的跨时刻视觉数据关联 / 重识别与位置更新**，不是“把RL上一个动作连到下一个动作”，也不是DQN算法缺陷。

当前 `apfg/perception/m1_observation.py::visible_mask`:
- 糖盒 `004_sugar_box`：使用黄/颜色阈值识别可见像素；易受光照、遮挡、相似颜色干扰；
- 其他目标（例如 mustard/soup）：`workspace_prompt` 以**初始固定 XY ROI**过滤点云。在 `apfg/scenes/m1.py` 目标初始XY约(.42,0)，ROI仅初始位置每轴±.035m（整个宽仅7cm），但推进的水平行程是.10m，Flip是.18m。
- `apfg/simulation/m1_env.py::observe()` 每步虽然重新获取真实RGB-D，但一直传入 `self.spec['target_hint']` **而不是动态更新目标位置/identity**。后续 `_measured_clouds()` 只取当前mask点，采用连通点云扩展，近接触物体可被错误合并。`apfg/baselines/candidate_generator.py::build()` 以这些点重新触发GraspNet；一旦目标移出固定ROI、倒向别处、被遮挡或与物体相接，可能误关联到旧位置的桌面/障碍或完全没有有效目标点。
- 因而 `NETWORK_RETURNED_NO_GRASPS` / `NO_LEGAL_CANDIDATE_AFTER_PREPARATION` 有时是**感知数据关联失败**，不一定说明Push/Flip物理无效，也不能把所有失败归因于关联；需逐案例定位。

应先做基于**实际观测 RGB-D + 初始目标提示 + 前后时序**的目标跟踪/再关联：更新目标mask、三维位置、不确定性和观测关联质量；候选对象过于近接时保留多个假设，不用隐藏物体位姿、仿真GT mask或正确物体ID强行关联。真值只用于独立评价关联误差/身份切换/丢失率。动作前后保存 target mask/pointcloud、candidate 数量、目标平移/姿态的评价真值和时间对齐信息，区分 `TARGET_LOST`、`WRONG_TARGET`、`NETWORK_EMPTY`、`IK_REJECT`、`CONTACT_FAILURE` 等错误。

这是 E1 环境的公平输入修复，不是独立科研创新；所有 Baseline 复用同一 tracker，并保留旧数据和协议版本。

## 4. 另一个根本限制：动作候选生成局限于“推动目标本身”

`apfg/baselines/candidate_generator.py` 中 Push/Flip 的接触点几何均由 `target` 点云分位值产生，四个水平轴向、固定 Push .10m / Flip .18m。**没有为周围阻塞抓取通道的非目标物体生成推动候选**。C1“通过移动障碍物打开抓取通道”的本意无法由当前动作池完整测试。候选总量实际最多约11项，数组24槽多数是padding；高层策略即使理想，也不能选择候选库中不存在的动作。Active只有2个预设视点；Grasp仅向高层暴露一个检查通过的候选。这些都是简化的工程动作空间，后续不能把上限误写成算法不行。

建议在同一RGB-D条件下构建：
- 对目标及近邻可观测障碍物分开生成 Push 候选（明确推谁、接触点/方向/行程，任务目标仍仅取出指定目标），无oracle instance mask；
- 对姿态困难物体生成真正有物理可达接触的侧向/上沿翻转候选；现有Flip就是较高接触点的stroke导致竖盒倾倒，不等于卧倒物体可翻起；
- Active候选生成应考虑可达的下一观察位姿与对抓取决策的潜在帮助，而非只有固定front/back/left；
- 更合理的Grasp候选池：多个物理可行6D候选，明确计划约束与预算，所有Baseline共享。

## 5. 下一步 Codex 执行范围：优先修复物理任务定义、目标关联和候选库，再谈训练扩量

### A. 只修改工程/场景版本，不污染旧结果
- 不改变旧`m1_scene_v1`、E1 v1/v2/v3，也不删除已有日志、视频、checkpoint；
- 建立 `apfg/scenes/clutter_v2.py`、`configs/scenes/v2/` 与新基线protocol版本，附来源资产与几何验证。不再默认固定仅4件且远距离摆放；
- 用 SR5+PGIA80mm 和真实物体尺度，生成 Direct、C1、C2、C3、Mixed；重点是**接触邻域密度、真实阻塞、物理可行的姿态改变、目标可见与取出可达**，而不是按场景名指定动作。

### B. 目标关联闭环
- 单独实现 `apfg/perception/target_association.py` 的 mask/pointcloud时序关联，保持相同RGB-D输入；不要偷用oracle GT。日志给出初始/后续点云、时间序列、相机相对移动、关联质量及丢失/误关联。
- 在共同的 `observe/step` 中动作完成后更新目标身份与位置；目标如果不可见则报告 LOST/UNCERTAIN、允许Active合理探索而不是乱给原始目标ROI。
- 加 poison-oracle、固定ROI位移失败复现及相似颜色障碍物测试。不要针对糖盒/芥末瓶写测试物体专用魔法阈值。

### C. 共同动作池和评测
- 可针对目标**或邻近非目标**生成 Push 候选；物理接触和离开轨迹均记录。不能把已知的真值身份/隐藏碰撞几何交给决策网络。
- Flip至少证明一个“初始卧倒→姿态变化→抓取成功”的真实完整例子，且Direct在同一实例不成功/确实显著困难；当前C2一个侧向stroke=0°不是 Flip 成功。
- Active观察时必须检验完整系统（固定+腕部）观测增益，不能选择仅对腕部有遮挡但固定相机已完全看见目标的场景。
- 统一评价：完整取出成功、合法抓取候选/轨迹改善、目标身份关联准确性、时间/非目标扰动、动作数量、失败分类。旧 Baseline 已训练模型可在新场景作为 zero-shot diagnostic，但模型特征/候选schema变化必须显式版本化，不兼容不硬加载。

### D. 限量验证门槛
- 开发层有限 smoke 验证：真实物体稳定、密度、目标关联动作前后和 Push/Flip/View 可执行性；保存 GUI 视频、RGBD、场景初态/末态、失败。
- E1 仍是总共三个前期实验中的**第一个门槛**，在新版本冻结协议、候选、控制器、样本和成本后，用同场景的Direct/Push/Flip/View比较完整取出，禁止挑胜例，不要提前训练网络来绕过未成立的任务。
- 若仍没有机械可行的“Direct失败、有效准备动作后成功”样本，停止大规模DQN/DP数据收集；必须先重新审查任务和场景。
- 强基线仍需未来复现GAPG/GOMP/MS-MEM/Uni-PNP/SPIN；目前仅候选DQN/BC/DP小样链路已运行，不能冒充论文强对比。

## 6. 文件及GitHub约束
维持当前Isaac4.5+Lab2.0.2+RTX3090；不得重装/搬数据。代码、配置、报告、真实关键帧同步 `LemonQC/GraspPushFlick`；原始大视频/资产/训练数据留当前项目目录。新研究版本与旧证据分开；修复后逐条报告场景的可解性、复杂度统计、真实因果对照、关联与抓取失败，所有FAILED/NOT_RUN保留。正式结果严格小样本解释，不能强行创造算法创新结论。
