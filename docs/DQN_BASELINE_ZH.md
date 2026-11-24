# 候选条件 Double DQN：实现边界与复现

实现状态为 **FROM_SCRATCH**：`apfg/baselines/dqn.py`，不是SB3固定动作输出DQN，也不是官方Atari环境复现。方法采用[Double DQN原论文](https://arxiv.org/abs/1509.06461)的在线选择/目标网络评估分离，并适配本项目的动态合法候选池。经验回放和目标网络的DQN基础见[原始DQN论文](https://arxiv.org/abs/1312.5602)。本轮没有重训GraspNet。

共享输入是传感器派生32维观测、至多24个24维候选描述和合法mask。每个候选使用相同权重的 `Q_theta(o,c)` 评分，不使用槽位ID嵌入，因此重排候选只会重排对应Q值，动态候选不会被当成跨时间固定动作槽。网络是观测MLP编码+候选拼接MLP，完整配置在 `configs/baselines/dqn.yaml`。RGB-D原件与特征来源由共享数据manifest记录；本模型并非直接从原始RGB训练视觉编码器。

对转移 `(o,C,m,a,r,o',C',m',d)`：

```
a* = argmax_{j : m'[j]=1} Q_online(o', C'[j])
y  = r + gamma * (1-d) * Q_target(o', C'[a*])
L  = mean SmoothL1(Q_online(o,C[a]), stop_gradient(y))
```

终止或下一步没有合法候选时bootstrap严格为0。当前动作必须在合法mask内才可写入replay。优化器为Adam，梯度裁剪，按配置间隔硬同步目标网络；目标网络不计算梯度。epsilon按实际加入replay的交互转移数线性下降，使用独立、可恢复的NumPy RNG。完整checkpoint保存online/target/optimizer/replay/更新次数/epsilon进度/RNG/训练来源，使用 `weights_only=True` 读取。

`CandidateDQN.act(state,epsilon=0)` 为评测；`epsilon=None` 使用训练日程；全非法池返回None。`observe(transition)`/`add_transition()` 接收真实环境转移，`update()` 执行实际反向传播并返回TD日志。公共库不自行启动Isaac或抢GPU；GPU版训练CLI申请共享项目锁，调用方不得再持同一锁等待子进程，以免嵌套锁死锁。

独立入口：

```bash
source scripts/project_env.sh
"$APFG_PYTHON" scripts/train_dqn.py --manifest outputs/baselines/<dataset>/manifest.jsonl --device cpu --updates 120
"$APFG_PYTHON" scripts/eval_dqn.py --manifest outputs/baselines/<dataset>/manifest.jsonl --checkpoint outputs/baselines/dqn/<run>/checkpoint.pt
```

训练CLI只加载 `split=train`，保留其中真实成功和失败；拒绝与holdout的episode、scene、对象和seed重叠，核验每个NPZ的SHA256。qualification/E1数据不能作为训练样本。该CLI对已经由真实在线交互产生的replay做小规模优化，须准确称为“小规模物理交互replay训练”，不能据此声称完成大规模在线RL。直接在线训练应在共享物理runner每次step之后调用observe/update。

输出在唯一项目内目录：`curve.csv/json`为每次实际更新的loss、TD误差、Q/target均值、梯度、replay量、epsilon、target同步和墙钟；checkpoint及原始回放仅本地。`eval_dqn.py` 只审计checkpoint在相同holdout观测上的推理和mask，**不执行机器人，也不输出物理成功率**。完整取出必须通过共享物理runner单独评估，并报告同一holdout scene IDs、控制预算及真实成败。

CPU测试 `tests/test_dqn.py` 的小样是显式数值单元测试，不是物理训练数据或科研实验。测试覆盖Double而非普通max-target、全非法/terminal目标、候选置换等变、合法随机探索、GT额外字段不进入网络、回放复制、真实梯度更新和精确checkpoint恢复、数据哈希与对象隔离。实际物理采样、训练曲线和heldout测试状态由本轮总报告记录；没有数据时应FAILED/NOT_RUN，不能把测试小样冒充训练结果。
