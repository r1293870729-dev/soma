# SOMA 实验手册逐项核对与通俗待办清单
**核对日期：2026-11-08**  
依据：experiment_manual.pdf v0.1（2026-08-09）、2026-08 至 2026-10 历史报告、2026-09-29 审计包、2026-10-24 B 定稿包及当前 GitHub main。跑过不等于通过手册验收；工程单测不等于科学实验。

## 0. 总结

SOMA 已经做过真实实验，但目前完成的是**历史探索实验 + 部分工程验证**，不是完整 Q1/Q2/Q3 证据链。

- 历史被动训练：审计包登记 30 个 run（controlled 3、dSprites 9、Shapes3D 9、CIFAR 9）。
- 2026-08-26 Xi-B 汇总报告：若表格每个 latent dimension 均有 3 seeds，则 controlled 4×3 + 其余三组各 3×3 = **39 个配置-seed 组合**，与审计包 30 个 run 不一致。必须用 run ID、DONE、manifest、seed 和 hash 对账。
- I1–I4：12 个父训练组合、214,128 行窗口级记录；不是 214,128 个独立样本。
- 主动 SOMA：9 个旧版 pilot（Signal/Psi、Fixed、Random 各 3 次），有积极描述性趋势，但存在未来候选访问、旧信号版本和指标定义问题。
- 2026-10-24 B 定稿包：61 项 CPU/工程测试、两个独立 600-step CPU 开发运行、19,200 个残差及重算一致性。这是工程开发证据，不是正式 B09、C03 或 D03 科学结果。
- 当前主信号已定为 B/mass；“选择 B”不代表 B 的正式验证已完成。
- 按 2026-10-24 最新任务表，A01–E06 没有科研任务自动 accepted；状态为 needs-revision 或 blocked。“未验收”不等于“从未做过”。

## 1. 三道考试：Q1 / Q2 / Q3

1. **Q1：信号本身有没有用？** 固定模型结构，只观察 B。要证明它能否比强检测器更早、更可靠地发现结构性问题，而且不是误差变大、数据变难或参考漂移造成。
2. **Q2：信号能不能帮模型选对编辑时机？** 允许 split/merge/grow/prune，但公平比较 B、固定时间、残差幅度、随机、oracle 五种触发。重点是 B 选的时间是否真的有用。
3. **Q3：真实自然数据流中是否仍有用？** 在 CLEAR/CLOC 独立运行，比较长期表现、遗忘、恢复、算力和内存成本。受控玩具数据不能替代自然流验证。

通俗比喻：Q1 检查“报警器准不准”；Q2 检查“报警后采取行动是否改善结果”；Q3 检查“真实环境长期使用是否值得成本”。

## 2. 已做实验、结果与问题

### 2.1 被动检测（旧版 B/MP-EMA）

2026-09-29 审计包记录 30 个 run：controlled 3（z=4）、dSprites 9（z=8/16/32）、Shapes3D 9（z=16/32/64）、CIFAR 9（z=16/32/64）。每次 3×1000 steps、batch 64、特征维 256、窗口 512，约 303 行信号日志，保存终态 checkpoint。

旧检测器比较报告（2026-09-01）的 change-point AUROC：

| 数据流 | 旧 Ψ / Xi-B | MSE | -谱熵 |
|---|---:|---:|---:|
| controlled | 0.825 | 0.882 | 0.838 |
| dSprites | 0.724 | 0.899 | 0.835 |
| CIFAR | 0.884 | 0.362 | 0.858 |
| 3D Shapes | 0.647 | 0.878 | 0.643 |

AUROC 0.5 约等于随机排序，越高越好。旧 Ψ 在 CIFAR 上明显优于 MSE，但在 controlled、dSprites、3D Shapes 上 MSE 更高。3D Shapes 精确值 0.646567，低于 0.65；“全部≥0.65”只能算四舍五入表述。

**不能据此说新版 B 已经胜过强检测器。** 这是旧 MP/EMA 版本、旧阶段标签定义下的探索性 change-point AUROC，不是固定 white + 独立经验 edge 的 B 正式测试，也不是手册 B09 要求的生成器级配对增量、matched-FAR、检测延迟和误报/1000步。

主要协议问题：
- 旧 signals_xib.py 按真实 stage_name 建立/选择 EMA 白化参考，阶段真值进入测量链。
- 使用 MP 理论 edge，而非手册要求的独立 edge 窗口经验顺序统计量。
- probe 验证集被重复用于报告 test 准确率，且拆分 seed 随 global step 变化；角色隔离不合格。
- 训练数据按阶段循环访问，不等于 CLEAR/CLOC 自然单遍流。
- 终态 checkpoint 不含足够的 optimizer/RNG/参考/全程残差状态，不能恢复完整历史测量链。

### 2.2 Xi-B_signal_report（2026-08-26）

报告给出最终 step 3000 的 Xi-B、MSE、KL、spike_rank 和 probe：
- controlled：z=4/8/16/32 的 Xi-B 约 18–19；
- dSprites：z=8/16 接近 0，z=32 约 0.0075；
- CIFAR/CIFAR-C：约 0.058–0.071；
- 3D Shapes：z=16/32/64 约 5.06/1.92/1.38；报告提醒 z=16 KL 异常大、训练不稳定。

**这里的 Xi-B 是信号数值，不是 AUROC。** 不能拿 18.4 与 0.825 比大小。信号幅度高也不等于检测准确率高。

**run 数有待对账：** PDF 表格列出 controlled 四个 z、每个 3 seeds；其余三组各三个 z、每个 3 seeds，合计 39 个配置-seed 组合；审计包只有 30 个 run，controlled 仅 z=4 三个 run。需对照原始 run 文件夹、manifest、seed、DONE 和 hash。未对账前不要把 39 写成已审计的 39 个完整 run。

跨数据集绝对 Xi-B 不能直接比较：报告注明 controlled 使用 d=64、W=1024、经验 tau_null；其余组使用 d=256、W=512、MP edge。参数化不同，不能把 controlled 高值解释成公平的跨数据集强弱排名。

### 2.3 超参数扫描（96/96）

2026-07-21 至 07-22 报告记录 96/96 runs 完成，扫描 batch=128、W=128/256/512/1024、EMA gamma、latent z=1/2/4/8/16/32、edge/mass 及 loss-only/loss-plus-xi。可作为历史探索，但 EMA 和 loss-plus-xi 与最终 B 协议不一致：
- 96/96 是旧版网格跑完，不是新版 B 校准验收。
- x=4、gamma、z=4 不能自动成为最终参数；需按独立 white/edge/cal/hval 角色重验。
- loss-plus-xi 结果不能作为 Q1 被动信号主证据，因为 Q1 禁止信号进入训练损失。

### 2.4 I1–I4 残差干预

已有 12 个父训练组合、214,128 行窗口记录：
- Parent：原始残差；
- I1：沿主特征方向注入低秩方向结构；
- I2：打散方向、匹配中心化 Frobenius 总能量的各向同性噪声；
- I3：逐样本保持残差长度、随机化方向；
- I4：整个窗口统一乘 2，改变幅度但不改变方向。

历史观察：I1 的 Ψ 增加；I2/I3 通常降低；I4 接近 Parent。这支持“读数对方向组织敏感、对整体统一缩放相对不敏感”的初步解释。

仍有问题：
- 214,128 是窗口行数，不是独立 n；真正独立层级是父训练组合/生成器。
- 父残差 chunk、完整谱、白化参考未全部追回；manifest 有旧 Windows 本地路径。
- 历史数据有 6 条件（含 I2 equispectral），当前脚本只生成 5 条件；代码与 manifest 版本不一致。
- I3 保持逐样本范数，不保证去均值后的窗口协方差能量逐窗不变。
- I2 的能量匹配合同需与手册最终版本逐字段核对。
- 因此 B10 是“做过但需修复/重算”，不是“已验收”。干预只改变离线诊断残差，不是模型训练因果实验。

### 2.5 主动 SOMA pilot

旧报告记录 Signal/Psi-trigger、Fixed、Random 各 3 seeds，共 9 runs。审计重算描述性均值：

| 条件 | runs | inspections | accepted | final eval MSE | 旧 TAEF |
|---|---:|---:|---:|---:|---:|
| Signal/Psi | 3 | 33 | 21 | 0.007024 | 0.289683 |
| Fixed | 3 | 36 | 15 | 0.011446 | 0.055556 |
| Random | 3 | 31 | 19 | 0.009899 | 0.150794 |

Signal 在这批 pilot 的 final eval MSE 更低、旧 TAEF 更高，是值得继续验证的现象；不是正式因果结论。

关键问题：
- 旧主动信号仍是 MP/EMA 版本，不是最终固定参考 B。
- 候选池预生成并按 cursor 消耗；代码/inspection 记录推导出早期检查可能使用尚未到达的未来阶段候选分布。需按到达时间修复后从头重跑。
- 缺 magnitude 和 oracle 两个关键触发条件。
- null 目标完整日志、独立 audit、matched no-edit shadow 不齐。
- optimizer/RNG/cursor/语义轴继承和原子回滚未证明符合手册。
- 旧 RRB 实际是 MSE 超额面积，不是手册要求的 restricted recovery time；旧 TAEF 贪心匹配有非最优反例，需精确二分图匹配重算。
- 旧 BWT 是 MSE 快照差，不是标准 probe-accuracy BWT；forgetting=0 不能声称“已证明降低遗忘”。

## 3. 2026-10-24 B 定稿包

B/mass 已确定为唯一主信号家族，正式标识 B_mass_fixed_white_empirical_v1。A/edge、C/gap、谱熵只作比较/消融。B 是固定参考白化后，窗口协方差迹归一化，再把所有严格超过独立经验 edge 的谱模态相对超额质量求和。

2026-10-24 包的工程增量：
- CPU float64 B 参考实现、独立残差重算接口；
- 精确 TAEF 匹配和 RRB 时长计算原语；
- 61 项测试通过；
- 两个独立 CPU 开发进程各 600 steps，19,200 个残差；训练、信号、残差和重算一致；
- 生产配置预检按预期拒绝未冻结配置。

这仍不是正式科学验收：无 GPU 后端一致性/速度验收；无正式生成器 B04/B05/B09；无完整 production SOMA 事务与新 B-trigger 主运行；无 CLEAR/CLOC 自然流最终运行；CPU 开发运行不是论文性能结果。

## 4. 对照手册的 38 项任务清单（状态按 2026-10-24 最新任务表）

状态词：**needs-revision** = 做过一部分但有缺陷/证据不全；**blocked** = 依赖条件未满足，不能开始正式验收。没有任务自动 accepted。

### A：准备与冻结
| ID | 状态 | 通俗解释 / 还要做什么 |
|---|---|---|
| A01 | needs-revision | 清点文件、run、seed、代码版本、许可证、父残差、hash；解决 Xi-B PDF 39 vs 审计账本 30。 |
| A02 | needs-revision | 正式环境、生产 B、GPU/CPU 一致性和完整重放。 |
| A03 | blocked | 每个样本只能属于 white/edge/cal/hval/train/probe/audit/test 一个角色；禁止阶段真值/未来数据泄漏。 |
| A04 | blocked | 事先冻结独立生成器数、seed、主指标、统计、停止/失败/重跑规则。 |
| A05 | needs-revision | 统一 run ID、配置/hash、失败状态、资源和完整日志。 |

### B：Q1，被动信号
| ID | 状态 | 通俗解释 / 还要做什么 |
|---|---|---|
| B01 | needs-revision | 用手工可算的小矩阵证明 B 公式算对。 |
| B02 | needs-revision | 测秩亏、近奇异、厚尾、CPU/GPU 精度是否改变 B。 |
| B03 | blocked | 用独立 edge 窗口定固定门槛；另测稳定期线上误报。 |
| B04 | blocked | 冻结 S1→T2→局部分裂生成器和独立 probe 数据划分。 |
| B05 | blocked | 固定拓扑、不编辑、不把 B 加进 loss，完整保存残差/谱/B/probe。 |
| B06 | blocked | 均值、尺度、噪声、平滑、参考污染等假变化对照。 |
| B07 | blocked | 4 倍容量、计算匹配、延长训练、参考漂移对照。 |
| B08 | blocked | 实现强多变量/子空间/顺序检测器，在独立验证流选最强对手。 |
| B09 | blocked | 未见过的最终测试流，比较 B 与最强对手的 AUROC、误报率和延迟。 |
| B10 | blocked（历史干预已做，有缺口） | 追回父残差，统一 6 条件版本，修复 I2/I3 匹配后重算。 |
| B11 | blocked | CLEAR/CLOC 固定拓扑被动监测，换读出器/模型检查稳健性。 |
| B12 | blocked | 形成 Q1 Fig.1–2、补图表和 Source Data 闭环。 |

### C：Q2，主动结构编辑
| ID | 状态 | 通俗解释 / 还要做什么 |
|---|---|---|
| C01 | blocked | split/merge/grow/prune 继承参数、optimizer 动量、RNG 和 cursor；失败能回滚。 |
| C02 | blocked | 校准参数门+B结构门、持续次数、冷却和六状态机。 |
| C03 | blocked | 用最终 B 从头跑主动 SOMA，日志可逐次重放。 |
| C04 | blocked | 公平比较 B、固定、幅度、随机、oracle 五种触发。 |
| C05 | blocked | 独立 audit 和 no-edit shadow，排除自然恢复假象。 |
| C06 | blocked | 按手册算恢复时长、TAEF 精确匹配和机会水平。 |
| C07 | blocked | 长时间/循环环境检查结构是否无限膨胀或来回反转。 |
| C08 | blocked | 生成 Q2 Fig.3、补图表和 Source Data。 |

### D：Q3，自然数据流
| ID | 状态 | 通俗解释 / 还要做什么 |
|---|---|---|
| D01 | blocked | 登记 CLEAR/CLOC 版本、数据快照、许可和可信事件。 |
| D02 | blocked | 公平登记各 baseline 能用的标签、回放、教师和额外资源。 |
| D03 | blocked | Gate B 通过后跑独立自然流正式实验。 |
| D04 | blocked | 可信事件算恢复；无可信事件就算长期表现，不造变化点。 |
| D05 | blocked | 记录参数、候选、访问、FLOPs、墙钟、显存、存储成本。 |
| D06 | blocked | 并列 B、检查、编辑、结构大小、连续更新和 CAC 解释机制。 |
| D07 | blocked | 形成 Q3 Fig.4、补充材料和 Source Data。 |

### E：统计和论文复现
| ID | 状态 | 通俗解释 / 还要做什么 |
|---|---|---|
| E01 | blocked（旧描述性重算已做） | 按生成器/独立运行配对统计，不把窗口当 n；做 cluster bootstrap、FDR、等效性。 |
| E02 | blocked | 主图和正文数字从机器数据自动生成。 |
| E03 | blocked | 补齐预登记敏感性图和失败条件。 |
| E04 | blocked | 补充表和机器可读实验登记。 |
| E05 | blocked | 每个结论能追到原始文件、代码、run、图点和 hash。 |
| E06 | blocked（工具部分已有） | 干净环境一键重建论文结果并归档。 |

## 5. 推荐执行顺序

1. **资产对账：** 对齐 Xi-B PDF 的 39 个配置-seed 与 30-run 审计清单；追回 I1–I4 parent residual、reference、6 条件代码版本。
2. **修 P0：** 固定 white、独立经验 edge、probe 隔离、全局 sample 角色、候选到达时间、完整状态日志。
3. **做 Q1：** B 算对 → 校准 → 负对照 → 容量/计算对照 → 强检测器预选 → 未见测试。
4. **Gate A 通过后做 Q2：** 完整编辑器和双门 → 五条件公平比较 → audit/shadow → 恢复/机会指标。
5. **Gate B 通过后做 Q3：** CLEAR/CLOC 独立自然流 → 资源账本 → 事件/无事件终点。
6. **最后 E：** 最高独立单位统计、图表、Source Data 和干净环境复现。

## 6. 最重要的判断

- **已做过：** 旧版 30-run 被动检测账本、Xi-B 历史汇总（run 数待对账）、96-run 旧超参扫描、I1–I4 历史干预、9-run 主动 pilot、B 定稿 CPU 工程测试。
- **做过但有问题：** 旧 MP/EMA 被动检测、I1–I4、主动 SOMA pilot、probe 评估、RRB/TAEF 旧实现、历史数据与版本登记。
- **尚未完成正式验收：** 新版 B 的 B03–B12、正式 Q2 的 C01–C08、CLEAR/CLOC 的 D01–D07、正式统计和论文重建 E01–E06。

## 7. 上传范围

建议目录：docs/review_20261108/（核对报告）；reports/source_reports/（历史报告文本）；docs/references/（手册/论文/讲解材料文本）；archives/（可上传的版本包）。

475 MB 的 soma已有资料.zip 和 159 MB 的 四个数据集检测器效果.zip 超过 GitHub 普通 Git 单文件 100 MB 限制；本次不强行塞入普通 Git，需先审查许可证/内容，再通过 Git LFS 或外部数据仓库上传。仓库保持 private。
