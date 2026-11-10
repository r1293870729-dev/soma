# Phase 0：阻塞项与交接

审计编号：ML-P0-20261110。日期：2026-11-10。审计基线：`36167e30798afad4b6c68a1a65a7a5df2b8c773e`。

本轮仅静态读取源码、配置和已有记录，生成审计文档；未导入历史训练模块，未执行训练、smoke test、数据集下载或 checkpoint 反序列化。历史代码、配置和结果保持原字节。这里的“已定位/已核对”不代表科学验收或获准进入 Phase 1。

路径别名：`H/` = 本机 `~/Downloads/soma已有资料/soma_psi_code_handoff_20260903(1)/`；`U/` = `~/Downloads/soma已有资料/psi_xib_unified_20260824/`；`P/` = `~/Downloads/soma已有资料/psi_generalization/`；`HAM/` = 同一资料目录下的 `HAM/`。这些历史源码目前不在远端 main 中，引用为本次扫描文件的原始行号，关键文件 SHA-256 见 [信号审计](legacy_b_implementation_audit.md)。

## 阶段状态

Phase 0 的静态盘点及四份审计文档已生成供复核；**未进入 Phase 1**。指定旧版统一 Xi-B 源码已定位，不能确认完整历史模型构建、唯一主任务验证指标和合规数据划分，因此后续方法适配/训练保持停止。不把“报告生成”或“源码能解析”记为实验科学验收。

|ID|阻塞项及证据|影响与需要的后续输入|
|---|---|---|
|P0-01|主动入口 `H/05.../scripts/run_controlled_stream_soma_psi.py:26` 导入 `scripts.run_table1_emergence_mlp8`；本地两份 handoff、资料 ZIP、当前 main 和两个既有审计分支均未找到|追回实际使用的 `_build_mlp_soma`、`_soma_configs` 文件及 hash/版本，才可确认架构、参数名与配置覆盖；不以自己写的工厂补齐|
|P0-02|真实入口分别为 `03` 两项、`05` 四个启用项、HAM 五个 G 项另有 D；不存在已验证的统一“三损失”事实|负责人选定本次多损失实验对应的历史主模型、入口、损失集合、共享层、原权重；不得按示例扩/减任务或自行把复杂度/探针 CE/B 算作训练项|
|P0-03|旧统一 probe 把 val 早停集同时作为 test 栏，split 随 global_step 变；主动流没有明确 hval/test 与最佳 checkpoint 规则|冻结主任务验证指标、数据角色与 checkpoint 选择；明确历史已看评价数据的用途。不得用 test 调权重/超参或选择模型|
|P0-04|旧 stage-keyed 信号依赖真实阶段；主动 candidate 提前生成全部未来阶段池，cursor 没有 arrival 检查|定位算法不等于认可协议。负责人需明确旧信号作为历史兼容诊断的权限与限制；正式比较的时间/标签权限问题需另阶段解决，Phase 0 不修代码|
|P0-05|旧被动30次原始 resolved_args/train_history/probe_history/checkpoint、主动9个 checkpoint 和完整运行环境锁均未在当前可见资料找到；现有源代码与运行缺少完整 commit/hash 链|追回具体运行资产与源码版本；终态 checkpoint 也不等于完整 EMA/窗口/RNG/cursor。当前报告只作静态与已有记录审计，不承诺重放|
|P0-06|负 Gaussian NLL 已由已有日志核实；R_use 多数 inspection 为零；逐项梯度范数/余弦与共同层梯度没有记录|先冻结负值/零初始值的处理定义，再批准未来诊断；不使用 abs 掩盖定义、不凭一个负余弦或损失尺度开 PCGrad；本轮不测梯度|
|P0-07|`H/01` 和父干预训练脚本有 `loss_plus_xi` 可微路径；仓库还有 `B_mass_fixed_white_empirical_v1`、本地 P/固定参考等其他信号|禁止选择这些路径自动替代指定旧统一 B；B 不进入总训练损失，记录需复用同一次诊断结果，不得重复推进 baseline/window|
|P0-08|原始 `multi_loss_optimization_guide(1).html` 未在 Downloads、Codex 工作目录及所审资料 ZIP 中定位；仓库已包含其来源摘要；正文 Exp-0..4 与来源补充 Exp-1..6 编号也不同|Phase 0 可依仓库明确约束完成盘点；未来方法顺序按来源补充，并取得原指南做逐项核对，不将示例假定为 SOMA 专属事实。本轮不实施这些方法|

`H/04` 的父残差 chunk、完整白化/谱状态和生成旧6条件的确切版本未取得，属于历史谱结果重放的附加缺口，不假装本轮重建了 I1–I4。旧 loader 中还存在 `np.load(allow_pickle=True)`、未限制的 torch.load；本轮没有执行，未来启动入口前必须依 AGENTS 审计加载行为。

## 已完成的只读核对

1. 完整读取指定实验协议、agent 入口及 AGENTS、DELIVERY_STATUS、EXPERIMENT_AUDIT、EXPERIMENT_PLAN、PROTOCOL_CONFLICTS、TASK_STATUS；以固定 main 提交作为版本边界。
2. 静态扫描本机85个历史 Python 文件、另一副本56个文件；85个均 AST 解析成功；73个不同字节内容；资料 ZIP 中全部85个 Python 文件与目录匹配。
3. 定位 `XiBSignal`、`StageBaseline`、`ResidualProjector` 的定义、调用、白化/EMA/MP/mass 公式；与主动 Psi、新固定参考 B 分开登记。
4. 读取9个既有pilot的27000行日志和100个有分量的inspection；核对负 NLL、四项目标恒等式，区分逐步值和 candidate-eval 值，未产生新训练数据。
5. 三个实际可见 HAM/YOLO 权重文件只做哈希/ZIP目录检查，未反序列化；未下载数据集。
6. 生成损失依赖图、原训练流程图，以及四份文档。本轮仓库修改范围仅为这四个新增 Markdown 文件；未更改任一历史代码、信号、配置、checkpoint、数据划分或报告原件。

## 交付文件与复核边界

- [损失函数盘点](loss_inventory.md)：原名称、公式、系数、主/辅助身份、共享参数、detach、数值范围、零/负值及未知梯度；含两张 Mermaid 图。
- [旧版 B 实现审计](legacy_b_implementation_audit.md)：真实源码、入口、配置、依赖、公式、版本差异、哈希与 logger 设计边界。
- [数据与划分审计](data_and_split_audit.md)：seed/预算/优化器/数据角色、评价、checkpoint 和缺失资产。
- 本文件：停止条件、阻塞项与后续所需输入。

本地附带机器清单保留文件路径、大小、SHA-256、原始行号检索和已有日志统计；历史源文件不属于这四份报告，远端发布状态以交接回执为准。报告中的缺失仅限本次可访问范围，不声称其他磁盘/机器不存在。没有运行仓库测试；验证方式是静态内容、数值记录恒等式、原件前后哈希和仅四份报告的 diff 检查。未测的逐项梯度、运行环境和科学公平性保持 U/blocked，不用工程检查替代实验验收。

等待负责人复核 Phase 0 并明确上述输入后，才可另行批准 Phase 1。当前未创建 `experiments/multi_loss_optimization/`、权重控制器、logger 实现或训练运行。
