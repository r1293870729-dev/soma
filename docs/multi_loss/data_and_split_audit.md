# Phase 0：数据、划分、训练入口与评价审计

审计编号：ML-P0-20261110。日期：2026-11-10。审计基线：`36167e30798afad4b6c68a1a65a7a5df2b8c773e`。

本轮仅静态读取源码、配置和已有记录，生成审计文档；未导入历史训练模块，未执行训练、smoke test、数据集下载或 checkpoint 反序列化。历史代码、配置和结果保持原字节。这里的“已定位/已核对”不代表科学验收或获准进入 Phase 1。

路径别名：`H/` = 本机 `~/Downloads/soma已有资料/soma_psi_code_handoff_20260903(1)/`；`U/` = `~/Downloads/soma已有资料/psi_xib_unified_20260824/`；`P/` = `~/Downloads/soma已有资料/psi_generalization/`；`HAM/` = 同一资料目录下的 `HAM/`。这些历史源码目前不在远端 main 中，引用为本次扫描文件的原始行号，关键文件 SHA-256 见 [信号审计](legacy_b_implementation_audit.md)。

## 1. 已访问与未访问的资产

远端固定为 main `36167e3`，递归树未截断；同时检查 `audit/manual-refresh-20261024@8a6e32d` 与 `audit/restructure-20260929@398019a`。本机扫描 `~/Downloads/soma已有资料/`、其 ZIP 和独立 handoff 副本；对源文件、配置、结果文本文档做 SHA-256 和静态检查。包含 9 份主动逐步 CSV（每份 3000 行）、9 份结构日志、9 份阶段评价 CSV 及 run_config、汇总表。

没有读取图像像素或运行数据 loader，没有下载数据，也未载入任何 pickle。`HAM/data` 的大数据内容不进入源码/训练日志哈希盘点；代码目录 `soma/data` 包含在源码扫描中。文件名检索还发现本机 `~/Downloads/data/CIFAR-C/` 的 motion_blur.npy、frost.npy、labels.npy；仅确认路径存在，没有检查 shape、版本、样本量、划分或合法用途。没有在本次检索范围内找到 dSprites/Shapes3D 的实际 hdf5/npz 文件。

仓库旧审计提及的 30 个被动 checkpoint、逐 run resolved_args 和原始 train/probe histories，本次目录及资料 ZIP 中未找到。该旧文档是 C 级来源，不能当成此次亲自读取了 30 个 checkpoint。

## 2. 统一 Xi-B 被动入口（H/03）

|项目|源码证据与结论|
|---|---|
|任务/模型|controlled 为 MLPVAE，图像为 ConvVAE；训练主目标是重构+KL；`train_xib_unified.py:288–307,450–481`。这不是分类任务三损失，也不是主动拓扑训练|
|seed、预算、batch、优化器|CLI seed 默认0，可传 seeds/latent-dims；stage_steps 默认[1000,1000,1000]、batch64、AdamW(lr=0.001, weight_decay=0)；`:133–175,451`。没有历史 resolved_args 时，这些只算默认值|
|controlled train/probe|每阶段分别以不同 seed 合成，train offset 为 stage*101，probe 另加50000；默认每阶段各4096；`:225–256`。不同 seed 并不是不可变 sample ID/交集验证，也没有独立主模型 hval/test|
|公共数据|dSprites/Shapes3D 分别用不同 seed 的 build_*_stages；未见跨调用固定 sample ID 交集证书。CIFAR 以同一 permutation 分出不相交 train/probe index，数量默认各4096；`:213–222,259–280`；不把 CIFAR 的不交保证泛化到所有数据集|
|probe 训练与评价|`:349–388` 将已冻结 latent 分成 70% train/30% val，用 val CE 早停并在同一 val 上报告 accuracy；返回 count 在日志中写成 probe_test_samples (`:405–430`)。没有独立 probe-test；分类器 AdamW(lr=0.005,wd=0.0001)，默认最多1000步、patience50|
|跨时间可比性|split seed 在 `:409` 加入 global_step，因此不同时点 probe train/val 成员改变；不能当作固定 holdout 轨迹|
|主任务指标|训练记录 objective/rec_mse；probe 是表征可读性辅助评价。没有明确冻结本次自动权重优化的唯一主任务验证指标，不能代负责人选 accuracy/F1 或 stage label|
|checkpoint|`:517–520` 固定预算结束保存 model/dataset/latent_dim，非最佳验证 checkpoint；完整 optimizer/RNG/EMA/窗口状态未被此保存语句覆盖。本次未找到实际文件，不声称检查过其中的键|

## 3. 主动 controlled SOMA（H/05）

实参来自已有 `runs/controlled_stream_soma_manual_nll_sigma01_randomseed_20260902/run_config.json`：seeds=[854007,891047466,995002589]，trigger_methods=signal/fixed/random，三阶段各1000步，stage_size=4096，eval_size=4096，candidate_size=1536，batch=128，8×8观测，hidden_dims=[128,128]。这是 9 个历史 pilot，不是本次新运行或正式 seed 对照。

TOML 指向 NLL sigma=0.1 配置；continuous 优化器是 Adam(lr=0.001,weight_decay=0)，`trainer.py:1110–1116` 还显式优化结构 B，W_A 固定离散。runner `:229–237` 会覆盖 manual_trigger_protocol、per_parameter_rms 等，故不能只审 TOML：连续更新按总梯度 `> gradient_gate=100` 执行（`trainer.py:918–921`），候选 warmup 20步、batch64 不走相同连续更新门。`scripts.run_table1_emergence_mlp8` 缺失，实际模型构建和 `_soma_configs` 的全部覆盖仍 U。

|角色|源码种子命名空间/数量|审计结果|
|---|---|---|
|stream train|runner `:262–270,424–452`；seed_offset=0；每阶段4096样本循环供1000个batch|循环采样，不是自然单遍到达流；阶段标签是生成器真值|
|signal calibration|`:248–261`；offset=30000，最多100步|冻结的是 gate；不等于冻结旧 Psi whitening baseline。未发现不可变 role ID/交集和重放证明|
|candidate warmup|`:455–480`；offset=100000；每阶段1536样本|提前 materialize 三阶段全部数据；不能仅凭与 eval 分 seed 就认定符合已到达约束|
|candidate evaluation|同上；offset=200000，每阶段1536|与 warmup 角色分开，但 trainer `:774–825` 去掉元数据并按 cursor 循环，不校验 arrival_step，允许过早消费未来阶段数据；保留为阻塞项，本轮未修复|
|stage evaluation|`:278–284,514–536`；offset=300000，每阶段4096|在 after_s1/after_t2/final 三快照上评价全部阶段。已有表用于描述性审计；没有另外命名且冻结的主任务 hval 与最终 test|

已有评价指标是平均像素 reconstruction_mse（越低越好），以及由阶段评价派生的 loss 版 BWT/forgetting；来源 `runner.py:539–609`。没有证明本次新实验应以它作为唯一“主任务验证指标”，也没有最佳验证 checkpoint 选择规则。checkpoint 标签固定为1000/2000/3000步（`:503–511`），最终保存语句是 model_state_dict+config（`:359`），不能替代完整训练状态快照。本机这 9 份 checkpoint 文件未找到。

## 4. 其他入口的划分不可互借

`H/05.../soma/experiments/controlled_manifolds.py:21,53–77,123–130` 确有 seed permutation 的 70/15/15 train/val/test。它属于另一个 controlled manifold loader；当前流式 runner 自建 ControlledPhaseDataset，不调用该划分，不能把 70/15/15 写成上述 9 个 pilot 的事实。其 emergence 评价在 `eval_table1_emergence.py:43` 将同一 latent/factor 数组同时传给 DCI train/test，不能自动当独立测试结果。

`P/` 另有 white/edge/probe-train/probe-val/probe-test 等角色和固定参考；这是不同 signal/training 分支，不能借它的角色设计声称旧统一 runner 已完成隔离。

HAM 使用 `HamImageDataset(mode='train')`，按 semantic attributes 中 split 字段筛选；Config seed42，batch64、每epoch replacement抽64样本、2000epoch、G Adam lr1e-4 / D Adam lr1e-3。代码没有给出可直接承接此次多损失实验的独立 val 选择规程。HAM 的预训练与恢复、跨域 text embedding、图像 transform 另成模型，不作为 `03` 或 `05` 的同模型复现。

## 5. 本机 checkpoint 实际盘点

|路径（相对资料目录）|bytes|SHA-256|读取方式|
|---|---:|---|---|
|`HAM/checkpoints/checkpoint_ep1600.pth`|61174491|`479c15925244e9b91eef89cc77541e1ed66dcf4bf42e888e568a2182eff1ae6e`|仅文件哈希/ZIP目录；未反序列化|
|`HAM/checkpoints/pretrain_ae_ep200.pth`|29073027|`b1948e78be5fe3e7210ddaeb791d4f202159f1870864f4e23aa85b19347013ab`|仅文件哈希/ZIP目录；未反序列化|
|`HAM/yolov8n.pt`|6549796|`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`|仅文件哈希/ZIP目录；未反序列化|

以上为两个 HAM checkpoint 和一份 YOLO 权重，不是旧统一 Xi-B 的 30 个或主动 pilot 的9个 checkpoint。只读取文件字节做哈希与 ZIP 目录元数据；没有 torch.load/pickle.load，不声称 tensor 结构、优化器状态或模型兼容性已验证。ZIP 中另有 HAM 的 clip embeddings 和 CIFAR tar.gz，未解包为新数据或作为训练输入。

## 6. 可支持与不能支持的结论

可以支持：所见源码的损失定义、指定 legacy B 的算法路径、已有9个pilot的日志字段/范围及其配置一致性。不能支持：历史运行精确可重放、共享层梯度已测、独立 test 已冻结、最佳权重方法已选、旧阶段真值/未来数据访问已经合规。现有历史评价已被查看，不能充当新方法调参后“未见过的最终测试”。下一阶段前须由负责人冻结基线、数据角色、sample/arrival IDs、验证指标、预算与 checkpoint 选择；本阶段不做任何这种修改。
