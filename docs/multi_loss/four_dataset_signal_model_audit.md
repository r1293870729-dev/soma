# 四数据集 Xi-B 信号验证：模型构建与数据入口资产审计

审计日期：2026-11-11  
审计来源：用户上传的 `soma_psi_code_handoff_20260903(1).zip`。本次逐项扫描 ZIP 内 124 个非 macOS 元数据文件，其中 56 个 Python 文件；检索模型构建入口、数据集构建入口、旧版 Xi-B 调用和缺失导入。仅静态读取，未执行训练、数据下载或 checkpoint 加载。

## 1. 核心结论

**四个数据集的旧版统一 Xi-B 信号验证，已有可用的历史模型构建与数据入口：**

- 入口：`03_psi_xib_unified/train_xib_unified.py`
- 模型构建：`build_model(args, latent_dim, device)`
- 统一信号：同目录 `signals_xib.py` 的 `XiBSignal.update_and_measure(residual, stage_name)`
- 数据构建：同目录 `data.py` 的 `build_controlled_bundle`、`build_dsprites_stages`、`build_shapes3d_stages`、`build_cifar_stages`

这条 03 路径正好覆盖 controlled、dSprites、3D Shapes、CIFAR 四类输入，并且 runner 说明为 loss-only Xi-B：训练目标是 reconstruction + beta-KL，Xi-B 是训练后残差结构监测，不作为可反向传播训练损失。

**但主动 SOMA（05）的模型工厂缺失问题仍然存在：** `05_soma_psi_gated/scripts/run_controlled_stream_soma_psi.py` 导入 `scripts.run_table1_emergence_mlp8` 中的 `_build_mlp_soma` 与 `_soma_configs`。这份 `run_table1_emergence_mlp8.py` 不在本次 ZIP 的文件清单中；全 ZIP 文本检索只找到该导入和文档中的文件名引用，没有找到函数定义。因此不能声称已找回 05 主动 SOMA 工厂。

就当前“在四个数据集上验证旧版信号”的目标而言，**应先以 03 统一 Xi-B runner 为共同实验入口，不要为此强行补写 05 的缺失工厂。** 这不等于 03 与 05 是同一个模型或同一套训练目标。

## 2. 模型构建入口

### 2.1 Controlled

文件：`03_psi_xib_unified/train_xib_unified.py`

- `MLPVAE(observation_dim, latent_dim, hidden_dim, hidden_layers)`
- 编码器为若干 Linear + GELU；分别输出 latent mean 与 logvar。
- 解码器由 latent 维度映射回 observation_dim。
- `build_model` 在 dataset=controlled 时实例化 MLPVAE。
- 默认参数（可由 CLI 覆盖）：observation_dim=64、intrinsic_dim=8、controlled_latent_dim=4、hidden_dim=32、hidden_layers=1、noise_std=0.02。

Controlled 三阶段为 circle (`s1`) → torus (`t2`) → torus with local split (`local_split`)。阶段标签包括 theta、phi、split、stage。该数据集是代码生成的受控向量观测，不是外部图像数据集。

### 2.2 dSprites

文件：`03_psi_xib_unified/models.py`、`train_xib_unified.py`、`data.py`

- 模型：`ConvVAE(input_shape=(1,64,64), channels=(32,64,128), latent_dim)`。
- 编码器使用 stride-2 Conv2d + GELU；线性层输出 mean/logvar；decoder 使用 ConvTranspose2d。
- 重构 likelihood 为 Bernoulli，使用 BCEWithLogits。
- 三阶段：position → position_shape → position_shape_orientation。
- 数据筛选固定 scale=3；第一阶段固定 shape/orientation，第二阶段放开 shape，第三阶段再放开 orientation。
- probe 因子为 x、y、shape、orientation。

### 2.3 3D Shapes

文件：同上。

- 模型：`ConvVAE(input_shape=(3,64,64), channels=(32,64,128,256), latent_dim)`。
- 重构 likelihood 配置为 Gaussian，但当前实现的训练重构项是 sigmoid 输出上的 MSE，并非显式 Gaussian NLL。
- 三阶段：background → background_object → background_object_shape_pose。
- 阶段逐步开放 floor_hue、wall_hue、object_hue、scale、shape、orientation。
- probe 标签覆盖六个生成因子。

### 2.4 CIFAR

文件：同上。

- 模型：`ConvVAE(input_shape=(3,32,32), channels=(64,128,256), latent_dim)`。
- 重构 likelihood 配置为 Gaussian，当前实现为 sigmoid 输出 MSE。
- 三阶段：CIFAR-10 clean train → CIFAR-C motion_blur severity 3 → CIFAR-C frost severity 3。
- probe 标签为 class 与 domain。
- CIFAR-10 由 torchvision 数据入口读取；CIFAR-C 需要对应 corruption 的 `.npy` 和 `labels.npy` 文件。

## 3. 统一训练目标与信号边界

03 runner 中模型训练目标为：

```text
L_train = L_reconstruction + beta_kl * KL_standard_normal
```

默认 beta_kl=0.001。probe 分类器在冻结 latent 上单独训练，不回传 VAE，因此不能将 probe CE 误记为第三个 VAE 联合训练损失。

Xi-B 通过训练步骤中保存的 reconstruction residual 调用 `XiBSignal.update_and_measure(residual, stage_name)`。本实验的信号验证应保持：
- 原版 `signals_xib.py` 不改公式、不换类、不替换 stage-keyed EMA / shrinkage whitening / MP edge / B-mass。
- 不把 B/mass 加入训练总损失。
- 不对 B 反向传播。
- 同一训练 step 只推进一次旧信号状态；日志不得额外调用 update。
- 四个数据集分别记录 dataset、stage、step、seed、checkpoint/model-state ID、signal source hash 和 config hash。

## 4. 数据入口与关键复核风险

|数据集|代码中的训练/探针构建|必须复核的事项|
|---|---|---|
|controlled|训练阶段 seed 为 seed + stage_offset*101；probe 使用额外 probe_seed_offset（默认 50,000）|不同 seed 不等于严格样本 ID 去重证明；需冻结生成样本 ID 或确定性索引并做交集检查|
|dSprites|训练和 probe 分别调用三阶段 builder，使用不同 seed|builder 可能在合法因子子集中有放回抽样；必须保存原始数据索引并检查 train/probe 交集，不能仅凭 seed 不同宣称独立|
|3D Shapes|同样按阶段过滤因子，再以 seed 抽样|需保存原始 HDF5 索引；检查连续块抽样造成的重复和跨角色交集|
|CIFAR|CIFAR clean 阶段在同一数据集上用 permutation 划分 train/probe；CIFAR-C corruption 按 severity=3 取切片|要审计 CIFAR-C 的样本身份、与 clean 数据源的对应关系和 train/probe 角色；保存明确 index 清单|

现有 runner 的 probe 逻辑会从 probe latent 再拆 train/validation 并用 validation CE 早停；同一验证分割还参与报告 accuracy。因此新实验不能把该 accuracy 直接称作独立 test。必须为四个数据集定义固定、不可随 step 改变的 probe-train / probe-val / final-test 身份清单；任何权重、阈值、checkpoint 选择只可使用 validation，最终 test 只做一次最终报告。

## 5. 建议的四数据集实验组织

建议先保持 03 历史统一 runner 的结构，按数据集分别执行同一信号审计：

1. controlled：验证信号对已知 intrinsic dimension 从 1→2 及 local split 变化的响应。
2. dSprites：验证在 position、shape、orientation 因子逐步增加时的阶段响应。
3. 3D Shapes：验证背景因子到物体颜色/尺度，再到 shape/orientation 扩展时的响应。
4. CIFAR：验证 clean → motion_blur → frost 的域/退化变化响应。

每个数据集都必须有独立的数据路径配置、数据索引清单、seed 清单和 stage 定义。先做只读数据资产核验与 split manifest；之后做小规模工程 smoke test；通过后才开展正式多 seed 实验。四个数据集不应直接合并成一个指标均值，先分别报告，再做跨数据集汇总。

## 6. 缺失资产和当前结论

- 已找到：03 的 MLPVAE、ConvVAE、统一 build_model、四数据集构建器、旧版 Xi-B 调用位置。
- 未找到：05 的 `run_table1_emergence_mlp8.py` 与 `_build_mlp_soma` / `_soma_configs` 定义。
- 未验证：数据文件是否在当前机器可用、各数据集样本索引是否严格隔离、完整训练依赖是否安装、四数据集历史运行是否可逐 seed 重放。
- 未执行：模型训练、smoke test、数据下载、checkpoint 加载。

**决策建议：** 若目标是验证旧版统一 Xi-B 信号在 CIFAR、dSprites、3D Shapes 和 controlled 四类数据上的泛化，优先沿用 03 入口和它的原始模型/数据构建代码。05 的缺失工厂只在未来明确要复现主动 SOMA 结构编辑实验时才必须找回；不得自行重写替代。
