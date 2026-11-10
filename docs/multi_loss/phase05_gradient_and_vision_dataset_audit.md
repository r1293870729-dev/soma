# 05 主动 SOMA 梯度诊断与视觉数据集入口审计

审计日期：2026-11-11  
审计对象：用户提供的 `soma_psi_code_handoff_20260903(1).zip` 和 `soma已有资料.zip` 中的 05 源码副本。静态检查源码、TOML、历史 run_config 和 CSV。未修改源码、未启动训练、未下载数据。

## 1. 结论摘要

本轮无法诚实地产出四项损失的数值梯度范数/余弦矩阵，也无法确认 05 在 CIFAR、3D Shapes 上有可运行的完整模型与数据配置。原因是当前归档存在关键缺失：

1. `scripts/run_controlled_stream_soma_psi.py` 导入 `scripts.run_table1_emergence_mlp8` 的 `_build_mlp_soma` 和 `_soma_configs`，但该模块及函数定义不在两个已扫描归档中。
2. 当前 controlled run 的实际模型构建依赖该缺失工厂。虽然包内 `soma.training.factory.build_model` 能构建 Conv SOMA，但它不是历史 runner 所引用的 MLP SOMA 工厂；不能拿它替代并将所得梯度冒充历史模型结果。
3. 05 包内只实现并导出 dSprites loader；没有 CIFAR / CIFAR-C、3D Shapes loader 实现，也没有对应 image-stream runner/config 文件。
4. 05 README 提到 `scripts/run_image_stream_soma_trigger_controls.py`、`scripts/run_four_dataset_soma_trigger_controls_20260901.ps1` 和若干旧实验配置，但这些脚本/配置不在归档文件清单中。
5. 历史 `stream_train_history.csv` 只记录 reconstruction、KL、总目标等字段，没有 separation、usage 原始值、加权值、逐项梯度范数或余弦矩阵。因此历史日志不能反推出缺失的梯度诊断。

因此：**梯度范数和冲突结论状态为 NOT MEASURED；不能判断有无持续冲突，也不能启用 PCGrad。** 这不是“没有冲突”，而是没有合格测量。

## 2. 四个损失在代码中的准确定义与权重

源码：`soma/training/trainer.py::_loss_components`，`soma/runtime/losses.py::posterior_block_regularization`。

实际 controlled TOML：
`configs/controlled_stream_soma_trigger_controls_nll_sigma01_randomseed_20260902.toml`

| 项目 | 原始损失 | 系数 | 当前 run 是否 active |
|---|---|---:|---|
| Reconstruction | Gaussian NLL，per-pixel variance = observation_sigma²，按每样本求和再 batch 平均 | 1 | 是 |
| Structured KL | common latent mixture prior KL + side latent isotropic Gaussian KL | beta_kl=0.01 | 是 |
| Separation | common/side posterior mean 的归一化交叉协方差平方和 | lambda_sep=0.1 | 是 |
| Usage | 对平均 node responsibility 未达到 rho_use/K 的部分施加平方惩罚 | lambda_use=0.1 | 是 |

本 run 配置还包括：
- `lambda_dis=0`：disentanglement 不启用。
- `lambda_orth=0`：orthogonality 不启用。
- `lambda_sym=0`：symmetry 不启用。
- `lambda_cmp=0.1`：complexity 结构代价存在，但不进入 `stage_one_objective.backward()`。

总训练梯度来自：

```
stage_one = rec + 0.01 * KL + 0.1 * separation + 0.1 * usage
stage_one.backward()
```

Complexity 的源码定义依赖结构状态（节点数、有效边数、common dimension），结构张量由离散拓扑状态构成；它作为候选结构目标，不应与四个连续参数损失一起做 GradNorm。

## 3. 共享参数范围：不能笼统称全模型参数完全共享

模型源码：`soma/models/networks.py`。

Encoder 参数分组：
- `encoder.backbone.*`：四项损失均可能经此共享特征干路回传；建议作为首要 GradNorm 诊断参数集 W_shared。
- `encoder.mu_c.*`：重构、KL、separation、usage 均可能通过 common posterior/responsibility 影响它。
- `encoder.logvar_c.*`：重构、KL 可能影响；separation/usage 当前实现基于 mean/responsibility，不直接依赖 logvar，因此其梯度覆盖需实测。
- `encoder.mu_r.*`：重构、KL、separation 可能影响；usage 通过 common responsibilities 不直接依赖 side mean。
- `encoder.logvar_r.*`：主要由重构采样路径与 KL 影响，separation/usage 当前定义不直接依赖它。

Decoder 参数：
- Reconstruction 直接影响 decoder。
- KL、separation、usage 的定义不直接依赖 decoder 输出，通常对 decoder 无梯度。

因此应分别报告：
1. backbone shared trunk 上的每项梯度范数与余弦；
2. common/side posterior heads 上每项梯度覆盖；
3. 全 encoder 汇总梯度；
4. decoder 梯度只作为 reconstruction 路径核对，不可把大量零梯度参数简单拼接后误读。

若 GradNorm 的权重目标定义为“共享参数梯度平衡”，首选 W_shared = `encoder.backbone` 参数；同时附报全 encoder 结果。最终需由负责人/实现者确认是否把 posterior heads 纳入 W_shared。

## 4. 为什么当前不能计算梯度数值

当前 runner 顶部导入：

```python
from scripts.run_table1_emergence_mlp8 import _build_mlp_soma, _soma_configs
```

但是 `run_table1_emergence_mlp8.py` 不存在于两个已扫描 ZIP 的 05 文件树中。runner 因此无法按历史路径构建其原始 MLP SOMA 模型。

虽然 `soma/training/factory.py::build_model` 存在，但它构建的是 ConvEncoder + ConvDecoder；将其用于该 controlled stream 后得到的梯度数字不是历史 MLP runner 的梯度，属于另一个模型实现。为遵守不替换历史模型的要求，本次拒绝用它生成伪历史结果。

另外，四个损失的随机采样项（reparameterization）需要固定 seed 和固定 batch；单个 batch 的结果不足以判断“持续冲突”。正式诊断应在同一 checkpoint/初始化、相同固定 batch 序列上测多个 batch，并覆盖 s1、t2、local_split 阶段。

## 5. 允许执行的梯度诊断协议（缺失工厂恢复后）

对每个阶段、多个固定 batch、至少 3 个配对 seed：
1. 加载同一个历史 MLP SOMA 模型与 checkpoint；不做参数更新。
2. 固定随机数状态，运行一次 forward，取四个 raw loss。
3. 对同一 `encoder.backbone` 参数分别计算：
   - raw gradient：`grad(L_i)`
   - weighted gradient：`grad(c_i * L_i)`，其中 c = (1, 0.01, 0.1, 0.1)。
4. 记录每项梯度 L2 norm、非零梯度参数占比、逐层 norm。
5. 对每对梯度计算 cosine similarity；只在两者均非零时定义 cosine。
6. 跨 batch、阶段、seed 报告均值/标准差、中位数、负余弦比例和连续负冲突区间。
7. 不改变优化器状态，不更新模型，不把 B/Psi 信号放入目标。
8. 若只有零星负 cosine，不启用 PCGrad；只有跨多个 batch/阶段持续显著负冲突才进入 PCGrad 试验。

必须输出 `gradient_norms.csv`、`gradient_cosines.csv`、`gradient_coverage.csv`、`gradient_diagnostic_protocol.md`。明确 raw 与 coefficient-weighted 两套结果，避免把原始损失梯度与当前实际更新贡献混淆。

## 6. 05 视觉数据集入口核验

| 数据集 | 05 源码实际资产 | 模型/配置状态 | 结论 |
|---|---|---|---|
| dSprites | `soma/data/dsprites.py`；`load_dsprites_splits`；`factory.build_dsprites_dataloaders` | 通用 SOMA ConvEncoder/ConvDecoder 可配置；但归档内没有专用 `dsprites_soma_probe.toml` 或 image-stream runner | 有数据 loader 组件，不足以证明端到端历史实验可运行 |
| 3D Shapes | 无 05 专用 loader | README 提到 image-stream runner 和输入 3×64×64，但脚本/config 缺失 | 不可运行复现 |
| CIFAR/CIFAR-C | 无 05 专用 loader | README 提到 clean CIFAR-10、motion blur severity 3、frost severity 3，输入 3×32×32；脚本/config 缺失 | 不可运行复现 |
| Controlled | `run_controlled_stream_soma_psi.py` 有三阶段 ControlledPhaseDataset；配置存在 | 实际 runner 所需 MLP builder 缺失；TOML 是 1×8×8，hidden dims 由 CLI 指定 | 数据构造代码存在，但历史端到端 runner 被缺失导入阻断 |

README 中记录的历史意图：
- dSprites：position → position+shape → position+shape+orientation；1×64×64；Bernoulli。
- 3D Shapes：background → background+object → all factors；3×64×64；Gaussian。
- CIFAR：clean CIFAR-10 → CIFAR-C motion_blur severity 3 → CIFAR-C frost severity 3；3×32×32；Gaussian。

这些是 README 描述的预期配置，不是当前归档内可执行的完整实现。不能将其写成“已找到可运行代码”。

## 7. 必须找回的资产清单

P0：
1. `05_soma_psi_gated/scripts/run_table1_emergence_mlp8.py`（至少含 `_build_mlp_soma`、`_soma_configs`）。
2. `scripts/run_image_stream_soma_trigger_controls.py`。
3. `scripts/run_four_dataset_soma_trigger_controls_20260901.ps1`。
4. README 所引用的 dSprites / Shapes3D / CIFAR 配置文件及其数据适配器。
5. 当时运行对应的 resolved config、checkpoint 和环境记录。

dSprites：
- `soma/data/dsprites.py` loader 已有；需要找到对应 TOML（README 指向 `configs/dsprites_soma_probe.toml`，但本次归档缺失）和数据文件路径/校验值。

3D Shapes / CIFAR：
- 需要找回 03/generalization 的原始 dataset/stage 构建器，并验证 05 的模型输入预处理契约；仅可复用数据集构建，不可复用/替换 03 的 VAE 模型。
- 需要找回当时 05 image-stream runner 的完整源码和每数据集 resolved config。

## 8. 当前阶段判定

| 工作 | 状态 |
|---|---|
| 四项损失定义/系数源码确认 | 完成 |
| 共享 Encoder 参数路径静态确认 | 完成 |
| 数值梯度范数 | BLOCKED：缺历史模型工厂 |
| 梯度余弦矩阵 | BLOCKED：缺历史模型工厂与固定批次协议 |
| 持续冲突结论 | NOT MEASURED；不得写“有”或“没有” |
| dSprites loader | 已有组件，端到端配置缺失 |
| 3D Shapes loader/config | 缺失 |
| CIFAR/CIFAR-C loader/config | 缺失 |
| 正式训练 / PCGrad | 未启动 |

**停止规则：** 在历史 MLP 工厂和 05 image-stream runner/config 恢复前，不启动正式梯度实验，不以 Conv factory 代替 MLP，不创建伪造的视觉 SOMA 实验入口。
