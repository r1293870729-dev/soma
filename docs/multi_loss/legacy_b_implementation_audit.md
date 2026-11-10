# Phase 0：旧版统一 Psi / Xi-B 实现审计

审计编号：ML-P0-20261110。日期：2026-11-10。审计基线：`36167e30798afad4b6c68a1a65a7a5df2b8c773e`。

本轮仅静态读取源码、配置和已有记录，生成审计文档；未导入历史训练模块，未执行训练、smoke test、数据集下载或 checkpoint 反序列化。历史代码、配置和结果保持原字节。这里的“已定位/已核对”不代表科学验收或获准进入 Phase 1。

路径别名：`H/` = 本机 `~/Downloads/soma已有资料/soma_psi_code_handoff_20260903(1)/`；`U/` = `~/Downloads/soma已有资料/psi_xib_unified_20260824/`；`P/` = `~/Downloads/soma已有资料/psi_generalization/`；`HAM/` = 同一资料目录下的 `HAM/`。这些历史源码目前不在远端 main 中，引用为本次扫描文件的原始行号，关键文件 SHA-256 见 [信号审计](legacy_b_implementation_audit.md)。

## 结论

已定位指定的 **stage-keyed EMA / shrinkage whitening / MP edge / B-mass** 源码：`H/03_psi_xib_unified/signals_xib.py` 中 `XiBSignal`，入口为 `update_and_measure(residual, stage_name)`。对应训练调用在 `train_xib_unified.py:454–481`。这解决了“源码是否存在”的问题；仍未取得旧 30 次被动运行的原始 resolved_args、逐 run code hash 和 checkpoint，不能断言每个历史运行实际执行了这份文件的此一哈希。

## 1. 源文件、入口和依赖

|实现|源位置/入口|可核对事实|
|---|---|---|
|统一被动 Xi-B|`H/03_psi_xib_unified/signals_xib.py:100–176`；`XiBSignal.update_and_measure:135`|以真实 `stage_name` 选 `StageBaseline`；当前阶段尚未 ready 时回退到最近 ready 阶段；先 measure 再 baseline update；所有阶段共享一个 residual deque|
|白化状态|同文件 `StageBaseline:51–97`、`update:67`、`whiten:89`|按 batch 收缩协方差，再 EMA；达到 warmup batch 数 ready；eigh 构造逆平方根|
|残差投影|`ResidualProjector:24–48`|固定高斯随机矩阵，seed 混合输入和目标维度；即使目标维度大于输入也投影，不能替换为另一实现的 identity 分支|
|被动训练入口|`H/03.../train_xib_unified.py:436–481`|`XiBSignal` 读出和所有模型 backward 分离；显式传入 `train_stage.name`|
|另一旧副本|`U/signals_xib.py:102–186`，`update_and_measure:137`|相同关键算法步骤；额外输出 spectral_entropy/effective_rank。哈希与 H/03 不同，不覆盖或合并|
|主动 Psi|`H/05.../soma/runtime/psi_signal.py:53–201`；`PrecisionWeightedPsiSignal`|持续全局 EMA，未按 stage 建字典；先 update_baseline，再 update_window、signal；其正则是加法 ridge，不是 H/03 的乘性 shrinkage；需作为独立版本保留|

核心依赖为 PyTorch、torch.Tensor、collections.deque、dataclasses 和 `03/utils.py:make_generator`。统一 runner 还依赖 NumPy、torch.optim.AdamW、torch.utils.data、data.py/models.py，图像分支含 torchvision，HDF5 分支含 h5py。主动包 pyproject 声明 Python>=3.10、torch>=2.1、numpy>=1.24 等；这些是版本下界，未发现可绑定历史运行的完整锁文件。审计仅使用系统 Python 3.9.6 标准库，不证明训练依赖已安装或兼容。

## 2. H/03 的实际计算公式与顺序

令像素残差 `r = xhat_pre − x`，投影 `f=r_flat P`，`P_ab ~ N(0,1)/sqrt(d)`。在该阶段的 baseline 中：

```text
S_batch = centered(f)^T centered(f) / max(1, batch_size−1)
S_shrink = (1−alpha)*S_batch + alpha*tr(S_batch)/d*I + epsilon*I
mu_stage <- (1−gamma)*mu_stage + gamma*batch_mean
Cov_stage <- (1−gamma)*Cov_stage + gamma*S_shrink
eta = (f−mu_stage) Cov_stage^(-1/2)
```

第一次 update 直接初始化均值和收缩协方差，此后 EMA；这一步在本次读出之后执行。窗口累计的是当时 baseline 白化过的 eta；baseline 变化时不会重新白化旧行，也不会因 stage 改变自动清空全局窗口。

```text
C = centered(window_eta)^T centered(window_eta) / (W−1)
C_shape = d*C / clamp_min(tr(C),epsilon)
lambda_i = eigvalsh(C_shape)
lambda_plus = (1 + sqrt(d/max(1,W−1)))^2
B_mass = sum_i max(lambda_i/lambda_plus−1, 0)
spike_rank = count_i(lambda_i > lambda_plus)
```

源码位置：收缩/EMA `:67–87`，whiten `:89–97`，阶段与更新顺序 `:124–149`，window/trace normalization/mass `:151–172`，MP edge `:174–176`。未 ready 和未满窗返回数值 0，同时保留 readiness；不能把这个 0 当成确认“无异常”。没有每活动模态平均，没有把 mass 限制到 [0,1]。

`train_xib_unified.py:451–461` 的 CLI 默认配置：batch=64、x_window=8，所以 W=512；`gamma=1/(20*x_window)=0.00625`，alpha=0.1，warmup=20 batch，feature_dim=256，signal seed=训练 seed+777，epsilon 取类默认 1e-5。此为源码默认，不是已追回全部历史 resolved_args。

## 3. H/05 主动路径的差别

`trainer.py:489–502`：projector→baseline update→window update→signal。`psi_signal.py:99–119` 将 baseline 与窗口输入 detach；`:_regularized_cov:189–201` 使用 `Cov + alpha*tr(Cov)/d*I + epsilon*I`。默认 projector 在 feature_dim>=输入维度时返回原 residual；实际维度由初始化重设。history 中 threshold≈1.5628054，与 d=64、W=1024 的 MP 公式相符；TOML 的 feature_dim=128 只是上限，在 8×8 输入和 identity 分支下实际为 64。

历史 TOML `:85–94` 指定 W=1024、gamma=0.00625、alpha=0.1、mp_edge、mass、warmup=20、observation 残差空间。`psi_space=observation` 使用保存 forward 的残差；feature 分支还会在 step 后通过 encoder 重算 reconstruction features，不能把两者时序混为相同。结构接受/重置行为是主动策略的一部分，不能以新 logger 改变。

因此 `H/03`、`U/` 与 `H/05` 不是可互换的一个类。多损失协议明确指向 stage-keyed 路径；若负责人要把它接入主动模型，需先确认该适配范围，而不是静默延用 H/05 或替换公式。

## 4. 必须单独列出的非目标实现

|位置|版本特征|本任务处理|
|---|---|---|
|仓库 `src/soma_b/signal.py:20,88`|`SIGNAL_ID="B_mass_fixed_white_empirical_v1"`，`BMonitor` 固定白化，独立经验顺序统计 edge，NumPy float64，无 stage 参数|当前仓库主 B 实现，但本次多损失实验明确不得用它自动替代旧版|
|仓库 `src/soma_protocol/signal.py:24–52,85–119`|`FixedReference`、`psi_window`、`calibrate_edge` 的固定参考/经验 edge CPU 参考|不属于旧 stage-keyed EMA/MP runner|
|`P/signals.py:32–246`|`SpectralCondensationSignal`；`fit_reference`、`fit_edge`，固定均值/逆平方根和经验边缘；只读出而不更新 EMA|另一非目标实现；不能因为名称含 Psi 就选它|
|`H/02.../signals.py:86–140`|`MultiSpectralXi`，stage 基线下同时记录 A/edge、B/mass、C/gap|相关历史比较器；应显式选 mass，不能把 edge 或 entropy 改名为 B|
|`H/01.../losses_pw.py:312–783`|`PrecisionWeightedResidualSignal`，可配置多种谱读出，且 `:578–652` 有可微 current-batch 接口|含训练耦合选项；本轮不选择、不调用|

仓库 `configs/signal_b_final.json` 的全局 author-selected B 不会被本轮更改；多损失专用协议对旧实现的要求仅在本次审计中记录，未改变任一信号配置。

## 5. logger 可行性：只设计接口，不实现

旧版信号已经通过 detach/no_grad 与模型损失图隔离。可在已有一次合法调用返回后复制 XiBLog/PsiLog 及状态摘要，关联 `step / checkpoint或模型状态ID / 源文件SHA / 配置SHA / stage与readout_source / readiness / window_count / mass / edge / spike_rank`。旧接口没有返回完整谱；未来若需谱，必须从同一次既有计算中旁路记录或对保存的状态副本离线重算，先验证逐字节读数和状态轨迹一致。本阶段没有添加代码或接口。

必须区分：记录既有值不应影响 optimizer；但主动实现本来就用 Psi 影响结构决策，因此不能声称“B 对模型轨迹没有影响”。禁止给 logger 再调用一次 `update_and_measure`，否则会推进 EMA/窗口；也不能再次执行随机 forward 或通过修改 signal state 采谱。历史随机 projector 使用独立 generator，但主动模型 forward、candidate warmup 和缺失工厂的 RNG 行为仍需未来验证。仅凭 no_grad 不能证明整个 logger 无副作用。

stage-keyed 真实阶段标签、持续 EMA 和跨参考混合窗口均是此旧算法的原样性质。本次定位不认证它符合新手册的因果/固定参考协议，亦未修复或替换它。

## 6. 源文件与归档哈希

|文件（相对资料目录）|SHA-256|
|---|---|
|`soma_psi_code_handoff_20260903(1)/03_psi_xib_unified/signals_xib.py`|`c28159db1116331a5d708062a585b408f8fd1e9a856188b6a3b8578e730e92e0`|
|`soma_psi_code_handoff_20260903(1)/03_psi_xib_unified/train_xib_unified.py`|`64f68c7c68a2649d60f977d882f9ac71af0c1e70ab03969b4eea8e52cd7bd495`|
|`psi_xib_unified_20260824/signals_xib.py`|`d85294ee553d796babff5277a1d71e7a39dc5baf7d9c69b920d12bf368676c71`|
|`soma_psi_code_handoff_20260903(1)/05_soma_psi_gated/soma/runtime/psi_signal.py`|`0ac7b8089f9bface0c02159e10903becf7c1cce83fa4d685b273579cb4659da5`|
|`soma_psi_code_handoff_20260903(1)/05_soma_psi_gated/soma/runtime/losses.py`|`c9fecd298c27190bbbbf3d17016bf49df5c50fff3254eec13b42bf1c5dd29930`|
|`soma_psi_code_handoff_20260903(1)/05_soma_psi_gated/soma/training/trainer.py`|`409ee6732ef1fa94cf0439eed50336d815fe3cfdec9c8db21b7a8e5b9553a4f5`|
|`soma_psi_code_handoff_20260903(1)/05_soma_psi_gated/scripts/run_controlled_stream_soma_psi.py`|`f6539e9e1e5a9f20d02b5bad29de14e34342f12e0e98306de5800be8eb9e927d`|
|`soma_psi_code_handoff_20260903(1)/05_soma_psi_gated/configs/controlled_stream_soma_trigger_controls_nll_sigma01_randomseed_20260902.toml`|`14e85fe1d4cabe3fae969b86a77b2d67a5f9c9a2db18d73d4176bc892557e641`|
|`HAM/ham_trainer.py`|`2d30a36c21155d3ed253025b2c3824e7cc8608b78df52ecea860cc8afd5c0d8c`|
|`HAM/losses.py`|`73ee08ad220633f5cc100ae918d0c97c6152d6f0046934281cb4c5085967a380`|
|`psi_generalization/signals.py`|`c6086fc8d958e56f648d7c962b72665cd625ee1fb9866c35ef4aae2d0f47eb27`|

完整资料 ZIP：`soma已有资料.zip`，498213350 bytes，SHA-256 `abaa3916d1b921b078e738b763660ca19300027b4a8b0fe11cad30e400faa602`。ZIP 列表 10546 项包含目录和 macOS 资源分支，不表示 10546 项实验资产；排除 `__MACOSX`/AppleDouble 后 Python 文件为 85 个，均与目录副本逐字节一致。另一个 handoff 副本所有已索引对应文本/配置/记录也相同，不能重复计数。

远端 main 和两条既有审计分支均未含这些历史目录或源码 ZIP；两审计分支各有 9 个 Python 文件，均为管理/参考/测试层，未找到缺失 MLP 构建脚本。仅据本次可访问范围陈述缺件，不断言其他机器不存在。参见 [数据审计](data_and_split_audit.md) 和 [阻塞项](phase0_blockers.md)。
