# 注释： 启用未来版本注解语义，避免类型注解在旧 Python 行为下提前求值。
from __future__ import annotations

# 注释： 导入模块 `math`，因为后续代码会使用其中的函数或类。
import math
# 注释： 导入依赖 `from collections import deque`，避免在后续代码中使用冗长模块前缀。
from collections import deque
# 注释： 导入依赖 `from dataclasses import dataclass`，避免在后续代码中使用冗长模块前缀。
from dataclasses import dataclass

# 注释： 导入模块 `torch`，因为后续代码会使用其中的函数或类。
import torch
# 注释： 导入依赖 `from torch import Tensor`，避免在后续代码中使用冗长模块前缀。
from torch import Tensor

from utils import make_torch_generator


# 注释： 把下面的日志结构声明为 dataclass，因为它只负责承载 Xi 计算结果，不需要手写初始化函数。
@dataclass
# 注释： 定义类 `PWLog`，用于承载每次 Xi 读数的所有诊断字段。
class PWLog:
    # 注释： 声明字段 `lambda_max`，表示 白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数；dataclass 会把它纳入日志对象。
    lambda_max: float
    # 注释： 声明字段 `xi_pw`，表示 精度加权残差各向异性信号 Xi；用于衡量残差是否集中到少数方向；dataclass 会把它纳入日志对象。
    xi_pw: float
    # 注释： 声明字段 `tau_null`，表示 空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代；dataclass 会把它纳入日志对象。
    tau_null: float
    # 注释： 声明字段 `trace_baseline`，表示 baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移；dataclass 会把它纳入日志对象。
    trace_baseline: float
    # 注释： 声明字段 `feature_dim`，表示 残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它；dataclass 会把它纳入日志对象。
    feature_dim: int
    # 注释： 声明字段 `window_ready`，表示 残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi；dataclass 会把它纳入日志对象。
    window_ready: bool
    # 注释： 设置 `lambda_edge: float`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向）为 `0.0`；这样后续步骤可以复用该状态而不是重复计算。
    lambda_edge: float = 0.0
    # 注释： 设置 `spike_rank: int`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
    spike_rank: int = 0
    # 注释： 设置 `baseline_ready: bool`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
    baseline_ready: bool = True
    # 注释： 设置 `baseline_warmup_batches: int`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
    baseline_warmup_batches: int = 0
    # 注释： 设置 `threshold_mode: str`（Xi 阈值模式；用于切换经验 tau_null 与理论 MP edge 两种读数）为 `"empirical"`；这样后续步骤可以复用该状态而不是重复计算。
    threshold_mode: str = "empirical"
    # 注释： 设置 `abs_e_before: float`（可吸收性信号中快速适应前，残差在最强方向 u 上的投影能量 E_before；旧 PW/MP 信号不用该字段所以默认为 0）为 `0.0`。
    abs_e_before: float = 0.0
    # 注释： 设置 `abs_e_after: float`（可吸收性信号中临时快速适应 n_fast 步后，同一方向 u 上仍残留的投影能量 E_after；它直接决定 Xi_abs 是否非零）为 `0.0`。
    abs_e_after: float = 0.0
    # 注释： 设置 `abs_rho: float`（可吸收性信号中的压缩比例 rho=(E_before-E_after)/E_before；越接近 1 表示连续训练越能吸收该残差方向）为 `0.0`。
    abs_rho: float = 0.0
    # 注释： 设置 `tau_abs: float`（可吸收性信号的容差阈值；快速适应后残留能量低于它时被视为噪声或可接受误差）为 `0.0`。
    tau_abs: float = 0.0
    # 注释： 设置 `div_trace: float`（快慢分歧信号中的迹项 tr(A_s^{-1} C_f)；它表示用慢记忆这把尺子衡量快记忆形状时的总体尺度）为 `0.0`。
    div_trace: float = 0.0
    # 注释： 设置 `div_logdet: float`（快慢分歧信号中的对数行列式项 logdet(A_s)-logdet(C_f)；它衡量快慢协方差体积差异）为 `0.0`。
    div_logdet: float = 0.0
    # 注释： 设置 `div_gamma_fast: float`（快协方差 EMA 更新率 gamma_f；越大越快响应新残差形状）为 `0.0`。
    div_gamma_fast: float = 0.0
    # 注释： 设置 `div_gamma_slow: float`（慢协方差 EMA 更新率 gamma_s；越小越代表长期旧常态）为 `0.0`。
    div_gamma_slow: float = 0.0
    # 注释： 设置 `div_trace_fast: float`（快协方差 C_f 的迹；用于审计快记忆残差总体能量是否异常放大）为 `0.0`。
    div_trace_fast: float = 0.0
    # 注释： 设置 `div_trace_slow: float`（慢协方差 C_s 的迹；用于审计慢记忆是否已经追上新残差形状）为 `0.0`。
    div_trace_slow: float = 0.0
    # 注释： 设置 `info_entropy: float`（第三信号中标签因子 f 的经验熵 H(f)；它是残差可预测信息的理论上界）为 `0.0`。
    info_entropy: float = 0.0
    # 注释： 设置 `info_cond_entropy: float`（第三信号中小分类器估计的条件熵 H(f|epsilon)；越低说明残差越能预测未编码因子）为 `0.0`。
    info_cond_entropy: float = 0.0
    # 注释： 设置 `info_accuracy: float`（第三信号中残差因子分类器在当前样本上的准确率；用于辅助判断 Xi_info 是否来自真实可分性）为 `0.0`。
    info_accuracy: float = 0.0
    # 注释： 设置 `info_classes: int`（第三信号中参与计算的有效类别数；类别少于 2 时熵为 0，Xi_info 也应为 0）为 `0`。
    info_classes: int = 0


# 注释： 定义类 `ResidualFeatureProjector`，用于把残差投影、白化、窗口谱读数封装到独立组件中。
class ResidualFeatureProjector:
    """Fixed random residual readout h for high-dimensional observations.

    If feature_dim >= residual_dim, this is the identity map. Otherwise residuals
    are projected with a deterministic Gaussian matrix scaled by sqrt(feature_dim).

    这里的 h(epsilon) 不是模型可学习层，而是一个固定读出矩阵：
    1. 高维图像/观测残差如果直接做协方差，维度太高且计算代价大；
    2. 固定随机投影把残差压到 d_epsilon=feature_dim 维，后续 Xi、MP edge、
       lambda_max 都在同一个低维残差特征空间里计算；
    3. 投影矩阵一旦生成就必须复用，否则不同 step 的 Xi 会混入“投影空间变化”
       这个额外随机因素，不能再解释为残差几何变化。
    """

    # 注释： 定义函数 `__init__`：保存初始化参数并准备内部缓存，真正依赖数据的矩阵稍后懒加载。
    def __init__(self, feature_dim: int | None, seed: int = 0) -> None:
        # 注释：`feature_dim` 是残差投影后的目标维度 d_epsilon。比如原始残差是 64 维或 3072 维，
        # 如果 `feature_dim=128`，后续 Xi 实际使用的是形状 `[batch, 128]` 的残差特征。
        # 若它为 None 或不小于原始残差维度，就不投影，直接使用展平残差。
        self.feature_dim = feature_dim
        # 注释：`self.seed` 是用户/实验给定的基础随机种子。它只负责让随机投影矩阵可复现；
        # 这个投影矩阵不是训练参数，不会被 optimizer 更新。
        self.seed = int(seed)
        # 注释：`self._matrix` 缓存固定随机投影矩阵 h。初始化时还不知道残差展平后的真实维度
        # input_dim，所以先设为 None；第一次看到 residual 后再生成匹配尺寸的 `[input_dim, feature_dim]` 矩阵。
        # 后续所有 step 复用同一矩阵，保证 Xi 的变化来自残差本身，而不是读出矩阵变化。
        self._matrix: Tensor | None = None
        # 注释：`self._input_dim` 记录当前缓存矩阵对应的原始残差维度。这样如果之后换了观测形状
        # 或残差维度，代码能检测到旧矩阵不再匹配，并重新生成正确尺寸的矩阵。
        self._input_dim: int | None = None

    # 注释： 定义函数 `__call__`：把原始残差展平并投影到 d_epsilon 维残差特征空间。
    def __call__(self, residual: Tensor) -> Tensor:
        # 注释：把每个样本的残差展平成一行。若 residual 是 `[B, C, H, W]` 图像残差，
        # 这里会变成 `[B, C*H*W]`；若本来就是向量观测，则保持为 `[B, D]`。
        # Xi 需要对“样本 × 残差维度”的矩阵估计协方差，所以必须先转成二维。
        flat = residual.flatten(start_dim=1).float()
        # 注释：`input_dim` 是展平后的原始残差维度 D_epsilon_raw，例如 32x32 RGB 图像就是 3072。
        # 它决定投影矩阵的行数，也决定当前缓存的矩阵还能不能继续复用。
        input_dim = flat.shape[1]
        # 注释：如果不要求投影，或目标维度已经大于等于原始维度，就直接返回原始残差。
        # 这样可以避免“升维随机投影”引入不必要噪声；投影只用于降维读出。
        if self.feature_dim is None or self.feature_dim >= input_dim:
            # 注释：返回展平残差本身，此时 h 是恒等映射，d_epsilon=input_dim。
            return flat
        # 注释：获取固定随机投影矩阵 h。这里不会每次重新采样，而是优先复用缓存；
        # 只有 input_dim、device 等发生变化时才重新生成。
        matrix = self._get_matrix(input_dim, flat.device, flat.dtype)
        # 注释：执行线性投影 `[B, input_dim] @ [input_dim, feature_dim] -> [B, feature_dim]`。
        # 这个输出就是后续 precision-weighted Xi 使用的残差特征 h(epsilon)。
        return flat @ matrix

    # 注释：按输入维度生成或复用固定随机投影矩阵 h。这个函数的核心目标是：
    # 同一实验配置下矩阵必须稳定复现；不同输入维度/投影维度下矩阵必须彼此区分。
    def _get_matrix(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        # 注释：如果已经有缓存矩阵，并且它的输入维度和设备都匹配当前残差，就直接复用。
        # 复用是必要的：否则 Xi 曲线会受随机投影矩阵重新采样影响，无法稳定比较不同 step。
        if self._matrix is not None and self._input_dim == input_dim and self._matrix.device == device:
            # 注释：返回缓存矩阵；这里不会重新调用随机数发生器。
            return self._matrix
        # 注释：创建一个 CPU 上的独立随机数发生器，用来生成固定随机投影矩阵。
        # `manual_seed(...)` 的意思是手动设置随机数发生器的初始种子；相同 seed 会生成相同的随机矩阵，
        # 因而同一实验可复现。这里不用全局 `torch.manual_seed`，是为了避免投影矩阵采样影响模型初始化、
        # dataloader 或其他随机过程。
        # 种子写成 `self.seed + input_dim * 1009 + int(self.feature_dim or 0)` 是为了把三个因素混合进去：
        # 1. `self.seed`：实验级基础种子，保证不同实验 seed 得到不同投影；
        # 2. `input_dim * 1009`：把原始残差维度编码进种子，避免 64 维残差和 3072 维残差误用同一随机序列；
        #    1009 是一个质数，用作简单的混合系数，降低不同维度组合碰撞到相近 seed 的概率；
        # 3. `int(self.feature_dim or 0)`：把目标投影维度 d_epsilon 也编码进去，避免 feature_dim=64 和
        #    feature_dim=128 时生成同一前缀矩阵。
        # 这样做不是训练技巧，而是工程上的确定性命名：同一 `(seed, input_dim, feature_dim)` 总是得到同一个 h。
        generator = make_torch_generator(self.seed + input_dim * 1009 + int(self.feature_dim or 0))
        # 注释：用上面的独立 generator 采样高斯随机矩阵，形状为 `[input_dim, feature_dim]`。
        # 除以 `sqrt(feature_dim)` 是常见随机投影缩放：让投影后每个样本的特征尺度不随投影维度线性增大，
        # 从而让不同 d_epsilon 下的协方差谱和 Xi 数值更可比。
        matrix = torch.randn(input_dim, int(self.feature_dim), generator=generator) / (float(self.feature_dim) ** 0.5)
        # 注释：把 CPU 上生成的矩阵搬到残差所在设备和 dtype。这样 CUDA 训练时不会发生 CPU/GPU 混算错误，
        # 同时保留“矩阵内容由 CPU generator 确定”的可复现性。
        self._matrix = matrix.to(device=device, dtype=dtype)
        # 注释：记录这份缓存矩阵对应的 input_dim。下一次调用时会用它判断矩阵尺寸是否仍然匹配。
        self._input_dim = input_dim
        # 注释：返回新生成并缓存好的固定投影矩阵。
        return self._matrix


# 注释： 定义类 `FastSlowDivergenceSignal`，实现最新文档中的第二个 Xi 信号：快慢分歧。
class FastSlowDivergenceSignal:
    """Fast/slow covariance divergence signal.

    The signal compares a fast residual covariance memory C_f with a slow one
    C_s.  It is intended to spike when residual geometry is changing, and to
    decay after the slow memory catches up.
    """

    # 注释： 定义初始化函数；保存快慢 EMA 参数，并把协方差缓存设为空，等第一次看到残差特征后再按真实维度初始化。
    def __init__(
        self,
        # 注释： `feature_dim` 是残差投影后的维度 d_epsilon，也是 C_f/C_s 两个协方差矩阵的边长。
        feature_dim: int,
        # 注释： `gamma_fast` 是快记忆更新率 gamma_f；文档建议 gamma_f 明显大于 gamma_s，例如 10 倍。
        gamma_fast: float,
        # 注释： `gamma_slow` 是慢记忆更新率 gamma_s；它越小，慢记忆越像长期旧常态。
        gamma_slow: float,
        # 注释： `alpha` 是给慢协方差加 shrinkage 的强度；文档写成 A_s=C_s+alpha*tr(C_s)/d*I，用于保证可逆。
        alpha: float,
        # 注释： `epsilon` 是额外数值下界，避免 logdet 或求解线性方程时遇到完全奇异矩阵。
        epsilon: float = 1e-6,
    ) -> None:
        # 注释： 记录残差特征维度；如果第一次输入维度不同，会在 `_ensure_state` 中按真实维度修正。
        self.feature_dim = int(feature_dim)
        # 注释： 记录快 EMA 更新率；后续每个 batch 都用它更新 C_f。
        self.gamma_fast = float(gamma_fast)
        # 注释： 记录慢 EMA 更新率；后续每个 batch 都用它更新 C_s。
        self.gamma_slow = float(gamma_slow)
        # 注释： 记录慢协方差 shrinkage 系数；它不是学习参数，只影响 Xi_div 数值稳定性。
        self.alpha = float(alpha)
        # 注释： 记录极小正数下界；用于协方差矩阵对角线正则，避免数值不可逆。
        self.epsilon = float(epsilon)
        # 注释： 快协方差 C_f 初始为空；第一次看到数据时会初始化成单位阵，表示初始快记忆各向同性。
        self.cov_fast: Tensor | None = None
        # 注释： 慢协方差 C_s 初始为空；第一次看到数据时也初始化成单位阵，表示初始旧常态各向同性。
        self.cov_slow: Tensor | None = None

    # 注释： 定义 `_ensure_state`，确保快慢协方差矩阵已经存在且维度、设备、dtype 与当前残差特征一致。
    def _ensure_state(self, features: Tensor) -> None:
        # 注释： 读取当前残差特征维度 d；这决定 C_f/C_s 的形状为 `[d,d]`。
        dim = int(features.shape[1])
        # 注释： 如果快慢协方差都已存在且维度匹配，就不用重新初始化，避免丢失历史记忆。
        if self.cov_fast is not None and self.cov_slow is not None and self.cov_fast.shape[0] == dim:
            # 注释： 确保缓存矩阵和当前 features 在同一设备/dtype；如果已经匹配，直接返回。
            if self.cov_fast.device == features.device and self.cov_fast.dtype == features.dtype:
                return
        # 注释： 更新 feature_dim 为当前真实维度；这让日志中的 d_epsilon 与实际矩阵一致。
        self.feature_dim = dim
        # 注释： 构造单位阵 I，作为快慢协方差的初始值；文档伪代码也写成 C_f,C_s <- I。
        identity = torch.eye(dim, device=features.device, dtype=features.dtype)
        # 注释： 初始化快协方差 C_f 为 I；这表示在还没有历史时，快记忆没有偏向任何方向。
        self.cov_fast = identity.clone()
        # 注释： 初始化慢协方差 C_s 为 I；这表示初始慢常态也假设各方向等方差。
        self.cov_slow = identity.clone()

    # 注释： 定义 `_batch_eta_cov`，把当前 batch 残差特征变成工程上的白化残差协方差。
    def _batch_eta_cov(self, features: Tensor) -> Tensor:
        # 注释： 对当前 batch 做中心化，去掉整体均值偏移，使协方差主要描述残差形状而不是均值位置。
        centered = features - features.mean(dim=0, keepdim=True)
        # 注释： 用无偏近似分母估计协方差；至少为 1，避免 batch 太小时除零。
        denom = max(1, centered.shape[0] - 1)
        # 注释： 计算当前 batch 的残差协方差，形状为 `[d,d]`。
        cov = centered.transpose(0, 1) @ centered / float(denom)
        # 注释： 计算平均方差 trace(cov)/d；后续用它做 trace 归一化，让 eta 的总尺度约等于 d。
        mean_variance = torch.trace(cov).clamp_min(self.epsilon) / float(max(1, cov.shape[0]))
        # 注释： 返回归一化后的协方差；这一步对应工程上的“白化残差 eta”，避免 Xi_div 被残差总幅度完全主导。
        return cov / mean_variance

    # 注释： 定义 `update_and_signal`，输入当前 batch 残差特征，更新快慢协方差并返回 Xi_div 与日志。
    def update_and_signal(self, features: Tensor) -> tuple[Tensor, PWLog]:
        # 注释： 确保 C_f/C_s 已初始化；第一次调用会把它们设为单位阵。
        self._ensure_state(features)
        # 注释： 静态类型提示：执行到这里 cov_fast/cov_slow 一定不为空。
        assert self.cov_fast is not None and self.cov_slow is not None
        # 注释： 历史快/慢 EMA 协方差只作为“已经发生过的记忆”，因此先 detach 成常量。
        # 这样本轮 loss_plus_xi 的梯度不会反向穿过历史窗口或历史 EMA 状态，避免跨 step 图膨胀。
        historical_fast = self.cov_fast.detach()
        historical_slow = self.cov_slow.detach()
        # 注释： 当前 batch 的归一化残差协方差不能 detach；它来自当前模型参数，Xi_div 的梯度必须从这里回传给 VAE。
        batch_cov = self._batch_eta_cov(features)
        # 注释： 构造“本轮候选快记忆” C_f'=(1-gamma_f)C_f+gamma_f*C_batch；C_f 是历史常量，C_batch 保留梯度。
        fast_candidate = (1.0 - self.gamma_fast) * historical_fast + self.gamma_fast * batch_cov
        # 注释： 构造“本轮候选慢记忆” C_s'=(1-gamma_s)C_s+gamma_s*C_batch；它同样只对当前 batch 保留梯度。
        slow_candidate = (1.0 - self.gamma_slow) * historical_slow + self.gamma_slow * batch_cov
        # 注释： 真正写回内部 EMA 缓存时必须断开计算图；缓存只记录历史状态，不应持有上一轮 autograd graph。
        with torch.no_grad():
            self.cov_fast = fast_candidate.detach()
            self.cov_slow = slow_candidate.detach()
        # 注释： 读取维度 d；公式中的 `-d` 就是用它抵消快慢一致时的迹项基线。
        dim = int(fast_candidate.shape[0])
        # 注释： 构造单位阵 I，用于给慢协方差和快协方差加入数值稳定项。
        identity = torch.eye(dim, device=features.device, dtype=features.dtype)
        # 注释： 计算慢候选协方差的 trace 平均尺度；这里使用 slow_candidate，让当前 batch 可以影响 Xi_div。
        slow_scale = torch.trace(slow_candidate).clamp_min(self.epsilon) / float(dim)
        # 注释： 构造 A_s=C_s'+alpha*tr(C_s')/d*I+epsilon*I；它是公式中被求逆的慢记忆矩阵。
        slow_matrix = slow_candidate + self.alpha * slow_scale * identity + self.epsilon * identity
        # 注释： 构造 C_f' 的数值稳定版本；文档 logdet(C_f) 需要正定矩阵，所以这里加 epsilon*I。
        fast_matrix = fast_candidate + self.epsilon * identity
        # 注释： 用线性方程求解 A_s^{-1} C_f，比显式求逆更稳定；其迹就是公式第一项。
        solved = torch.linalg.solve(slow_matrix, fast_matrix)
        # 注释： 计算迹项 tr(A_s^{-1} C_f)。
        trace_term = torch.trace(solved)
        # 注释： 计算 logdet(A_s) 和 logdet(C_f) 的差；这是零均值高斯 KL 闭式里的体积比项。
        logdet_term = torch.linalg.slogdet(slow_matrix).logabsdet - torch.linalg.slogdet(fast_matrix).logabsdet
        # 注释： 按文档公式 Xi_div=tr-d+ld，并用 ReLU 截断负值，避免数值误差导致轻微负信号。
        xi = torch.relu(trace_term - float(dim) + logdet_term)
        # 注释： 计算快协方差最大特征值，用于兼容旧日志里的 lambda_max_batch 字段。
        lambda_max = torch.linalg.eigvalsh(fast_matrix).clamp_min(0.0).max()
        # 注释： 构造统一日志对象；xi_pw 字段在该模式下表示 Xi_div。
        log = PWLog(
            # 注释： `lambda_max` 记录快协方差最大特征值，帮助判断快记忆是否出现方向性尖峰。
            lambda_max=float(lambda_max.detach().item()),
            # 注释： `xi_pw` 兼容列保存 Xi_div，便于旧画图脚本直接读取。
            xi_pw=float(xi.detach().item()),
            # 注释： 第二信号没有 tau_null 阈值，所以这里写 0。
            tau_null=0.0,
            # 注释： `trace_baseline` 兼容字段记录慢协方差 trace，表示慢记忆总体能量。
            trace_baseline=float(torch.trace(slow_candidate.detach()).item()),
            # 注释： 记录残差特征维度 d_epsilon。
            feature_dim=dim,
            # 注释： 快慢 EMA 从第一步开始就可读，因此 window_ready 设为 True。
            window_ready=True,
            # 注释： 第二信号不使用 MP edge，因此 lambda_edge 写 0。
            lambda_edge=0.0,
            # 注释： 第二信号不统计超阈谱秩，因此 spike_rank 写 0。
            spike_rank=0,
            # 注释： 快慢协方差一经初始化即可使用，因此 baseline_ready 设为 True。
            baseline_ready=True,
            # 注释： 第二信号没有 baseline warmup batch 计数，因此写 0。
            baseline_warmup_batches=0,
            # 注释： 标记阈值/信号模式为 fast_slow_divergence，方便 CSV 审计。
            threshold_mode="fast_slow_divergence",
            # 注释： 保存公式迹项，便于报告中检查 Xi_div 的组成。
            div_trace=float(trace_term.detach().item()),
            # 注释： 保存 logdet 项，便于报告中检查 Xi_div 的组成。
            div_logdet=float(logdet_term.detach().item()),
            # 注释： 保存 gamma_f，便于复现实验设置。
            div_gamma_fast=float(self.gamma_fast),
            # 注释： 保存 gamma_s，便于复现实验设置。
            div_gamma_slow=float(self.gamma_slow),
            # 注释： 保存快协方差 trace，便于观察快记忆是否被噪声放大。
            div_trace_fast=float(torch.trace(fast_candidate.detach()).item()),
            # 注释： 保存慢协方差 trace，便于观察慢记忆是否追上快记忆。
            div_trace_slow=float(torch.trace(slow_candidate.detach()).item()),
        )
        # 注释： 返回可微 Xi_div；历史状态已 detach，但当前 batch 没 detach，所以 loss_plus_xi 能真实改变模型参数。
        return xi, log


# 注释： 定义类 `PrecisionWeightedResidualSignal`，用于把残差投影、白化、窗口谱读数封装到独立组件中。
class PrecisionWeightedResidualSignal:
    """Precision-weighted residual anisotropy signal.

    The document-level signal is computed from a window covariance of whitened
    residuals, not from a single batch.  The baseline mean/covariance is warmed
    up from several batches before any signal is emitted, so the first batch no
    longer defines the whole normal residual baseline.
    """

    # 注释： 定义函数 `__init__`：保存初始化参数并准备内部缓存，真正依赖数据的矩阵稍后懒加载。
    def __init__(
        # 注释： 在当前多行结构中加入 `self`；多行写法便于审计实验参数。
        self,
        # 注释： 声明字段 `feature_dim`，表示 残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它；dataclass 会把它纳入日志对象。
        feature_dim: int,
        # 注释： 声明字段 `window_samples`，表示 白化残差窗口长度 W；决定谱估计平滑程度，也决定 MP edge 中的 d_epsilon/W；dataclass 会把它纳入日志对象。
        window_samples: int,
        # 注释： 声明字段 `gamma`，表示 baseline 均值和协方差的 EMA 更新率；小 gamma 让正常残差基线缓慢适应，避免突变被立刻吸收；dataclass 会把它纳入日志对象。
        gamma: float,
        # 注释： 声明字段 `alpha`，表示 协方差 shrinkage 正则强度；防止高维小样本协方差病态或不可逆；dataclass 会把它纳入日志对象。
        alpha: float,
        # 注释： 设置 `tau_null: float`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `0.0`；这样后续步骤可以复用该状态而不是重复计算。
        tau_null: float = 0.0,
        # 注释： 设置 `epsilon: float`（协方差求逆时的数值稳定下界；避免特征值过小导致白化爆炸）为 `1e-6`；这样后续步骤可以复用该状态而不是重复计算。
        epsilon: float = 1e-6,
        # 注释： 设置 `threshold_mode: str`（Xi 阈值模式；用于切换经验 tau_null 与理论 MP edge 两种读数）为 `"empirical"`；这样后续步骤可以复用该状态而不是重复计算。
        threshold_mode: str = "empirical",
        # 注释： 设置 `spectral_readout: str`（compact 谱信号读数类型；edge 对应最大特征值逃逸，mass 对应所有超边缘谱质量，gap 对应第一谱峰相对第二谱峰/边缘的间隔）为 `"edge"`。
        spectral_readout: str = "edge",
        # 注释： 设置 `baseline_warmup_batches: int`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `1`；这样后续步骤可以复用该状态而不是重复计算。
        baseline_warmup_batches: int = 1,
    # 注释： 执行 `) -> None:`；这是当前实验流程中的必要操作。
    ) -> None:
        # 注释： 设置 `self.feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `int(feature_dim)`；这样后续步骤可以复用该状态而不是重复计算。
        self.feature_dim = int(feature_dim)
        # 注释： 设置 `self.window_samples`（白化残差窗口长度 W；决定谱估计平滑程度，也决定 MP edge 中的 d_epsilon/W）为 `max(2, int(window_samples))`；这样后续步骤可以复用该状态而不是重复计算。
        self.window_samples = max(2, int(window_samples))
        # 注释： 设置 `self.gamma`（baseline 均值和协方差的 EMA 更新率；小 gamma 让正常残差基线缓慢适应，避免突变被立刻吸收）为 `float(gamma)`；这样后续步骤可以复用该状态而不是重复计算。
        self.gamma = float(gamma)
        # 注释： 设置 `self.alpha`（协方差 shrinkage 正则强度；防止高维小样本协方差病态或不可逆）为 `float(alpha)`；这样后续步骤可以复用该状态而不是重复计算。
        self.alpha = float(alpha)
        # 注释： 设置 `self.tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `float(tau_null)`；这样后续步骤可以复用该状态而不是重复计算。
        self.tau_null = float(tau_null)
        # 注释： 设置 `self.epsilon`（协方差求逆时的数值稳定下界；避免特征值过小导致白化爆炸）为 `float(epsilon)`；这样后续步骤可以复用该状态而不是重复计算。
        self.epsilon = float(epsilon)
        # 注释： 检查条件 `threshold_mode not in {"empirical", "mp_edge"}`；该条件决定是否进入当前实验分支。
        if threshold_mode not in {"empirical", "mp_edge"}:
            # 注释： 抛出 `ValueError(f"Unknown threshold mode: {threshold_mode}")`，用于明确暴露非法参数或子进程失败。
            raise ValueError(f"Unknown threshold mode: {threshold_mode}")
        # 注释： 设置 `self.threshold_mode`（Xi 阈值模式；用于切换经验 tau_null 与理论 MP edge 两种读数）为 `threshold_mode`；这样后续步骤可以复用该状态而不是重复计算。
        self.threshold_mode = threshold_mode
        # 注释： 检查 spectral_readout 是否是文档中三个 compact 谱信号之一；非法值直接报错，避免误把旧方案当新信号跑。
        if spectral_readout not in {"edge", "mass", "gap"}:
            raise ValueError(f"Unknown spectral readout: {spectral_readout}")
        # 注释： 保存当前谱读数类型。edge/mass/gap 共享白化、窗口中心化、trace normalization 和阈值，只在 Xi 公式处不同。
        self.spectral_readout = spectral_readout
        # 注释： 设置 `self.baseline_warmup_batches`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `max(1, int(baseline_warmup_batches))`；这样后续步骤可以复用该状态而不是重复计算。
        self.baseline_warmup_batches = max(1, int(baseline_warmup_batches))
        # 注释： 设置 `self.mu: Tensor | None`（baseline 残差均值 mu_0；初始化前置空是因为必须先看到 warmup 数据才能估计）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
        self.mu: Tensor | None = None
        # 注释： 设置 `self.cov: Tensor | None`（baseline 残差协方差 Sigma_0；初始化前置空是因为协方差需要多批残差估计）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
        self.cov: Tensor | None = None
        # 注释： 设置 `self._window: deque[Tensor]`（最近 W 个白化残差样本窗口；用固定长度队列自动丢弃旧残差，使 Xi 反映近期状态）为 `deque(maxlen=self.window_samples)`；这样后续步骤可以复用该状态而不是重复计算。
        self._window: deque[Tensor] = deque(maxlen=self.window_samples)
        # 注释： 设置 `self._warmup_features: list[Tensor]`（baseline 初始化前暂存的 warmup 残差特征；累计足够 batch 后一次性估计 mu_0/Sigma_0）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
        self._warmup_features: list[Tensor] = []

    # 注释： 定义函数 `reset_window`：封装当前脚本中的专用实验逻辑，方便复用和测试。
    def reset_window(self) -> None:
        # 注释： 执行 `self._window.clear()`；这是当前实验流程中的必要操作。
        self._window.clear()

    # 注释： 定义函数 `baseline_ready`：判断当前 signal 是否已有可用 mu_0/Sigma_0；没有则不能可靠白化残差。
    def baseline_ready(self) -> bool:
        """Return whether mu_0 and Sigma_0 have finished warmup."""
        # 注释： 返回 `self.mu is not None and self.cov is not None`，把当前计算结果交给调用方继续训练、记录或判断。
        return self.mu is not None and self.cov is not None

    # 注释： 定义函数 `readout_ready`：判断 baseline 已 ready 且窗口已满；只有这样 Xi 才不是 warmup 阶段的零读数。
    def readout_ready(self) -> bool:
        """Return whether this signal can emit a nonzero window Xi readout."""
        # 注释： 返回 `self.baseline_ready() and len(self._window) >= self.window_samples`，把当前计算结果交给调用方继续训练、记录或判断。
        return self.baseline_ready() and len(self._window) >= self.window_samples

    # 注释： 定义函数 `initialize`：用累计的 warmup 特征一次性初始化正常残差均值和协方差。
    def initialize(self, features: Tensor) -> None:
        # 注释： 设置 `dim`（当前 latent 维度或特征维度；很多矩阵大小需要由它决定）为 `features.shape[1]`；这样后续步骤可以复用该状态而不是重复计算。
        dim = features.shape[1]
        # 注释： 设置 `self.feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `int(dim)`；这样后续步骤可以复用该状态而不是重复计算。
        self.feature_dim = int(dim)
        # 注释： 设置 `self.mu`（baseline 残差均值 mu_0；初始化前置空是因为必须先看到 warmup 数据才能估计）为 `features.detach().mean(dim=0)`；这样后续步骤可以复用该状态而不是重复计算。
        self.mu = features.detach().mean(dim=0)
        # 注释： 设置 `centered`（减去均值后的残差特征；协方差必须基于中心化变量估计）为 `features.detach() - self.mu`；这样后续步骤可以复用该状态而不是重复计算。
        centered = features.detach() - self.mu
        # 注释： 设置 `denom`（协方差估计分母；至少为 1 是为了 batch 太小时避免除零）为 `max(1, centered.shape[0] - 1)`；这样后续步骤可以复用该状态而不是重复计算。
        denom = max(1, centered.shape[0] - 1)
        # 注释： 设置 `self.cov`（baseline 残差协方差 Sigma_0；初始化前置空是因为协方差需要多批残差估计）为 `centered.transpose(0, 1) @ centered / float(denom)`；这样后续步骤可以复用该状态而不是重复计算。
        self.cov = centered.transpose(0, 1) @ centered / float(denom)
        # 注释： 设置 `self.cov`（baseline 残差协方差 Sigma_0；初始化前置空是因为协方差需要多批残差估计）为 `self.cov + self.epsilon * torch.eye(dim, device=features.device, dtype=features.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        self.cov = self.cov + self.epsilon * torch.eye(dim, device=features.device, dtype=features.dtype)
        # 注释： 执行 `self._warmup_features.clear()`；这是当前实验流程中的必要操作。
        self._warmup_features.clear()

    # 注释： 定义函数 `update_baseline`：用当前 batch 残差特征初始化或 EMA 更新正常残差 baseline。
    def update_baseline(self, features: Tensor) -> None:
        # 注释： 设置 `detached`（停止梯度后的残差特征；baseline 更新不应反向影响模型参数）为 `features.detach()`；这样后续步骤可以复用该状态而不是重复计算。
        detached = features.detach()
        # 注释： 检查条件 `self.mu is None or self.cov is None`；该条件决定是否进入当前实验分支。
        if self.mu is None or self.cov is None:
            # Warm up the normal residual baseline with multiple batches.
            # This avoids using a possibly unrepresentative first batch as
            # the sole estimate of mu_0 and Sigma_0.
            # 注释： 执行 `self._warmup_features.append(detached)`；这是当前实验流程中的必要操作。
            self._warmup_features.append(detached)
            # 注释： 检查条件 `len(self._warmup_features) >= self.baseline_warmup_batches`；该条件决定是否进入当前实验分支。
            if len(self._warmup_features) >= self.baseline_warmup_batches:
                # 注释： 设置 `self.initialize(torch.cat(self._warmup_features, dim`（_warmup_features, dim）为 `0))`；这样后续步骤可以复用该状态而不是重复计算。
                self.initialize(torch.cat(self._warmup_features, dim=0))
            # 注释： 当前分支已经完成必要处理，直接结束函数避免继续执行。
            return
        # 注释： 先计算当前 batch 的残差均值。它只用于慢速更新 baseline 中心 mu_0，
        # 不能直接拿来给当前 batch 去中心化；否则每个 batch 自己的整体偏移会被消掉，
        # 与文档中“围绕慢速 baseline 均值 mu_0 估计正常残差形状”的定义不一致。
        batch_mu = detached.mean(dim=0)
        # 注释： 按文档公式先得到更新后的慢速 baseline 均值：
        # mu_0 <- (1-gamma) * mu_0 + gamma * mean(epsilon_t)。
        # 这里的 new_mu 仍然是很慢变化的“正常残差中心”，不是当前 batch 自己的中心。
        new_mu = (1.0 - self.gamma) * self.mu + self.gamma * batch_mu
        # 注释： 协方差必须围绕慢速 baseline 均值 new_mu 计算，而不是围绕 batch_mu 计算。
        # 这样如果新阶段残差整体沿某个方向偏移，(epsilon_t - mu_0) 会保留这个方向性，
        # 后续白化残差和 Xi 才能看到“相对旧正常中心的新方向/新偏移”。
        centered = detached - new_mu
        # 注释： 设置 `denom`（协方差估计分母；至少为 1 是为了 batch 太小时避免除零）为 `max(1, centered.shape[0] - 1)`；这样后续步骤可以复用该状态而不是重复计算。
        denom = max(1, centered.shape[0] - 1)
        # 注释： 设置 `batch_cov`（当前 batch 的残差协方差；用于 EMA 更新正常残差形状）为 `centered.transpose(0, 1) @ centered / float(denom)`；这样后续步骤可以复用该状态而不是重复计算。
        batch_cov = centered.transpose(0, 1) @ centered / float(denom)
        # 注释： 设置 `gamma`（baseline 均值和协方差的 EMA 更新率；小 gamma 让正常残差基线缓慢适应，避免突变被立刻吸收）为 `self.gamma`；这样后续步骤可以复用该状态而不是重复计算。
        gamma = self.gamma
        # 注释： 保存上面算好的慢速均值更新结果。注意协方差也已经使用同一个 new_mu 作为中心，
        # 因此 mu_0 和 Sigma_0 对应同一轮 baseline 状态。
        self.mu = new_mu
        # 注释： 设置 `self.cov`（baseline 残差协方差 Sigma_0；初始化前置空是因为协方差需要多批残差估计）为 `(1.0 - gamma) * self.cov + gamma * batch_cov`；这样后续步骤可以复用该状态而不是重复计算。
        self.cov = (1.0 - gamma) * self.cov + gamma * batch_cov

    # 注释： 定义函数 `batch_signal`：保留旧版单 batch Xi 读数，方便兼容早期实验。
    def batch_signal(self, features: Tensor) -> tuple[Tensor, PWLog]:
        """Deprecated single-batch readout kept for compatibility.

        New experiments should call update_window(...) followed by
        window_signal(...), because the document defines C_hat_t over the
        recent residual window.
        """
        # 注释： 检查条件 `self.mu is None or self.cov is None`；该条件决定是否进入当前实验分支。
        if self.mu is None or self.cov is None:
            # 注释： 返回 `self._zero_signal(features)`，把当前计算结果交给调用方继续训练、记录或判断。
            return self._zero_signal(features)
        # 注释： 执行 `assert self.mu is not None and self.cov is not None`；这是当前实验流程中的必要操作。
        assert self.mu is not None and self.cov is not None
        # 注释： 设置 `invsqrt`（baseline 协方差逆平方根 Sigma_0^{-1/2}；用于把正常残差白化到近似各向同性）为 `self._inverse_sqrt_cov(features.device, features.dtype).detach()`；这样后续步骤可以复用该状态而不是重复计算。
        invsqrt = self._inverse_sqrt_cov(features.device, features.dtype).detach()
        # 注释： 设置 `eta`（白化后的残差特征 eta；Xi 实际分析的是 eta 的窗口协方差谱）为 `(features - self.mu.detach()) @ invsqrt.transpose(0, 1)`；这样后续步骤可以复用该状态而不是重复计算。
        eta = (features - self.mu.detach()) @ invsqrt.transpose(0, 1)
        # Spectral-edge scheme A uses the shape spectrum: center the current
        # whitened samples inside the readout window, compute an unbiased
        # covariance, and normalize its trace to feature_dim. This removes total
        # residual energy and keeps only how energy is distributed over
        # directions.
        eigenvalues = self._shape_eigenvalues(eta)
        # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `eigenvalues.max()`；这样后续步骤可以复用该状态而不是重复计算。
        lambda_max = eigenvalues.max()
        # 注释： 设置 `threshold`（当前 Xi 使用的阈值；扣除随机有限样本造成的正常谱上界）为 `self.threshold(features.device, features.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        threshold = self.threshold(features.device, features.dtype)
        # 注释： 按 compact 文档选择 edge/mass/gap 三种读数之一。三者都使用同一套形状谱，只改变“超出谱边缘”的汇总方式。
        xi = self._spectral_readout(eigenvalues, threshold)
        # 注释： 设置 `log`（当前 Xi 计算的日志结构；把谱读数、阈值、ready 状态写入 CSV 方便审计）为 `PWLog(`；这样后续步骤可以复用该状态而不是重复计算。
        log = PWLog(
            # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `float(lambda_max.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_max=float(lambda_max.detach().item()),
            # 注释： 设置 `xi_pw`（精度加权残差各向异性信号 Xi；用于衡量残差是否集中到少数方向）为 `float(xi.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            xi_pw=float(xi.detach().item()),
            # 注释： 设置 `tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `float(threshold.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            tau_null=float(threshold.detach().item()),
            # 注释： 设置 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移）为 `float(torch.trace(self.cov.detach()).item())`；这样后续步骤可以复用该状态而不是重复计算。
            trace_baseline=float(torch.trace(self.cov.detach()).item()),
            # 注释： 设置 `feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `int(features.shape[1])`；这样后续步骤可以复用该状态而不是重复计算。
            feature_dim=int(features.shape[1]),
            # 注释： 设置 `window_ready`（残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi）为 `len(self._window) >= self.window_samples`；这样后续步骤可以复用该状态而不是重复计算。
            window_ready=len(self._window) >= self.window_samples,
            # 注释： 设置 `lambda_edge`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向）为 `self.mp_edge()`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_edge=self.mp_edge(),
            # 注释： 设置 `spike_rank`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度）为 `int((eigenvalues > threshold).sum().detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            spike_rank=int((eigenvalues > threshold).sum().detach().item()),
            # 注释： 设置 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_ready=True,
            # 注释： 设置 `baseline_warmup_batches`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `len(self._warmup_features)`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_warmup_batches=len(self._warmup_features),
            # 注释： threshold_mode 同时写入阈值来源和 compact 读数类型，方便 CSV 中区分 edge/mass/gap 三个新信号。
            threshold_mode=f"{self.threshold_mode}:{self.spectral_readout}",
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
        # 注释： 返回 `xi, log`，把当前计算结果交给调用方继续训练、记录或判断。
        return xi, log

    # 注释： 定义函数 `update_window`：把当前 batch 白化残差加入最近窗口，使 Xi 反映近期残差几何。
    def update_window(self, features: Tensor) -> None:
        # 注释： 检查条件 `self.mu is None or self.cov is None`；该条件决定是否进入当前实验分支。
        if self.mu is None or self.cov is None:
            # 注释： 当前分支已经完成必要处理，直接结束函数避免继续执行。
            return
        # 注释： 设置 `invsqrt`（baseline 协方差逆平方根 Sigma_0^{-1/2}；用于把正常残差白化到近似各向同性）为 `self._inverse_sqrt_cov(features.device, features.dtype).detach()`；这样后续步骤可以复用该状态而不是重复计算。
        invsqrt = self._inverse_sqrt_cov(features.device, features.dtype).detach()
        # 注释： 设置 `eta`（白化后的残差特征 eta；Xi 实际分析的是 eta 的窗口协方差谱）为 `(features.detach() - self.mu.detach()) @ invsqrt.transpose(0, 1)`；这样后续步骤可以复用该状态而不是重复计算。
        eta = (features.detach() - self.mu.detach()) @ invsqrt.transpose(0, 1)
        # 注释： 遍历 `row in eta.cpu()`，用于逐个处理 batch、阶段或 sweep 组合。
        for row in eta.cpu():
            # 注释： 执行 `self._window.append(row)`；这是当前实验流程中的必要操作。
            self._window.append(row)

    # 注释： 定义函数 `window_signal`：从最近白化残差窗口计算不可微 Xi、MP 超阈方向数和日志。
    def window_signal(self, features: Tensor) -> tuple[Tensor, PWLog]:
        """Compute Xi from the recent whitened-residual window C_hat_t."""
        # 注释： 检查条件 `self.mu is None or self.cov is None or len(self._window) < self.window_samples`；该条件决定是否进入当前实验分支。
        if self.mu is None or self.cov is None or len(self._window) < self.window_samples:
            # 注释： 返回 `self._zero_signal(features)`，把当前计算结果交给调用方继续训练、记录或判断。
            return self._zero_signal(features)
        # 注释： 设置 `stacked`（窗口内白化残差拼接矩阵；用它估计最近窗口协方差 C_hat_t）为 `torch.stack(list(self._window), dim=0).to(device=features.device, dtype=features.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        stacked = torch.stack(list(self._window), dim=0).to(device=features.device, dtype=features.dtype)
        # Spectral-edge scheme A uses the shape spectrum: center the current
        # whitened window and trace-normalize its covariance before eigenvalues.
        eigenvalues = self._shape_eigenvalues(stacked)
        # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `eigenvalues.max()`；这样后续步骤可以复用该状态而不是重复计算。
        lambda_max = eigenvalues.max()
        # 注释： 设置 `threshold`（当前 Xi 使用的阈值；扣除随机有限样本造成的正常谱上界）为 `self.threshold(features.device, features.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        threshold = self.threshold(features.device, features.dtype)
        # 注释： 按 compact 文档选择 edge/mass/gap 三种读数之一。三者都使用同一套形状谱，只改变“超出谱边缘”的汇总方式。
        xi = self._spectral_readout(eigenvalues, threshold)
        # 注释： 设置 `log`（当前 Xi 计算的日志结构；把谱读数、阈值、ready 状态写入 CSV 方便审计）为 `PWLog(`；这样后续步骤可以复用该状态而不是重复计算。
        log = PWLog(
            # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `float(lambda_max.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_max=float(lambda_max.detach().item()),
            # 注释： 设置 `xi_pw`（精度加权残差各向异性信号 Xi；用于衡量残差是否集中到少数方向）为 `float(xi.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            xi_pw=float(xi.detach().item()),
            # 注释： 设置 `tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `float(threshold.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            tau_null=float(threshold.detach().item()),
            # 注释： 设置 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移）为 `float(torch.trace(self.cov.detach()).item())`；这样后续步骤可以复用该状态而不是重复计算。
            trace_baseline=float(torch.trace(self.cov.detach()).item()),
            # 注释： 设置 `feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `int(self.feature_dim)`；这样后续步骤可以复用该状态而不是重复计算。
            feature_dim=int(self.feature_dim),
            # 注释： 设置 `window_ready`（残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
            window_ready=True,
            # 注释： 设置 `lambda_edge`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向）为 `self.mp_edge()`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_edge=self.mp_edge(),
            # 注释： 设置 `spike_rank`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度）为 `int((eigenvalues > threshold).sum().detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            spike_rank=int((eigenvalues > threshold).sum().detach().item()),
            # 注释： 设置 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_ready=True,
            # 注释： 设置 `baseline_warmup_batches`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `len(self._warmup_features)`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_warmup_batches=len(self._warmup_features),
            # 注释： threshold_mode 同时写入阈值来源和 compact 读数类型，方便 CSV 中区分 edge/mass/gap 三个新信号。
            threshold_mode=f"{self.threshold_mode}:{self.spectral_readout}",
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
        # 注释： 返回 `xi, log`，把当前计算结果交给调用方继续训练、记录或判断。
        return xi, log

    # 注释： 定义函数 `differentiable_window_signal`：用历史窗口加当前 batch 计算可反传 Xi，使 loss_plus_xi 真正影响模型。
    def differentiable_window_signal(self, features: Tensor) -> tuple[Tensor, PWLog]:
        """Compute Xi with gradients through the current batch residuals.

        The baseline and historical residual window are constants. The current
        batch is whitened without detaching `features`, so `loss_plus_xi` can
        backpropagate through Xi into the reconstruction model.
        """
        # 注释： 检查条件 `self.mu is None or self.cov is None`；该条件决定是否进入当前实验分支。
        if self.mu is None or self.cov is None:
            # 注释： 返回 `self._zero_signal(features)`，把当前计算结果交给调用方继续训练、记录或判断。
            return self._zero_signal(features)
        # 注释： 设置 `invsqrt`（baseline 协方差逆平方根 Sigma_0^{-1/2}；用于把正常残差白化到近似各向同性）为 `self._inverse_sqrt_cov(features.device, features.dtype).detach()`；这样后续步骤可以复用该状态而不是重复计算。
        invsqrt = self._inverse_sqrt_cov(features.device, features.dtype).detach()
        # 注释： 设置 `eta_current`（eta_current）为 `(features - self.mu.detach()) @ invsqrt.transpose(0, 1)`；这样后续步骤可以复用该状态而不是重复计算。
        eta_current = (features - self.mu.detach()) @ invsqrt.transpose(0, 1)

        # 注释： 设置 `pieces`（pieces）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
        pieces = []
        # 注释： 检查条件 `len(self._window) > 0`；该条件决定是否进入当前实验分支。
        if len(self._window) > 0:
            # 注释： 设置 `history`（train_history.csv 读出的训练日志；最终 summary 从最后一行提取指标）为 `torch.stack(list(self._window), dim=0).to(device=features.device, dtype=features.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
            history = torch.stack(list(self._window), dim=0).to(device=features.device, dtype=features.dtype)
            # 注释： 执行 `pieces.append(history.detach())`；这是当前实验流程中的必要操作。
            pieces.append(history.detach())
        # 注释： 执行 `pieces.append(eta_current)`；这是当前实验流程中的必要操作。
        pieces.append(eta_current)
        # 注释： 设置 `stacked`（窗口内白化残差拼接矩阵；用它估计最近窗口协方差 C_hat_t）为 `torch.cat(pieces, dim=0)`；这样后续步骤可以复用该状态而不是重复计算。
        stacked = torch.cat(pieces, dim=0)

        # 注释： 检查条件 `stacked.shape[0] > self.window_samples`；该条件决定是否进入当前实验分支。
        if stacked.shape[0] > self.window_samples:
            # 注释： 设置 `stacked`（窗口内白化残差拼接矩阵；用它估计最近窗口协方差 C_hat_t）为 `stacked[-self.window_samples :]`；这样后续步骤可以复用该状态而不是重复计算。
            stacked = stacked[-self.window_samples :]
        # 注释： 检查条件 `stacked.shape[0] < self.window_samples`；该条件决定是否进入当前实验分支。
        if stacked.shape[0] < self.window_samples:
            # 注释： 返回 `self._zero_signal(features)`，把当前计算结果交给调用方继续训练、记录或判断。
            return self._zero_signal(features)

        # Spectral-edge scheme A uses the shape spectrum: center the current
        # whitened window and trace-normalize its covariance before eigenvalues.
        eigenvalues = self._shape_eigenvalues(stacked)
        # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `eigenvalues.max()`；这样后续步骤可以复用该状态而不是重复计算。
        lambda_max = eigenvalues.max()
        # 注释： 设置 `threshold`（当前 Xi 使用的阈值；扣除随机有限样本造成的正常谱上界）为 `self.threshold(features.device, features.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        threshold = self.threshold(features.device, features.dtype)
        # 注释： 按 compact 文档选择 edge/mass/gap 三种读数之一。三者都使用同一套形状谱，只改变“超出谱边缘”的汇总方式。
        xi = self._spectral_readout(eigenvalues, threshold)
        # 注释： 设置 `log`（当前 Xi 计算的日志结构；把谱读数、阈值、ready 状态写入 CSV 方便审计）为 `PWLog(`；这样后续步骤可以复用该状态而不是重复计算。
        log = PWLog(
            # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `float(lambda_max.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_max=float(lambda_max.detach().item()),
            # 注释： 设置 `xi_pw`（精度加权残差各向异性信号 Xi；用于衡量残差是否集中到少数方向）为 `float(xi.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            xi_pw=float(xi.detach().item()),
            # 注释： 设置 `tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `float(threshold.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
            tau_null=float(threshold.detach().item()),
            # 注释： 设置 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移）为 `float(torch.trace(self.cov.detach()).item())`；这样后续步骤可以复用该状态而不是重复计算。
            trace_baseline=float(torch.trace(self.cov.detach()).item()),
            # 注释： 设置 `feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `int(self.feature_dim)`；这样后续步骤可以复用该状态而不是重复计算。
            feature_dim=int(self.feature_dim),
            # 注释： 设置 `window_ready`（残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
            window_ready=True,
            # 注释： 设置 `lambda_edge`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向）为 `self.mp_edge()`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_edge=self.mp_edge(),
            # 注释： 设置 `spike_rank`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度）为 `int((eigenvalues.detach() > threshold.detach()).sum().item())`；这样后续步骤可以复用该状态而不是重复计算。
            spike_rank=int((eigenvalues.detach() > threshold.detach()).sum().item()),
            # 注释： 设置 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_ready=True,
            # 注释： 设置 `baseline_warmup_batches`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `len(self._warmup_features)`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_warmup_batches=len(self._warmup_features),
            # 注释： threshold_mode 同时写入阈值来源和 compact 读数类型，方便 CSV 中区分 edge/mass/gap 三个新信号。
            threshold_mode=f"{self.threshold_mode}:{self.spectral_readout}",
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
        # 注释： 返回 `xi, log`，把当前计算结果交给调用方继续训练、记录或判断。
        return xi, log

    # 注释： 定义函数 `window_lambda_max`：只读取窗口最大特征值，用于校准和画 lambda_max 曲线。
    def window_lambda_max(self, device: torch.device | None = None) -> float:
        # 注释： 检查条件 `len(self._window) < 2`；该条件决定是否进入当前实验分支。
        if len(self._window) < 2:
            # 注释： 返回 `0.0`，把当前计算结果交给调用方继续训练、记录或判断。
            return 0.0
        # 注释： 设置 `stacked`（窗口内白化残差拼接矩阵；用它估计最近窗口协方差 C_hat_t）为 `torch.stack(list(self._window), dim=0)`；这样后续步骤可以复用该状态而不是重复计算。
        stacked = torch.stack(list(self._window), dim=0)
        # 注释： 检查条件 `device is not None`；该条件决定是否进入当前实验分支。
        if device is not None:
            # 注释： 设置 `stacked`（窗口内白化残差拼接矩阵；用它估计最近窗口协方差 C_hat_t）为 `stacked.to(device)`；这样后续步骤可以复用该状态而不是重复计算。
            stacked = stacked.to(device)
        eigenvalues = self._shape_eigenvalues(stacked)
        return float(eigenvalues.max().item())

    # 注释： 定义函数 `mp_edge`：根据 d_epsilon/W 计算 Marchenko-Pastur 随机谱右边缘。
    def mp_edge(self) -> float:
        # The new spectral-edge document uses n = W - 1 after window centering.
        c = float(self.feature_dim) / float(max(1, self.window_samples - 1))
        # 注释： 返回 `float((1.0 + math.sqrt(c)) ** 2)`，把当前计算结果交给调用方继续训练、记录或判断。
        return float((1.0 + math.sqrt(c)) ** 2)

    def _shape_eigenvalues(self, whitened_window: Tensor) -> Tensor:
        """Return eigenvalues of centered, trace-normalized whitened covariance.

        This is the shared spectrum required by the compact spectral residual
        signal document:

            centered = eta - mean_window(eta)
            C_hat = centered.T @ centered / (W - 1)
            C_tilde = d * C_hat / trace(C_hat)

        The returned eigenvalues have mean approximately one whenever the trace
        is nonzero, so Xi responds to spectral shape rather than total energy.
        """
        if whitened_window.ndim != 2:
            raise ValueError("whitened_window must have shape [W, d]")
        centered = whitened_window - whitened_window.mean(dim=0, keepdim=True)
        denom = float(max(1, centered.shape[0] - 1))
        cov = centered.transpose(0, 1) @ centered / denom
        trace = torch.trace(cov).clamp_min(self.epsilon)
        cov = cov * (float(cov.shape[0]) / trace)
        return torch.linalg.eigvalsh(cov).clamp_min(0.0)

    # 注释： 定义函数 `_spectral_readout`：把同一组形状谱特征值转换为 compact 文档中的三种 Xi 信号。
    def _spectral_readout(self, eigenvalues: Tensor, threshold: Tensor) -> Tensor:
        """Return edge, mass, or gap signal from trace-normalized eigenvalues."""
        # 注释： threshold 是经验谱边缘或 MP 谱边缘；clamp 防止极端校准失败时除零。
        edge = threshold.clamp_min(self.epsilon)
        # 注释： lambda_1 是最大特征值，代表当前窗口最强方向性残差。
        lambda_1 = eigenvalues[-1]
        # 注释： edge 信号对应文档 Scheme A，只关心最大谱峰相对边缘超出多少。
        if self.spectral_readout == "edge":
            return torch.relu(lambda_1 / edge - 1.0)
        # 注释： mass 信号对应文档 Scheme B，对所有超过谱边缘的方向求和，反映异常方向的总谱质量。
        if self.spectral_readout == "mass":
            return torch.relu(eigenvalues / edge - 1.0).sum()
        # 注释： gap 信号对应文档 Scheme C，要求第一谱峰不仅超过边缘，也要明显高于第二谱峰，突出“单一主导方向”。
        lambda_2_or_edge = edge
        if eigenvalues.numel() >= 2:
            lambda_2_or_edge = torch.maximum(eigenvalues[-2], edge)
        return torch.relu((lambda_1 - lambda_2_or_edge) / edge)

    # 注释： 定义函数 `threshold`：根据 threshold_mode 返回 empirical tau_null 或 MP edge。
    def threshold(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        # 注释： 设置 `value`（value）为 `self.mp_edge() if self.threshold_mode == "mp_edge" else self.tau_null`；这样后续步骤可以复用该状态而不是重复计算。
        value = self.mp_edge() if self.threshold_mode == "mp_edge" else self.tau_null
        # 注释： 返回 `torch.tensor(value, device=device, dtype=dtype)`，把当前计算结果交给调用方继续训练、记录或判断。
        return torch.tensor(value, device=device, dtype=dtype)

    # 注释： 定义函数 `_zero_signal`：baseline 或窗口未 ready 时返回 Xi=0，同时记录为何还不能读数。
    def _zero_signal(self, features: Tensor) -> tuple[Tensor, PWLog]:
        # 注释： 设置 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照）为 `self.mu is not None and self.cov is not None`；这样后续步骤可以复用该状态而不是重复计算。
        baseline_ready = self.mu is not None and self.cov is not None
        # 注释： 设置 `threshold`（当前 Xi 使用的阈值；扣除随机有限样本造成的正常谱上界）为 `self.mp_edge() if self.threshold_mode == "mp_edge" else self.tau_null`；这样后续步骤可以复用该状态而不是重复计算。
        threshold = self.mp_edge() if self.threshold_mode == "mp_edge" else self.tau_null
        # 注释： 设置 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移）为 `float(torch.trace(self.cov.detach()).item()) if self.cov is not None else 0.0`；这样后续步骤可以复用该状态而不是重复计算。
        trace_baseline = float(torch.trace(self.cov.detach()).item()) if self.cov is not None else 0.0
        # 注释： 返回 `features.new_tensor(0.0), PWLog(`，把当前计算结果交给调用方继续训练、记录或判断。
        return features.new_tensor(0.0), PWLog(
            # 注释： 设置 `lambda_max`（白化残差窗口协方差矩阵的最大特征值；它是 Xi 判断“最强异常方向”的直接谱读数）为 `0.0`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_max=0.0,
            # 注释： 设置 `xi_pw`（精度加权残差各向异性信号 Xi；用于衡量残差是否集中到少数方向）为 `0.0`；这样后续步骤可以复用该状态而不是重复计算。
            xi_pw=0.0,
            # 注释： 设置 `tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `threshold`；这样后续步骤可以复用该状态而不是重复计算。
            tau_null=threshold,
            # 注释： 设置 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移）为 `trace_baseline`；这样后续步骤可以复用该状态而不是重复计算。
            trace_baseline=trace_baseline,
            # 注释： 设置 `feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `int(features.shape[1])`；这样后续步骤可以复用该状态而不是重复计算。
            feature_dim=int(features.shape[1]),
            # 注释： 设置 `window_ready`（残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi）为 `len(self._window) >= self.window_samples`；这样后续步骤可以复用该状态而不是重复计算。
            window_ready=len(self._window) >= self.window_samples,
            # 注释： 设置 `lambda_edge`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向）为 `self.mp_edge()`；这样后续步骤可以复用该状态而不是重复计算。
            lambda_edge=self.mp_edge(),
            # 注释： 设置 `spike_rank`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
            spike_rank=0,
            # 注释： 设置 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照）为 `baseline_ready`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_ready=baseline_ready,
            # 注释： 设置 `baseline_warmup_batches`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `len(self._warmup_features)`；这样后续步骤可以复用该状态而不是重复计算。
            baseline_warmup_batches=len(self._warmup_features),
            # 注释： threshold_mode 同时写入阈值来源和 compact 读数类型；即使还没 ready，也能知道当前配置。
            threshold_mode=f"{self.threshold_mode}:{self.spectral_readout}",
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )

    # 注释： 定义函数 `_regularized_cov`：对 baseline 协方差做 shrinkage，解决高维小样本协方差病态问题。
    def _regularized_cov(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        # 注释： 执行 `assert self.cov is not None`；这是当前实验流程中的必要操作。
        assert self.cov is not None
        # 注释： 设置 `cov`（baseline 残差协方差 Sigma_0；初始化前置空是因为协方差需要多批残差估计）为 `self.cov.to(device=device, dtype=dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        cov = self.cov.to(device=device, dtype=dtype)
        # 注释： 设置 `dim`（当前 latent 维度或特征维度；很多矩阵大小需要由它决定）为 `cov.shape[0]`；这样后续步骤可以复用该状态而不是重复计算。
        dim = cov.shape[0]
        # 注释： 设置 `avg_scale`（avg_scale）为 `torch.trace(cov).clamp_min(self.epsilon) / float(dim)`；这样后续步骤可以复用该状态而不是重复计算。
        avg_scale = torch.trace(cov).clamp_min(self.epsilon) / float(dim)
        # 注释： 设置 `eye`（eye）为 `torch.eye(dim, device=device, dtype=dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        eye = torch.eye(dim, device=device, dtype=dtype)
        # 注释： 返回 `cov + self.alpha * avg_scale * eye + self.epsilon * eye`，把当前计算结果交给调用方继续训练、记录或判断。
        return cov + self.alpha * avg_scale * eye + self.epsilon * eye

    # 注释： 定义函数 `_inverse_sqrt_cov`：计算 Sigma_0^{-1/2}，把残差变成精度加权/白化残差。
    def _inverse_sqrt_cov(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        # 注释： 设置 `cov_reg`（正则化后的 baseline 协方差；用于稳定求逆平方根）为 `self._regularized_cov(device, dtype)`；这样后续步骤可以复用该状态而不是重复计算。
        cov_reg = self._regularized_cov(device, dtype)
        # 注释： 设置 `eigenvalues, eigenvectors`（eigenvalues, eigenvectors）为 `torch.linalg.eigh(cov_reg)`；这样后续步骤可以复用该状态而不是重复计算。
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)
        # 注释： 设置 `invsqrt_values`（协方差特征值的逆平方根；小特征值会被 clamp 避免无限放大）为 `eigenvalues.clamp_min(self.epsilon).rsqrt()`；这样后续步骤可以复用该状态而不是重复计算。
        invsqrt_values = eigenvalues.clamp_min(self.epsilon).rsqrt()
        # 注释： 返回 `(eigenvectors * invsqrt_values.unsqueeze(0)) @ eigenvectors.transpose(0, 1)`，把当前计算结果交给调用方继续训练、记录或判断。
        return (eigenvectors * invsqrt_values.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
