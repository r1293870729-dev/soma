# 注释： 启用未来版本注解语义，避免类型注解在旧 Python 行为下提前求值。
from __future__ import annotations

# 注释： 导入模块 `argparse`，因为后续代码会使用其中的函数或类。
import argparse
# 注释： 导入 `deque`，用于保存最近若干个训练 batch；可吸收性信号需要在同一批近期样本上比较快适应前后的残差能量。
from collections import deque
# 注释： 导入模块 `csv`，因为后续代码会使用其中的函数或类。
import csv
# 注释： 导入模块 `math`，因为后续代码会使用其中的函数或类。
import math
# 注释： 导入模块 `time`，因为后续代码会使用其中的函数或类。
import time
# 注释： 导入依赖 `from dataclasses import dataclass`，避免在后续代码中使用冗长模块前缀。
from dataclasses import dataclass
# 注释： 导入依赖 `from pathlib import Path`，避免在后续代码中使用冗长模块前缀。
from pathlib import Path
# 注释： 导入依赖 `from typing import Iterator`，避免在后续代码中使用冗长模块前缀。
from typing import Iterator

# 注释： 导入模块 `matplotlib.pyplot as plt`，因为后续代码会使用其中的函数或类。
import matplotlib.pyplot as plt
# 注释： 导入模块 `numpy as np`，因为后续代码会使用其中的函数或类。
import numpy as np
# 注释： 导入模块 `torch`，因为后续代码会使用其中的函数或类。
import torch
# 注释： 导入依赖 `from ignite.engine import Engine, Events`，避免在后续代码中使用冗长模块前缀。
from ignite.engine import Engine, Events
# 注释： 导入依赖 `from ignite.handlers import EarlyStopping`，避免在后续代码中使用冗长模块前缀。
from ignite.handlers import EarlyStopping
# 注释： 导入依赖 `from torch import Tensor, nn`，避免在后续代码中使用冗长模块前缀。
from torch import Tensor, nn
# 注释： 导入依赖 `from torch.optim import AdamW`，避免在后续代码中使用冗长模块前缀。
from torch.optim import AdamW
# 注释： 导入依赖 `from torch.utils.data import DataLoader, Dataset, TensorDataset`，避免在后续代码中使用冗长模块前缀。
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 注释： 导入依赖 `from datasets import MixedCifarCDataset, build_cifar10_dataset, cycle_loader`，避免在后续代码中使用冗长模块前缀。
from datasets import MixedCifarCDataset, build_cifar10_dataset, cycle_loader
# 注释： 导入依赖 `from losses_pw import PrecisionWeightedResidualSignal, ResidualFeatureProjector`，避免在后续代码中使用冗长模块前缀。
from losses_pw import FastSlowDivergenceSignal, PWLog, PrecisionWeightedResidualSignal, ResidualFeatureProjector
# 注释： 导入依赖 `from models import ConvVAE, StrongConvVAE, kl_standard_normal, reconstruction_loss`，避免在后续代码中使用冗长模块前缀。
from models import ConvVAE, StrongConvVAE, kl_standard_normal, reconstruction_loss
# 注释： 导入依赖 `from utils import append_csv, resolve_device, set_seed, write_json`，避免在后续代码中使用冗长模块前缀。
from utils import append_csv, make_torch_generator, resolve_device, set_seed, write_json


# 注释： 设置 `ROOT`（当前脚本所在的 exp1_new 根目录；所有相对路径都基于它构造，方便迁移项目）为 `Path(__file__).resolve().parent`；这样后续步骤可以复用该状态而不是重复计算。
ROOT = Path(__file__).resolve().parent
# 注释： 设置 `INVALID_LABEL`（INVALID_LABEL）为 `-1`；这样后续步骤可以复用该状态而不是重复计算。
INVALID_LABEL = -1


# 注释： 把下面的日志结构声明为 dataclass，因为它只负责承载 Xi 计算结果，不需要手写初始化函数。
@dataclass(frozen=True)
# 注释： 定义类 `Stage`，用于把残差投影、白化、窗口谱读数封装到独立组件中。
class Stage:
    # 注释： 声明字段 `name`，表示 name；dataclass 会把它纳入日志对象。
    name: str
    # 注释： 声明字段 `steps`，表示 steps；dataclass 会把它纳入日志对象。
    steps: int
    # 注释： 声明字段 `loader`，表示 loader；dataclass 会把它纳入日志对象。
    loader: DataLoader


# 注释： 定义类 `MLPVAE`，用于把残差投影、白化、窗口谱读数封装到独立组件中。
class MLPVAE(nn.Module):
    # 注释： 定义函数 `__init__`：保存初始化参数并准备内部缓存，真正依赖数据的矩阵稍后懒加载。
    def __init__(self, observation_dim: int, latent_dim: int, hidden_dim: int, hidden_layers: int = 2) -> None:
        # 注释： 执行 `super().__init__()`；这是当前实验流程中的必要操作。
        super().__init__()
        # 注释： 设置 `encoder_layers: list[nn.Module]`（encoder_layers）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
        encoder_layers: list[nn.Module] = []
        # 注释： 设置 `input_dim`（展平残差的原始维度；用于判断是否需要投影以及生成匹配尺寸的投影矩阵）为 `observation_dim`；这样后续步骤可以复用该状态而不是重复计算。
        input_dim = observation_dim
        # 注释： 遍历 `_ in range(max(1, int(hidden_layers)))`，用于逐个处理 batch、阶段或 sweep 组合。
        for _ in range(max(1, int(hidden_layers))):
            # 注释： 执行 `encoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])`；这是当前实验流程中的必要操作。
            encoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])
            # 注释： 设置 `input_dim`（展平残差的原始维度；用于判断是否需要投影以及生成匹配尺寸的投影矩阵）为 `hidden_dim`；这样后续步骤可以复用该状态而不是重复计算。
            input_dim = hidden_dim
        # 注释： 设置 `self.encoder`（encoder）为 `nn.Sequential(*encoder_layers)`；这样后续步骤可以复用该状态而不是重复计算。
        self.encoder = nn.Sequential(*encoder_layers)
        # 注释： 设置 `self.mean`（mean）为 `nn.Linear(hidden_dim, latent_dim)`；这样后续步骤可以复用该状态而不是重复计算。
        self.mean = nn.Linear(hidden_dim, latent_dim)
        # 注释： 设置 `self.logvar`（logvar）为 `nn.Linear(hidden_dim, latent_dim)`；这样后续步骤可以复用该状态而不是重复计算。
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        # 注释： 设置 `decoder_layers: list[nn.Module]`（decoder_layers）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
        decoder_layers: list[nn.Module] = []
        # 注释： 设置 `input_dim`（展平残差的原始维度；用于判断是否需要投影以及生成匹配尺寸的投影矩阵）为 `latent_dim`；这样后续步骤可以复用该状态而不是重复计算。
        input_dim = latent_dim
        # 注释： 遍历 `_ in range(max(1, int(hidden_layers)))`，用于逐个处理 batch、阶段或 sweep 组合。
        for _ in range(max(1, int(hidden_layers))):
            # 注释： 执行 `decoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])`；这是当前实验流程中的必要操作。
            decoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])
            # 注释： 设置 `input_dim`（展平残差的原始维度；用于判断是否需要投影以及生成匹配尺寸的投影矩阵）为 `hidden_dim`；这样后续步骤可以复用该状态而不是重复计算。
            input_dim = hidden_dim
        # 注释： 执行 `decoder_layers.append(nn.Linear(hidden_dim, observation_dim))`；这是当前实验流程中的必要操作。
        decoder_layers.append(nn.Linear(hidden_dim, observation_dim))
        # 注释： 设置 `self.decoder`（decoder）为 `nn.Sequential(*decoder_layers)`；这样后续步骤可以复用该状态而不是重复计算。
        self.decoder = nn.Sequential(*decoder_layers)

    # 注释： 定义函数 `encode`：封装当前脚本中的专用实验逻辑，方便复用和测试。
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # 注释： 设置 `features`（投影后的残差特征 epsilon_h；Xi 不直接用原始像素残差而用这个特征）为 `self.encoder(x)`；这样后续步骤可以复用该状态而不是重复计算。
        features = self.encoder(x)
        # 注释： 返回 `self.mean(features), self.logvar(features).clamp(-12.0, 12.0)`，把当前计算结果交给调用方继续训练、记录或判断。
        return self.mean(features), self.logvar(features).clamp(-12.0, 12.0)

    # 注释： 定义函数 `reparameterize`：封装当前脚本中的专用实验逻辑，方便复用和测试。
    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        # 注释： 检查条件 `self.training`；该条件决定是否进入当前实验分支。
        if self.training:
            # 注释： 返回 `mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)`，把当前计算结果交给调用方继续训练、记录或判断。
            return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        # 注释： 返回 `mean`，把当前计算结果交给调用方继续训练、记录或判断。
        return mean

    # 注释： 定义函数 `forward`：封装当前脚本中的专用实验逻辑，方便复用和测试。
    def forward(self, x: Tensor):
        # 注释： 设置 `mean, logvar`（mean, logvar）为 `self.encode(x)`；这样后续步骤可以复用该状态而不是重复计算。
        mean, logvar = self.encode(x)
        # 注释： 设置 `latent`（latent）为 `self.reparameterize(mean, logvar)`；这样后续步骤可以复用该状态而不是重复计算。
        latent = self.reparameterize(mean, logvar)
        # 注释： 返回 `self.decoder(latent), mean, logvar`，把当前计算结果交给调用方继续训练、记录或判断。
        return self.decoder(latent), mean, logvar


# 注释： 定义函数 `make_orthogonal_embedding`：生成受控流形使用的固定随机正交嵌入矩阵。
# 受控流形本身先在低维内禀坐标里生成，例如 circle 是 `[cos(theta), sin(theta)]`，
# torus 是 `[cos(theta), sin(theta), cos(phi), sin(phi)]`；这里再把这些低维坐标
# 乘到一个高维正交基上，得到观测空间里的向量。这样做的目的不是增加标签信息，
# 而是避免模型只看到手工排列好的坐标轴，让实验更接近“未知观测空间里的潜在结构”。
def make_orthogonal_embedding(observation_dim: int, intrinsic_dim: int, seed: int) -> Tensor:
    # 注释： 创建一个独立的 PyTorch 随机数发生器，并用 `manual_seed(seed)` 固定它的初始状态。
    # `manual_seed` 的意思是“手动设定随机种子”：只要 seed 一样，后面的 `torch.randn`
    # 就会生成完全相同的随机矩阵。这里不用全局 `torch.manual_seed`，是为了让“嵌入矩阵采样”
    # 和模型初始化、VAE 重参数采样、DataLoader 打乱等随机过程彼此隔离。
    # 传进来的 seed 通常是 `args.seed + 123`：`args.seed` 控制整次实验复现，
    # `+123` 是给“正交嵌入矩阵”单独留出的偏移量，避免它和别的随机模块共用同一段随机序列。
    generator = make_torch_generator(seed)
    # 注释： 先采样一个 `[observation_dim, intrinsic_dim]` 的高斯随机矩阵。
    # 它还不是正交矩阵，只是作为 QR 分解的原料；因为 generator 已经固定 seed，
    # 所以同一实验配置下这里得到的原始随机矩阵也固定。
    matrix = torch.randn(observation_dim, intrinsic_dim, generator=generator)
    # 注释： 对随机矩阵做 QR 分解，取 Q 作为列正交矩阵。
    # Q 的列向量两两正交且范数为 1，所以低维流形坐标映射到观测空间后不会人为拉伸某个内禀方向。
    # `_` 是 R 矩阵，这里不需要；我们只需要正交基 Q。
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    # 注释： 返回 float32 的正交嵌入矩阵，后续数据集会用 `u @ Q.T` 把内禀坐标投到观测空间。
    return q.float()


# 注释： 定义类 `ControlledPhaseDataset`，用于把残差投影、白化、窗口谱读数封装到独立组件中。
class ControlledPhaseDataset(Dataset[tuple[Tensor, Tensor]]):
    """Controlled stream: circle -> torus -> torus + two-dimensional local split."""

    # 注释： 定义函数 `__init__`：保存初始化参数并准备内部缓存，真正依赖数据的矩阵稍后懒加载。
    def __init__(
        # 注释： 在当前多行结构中加入 `self`；多行写法便于审计实验参数。
        self,
        # 注释： 声明字段 `phase`，表示 phase；dataclass 会把它纳入日志对象。
        phase: str,
        # 注释： 声明字段 `embedding`，表示 embedding；dataclass 会把它纳入日志对象。
        embedding: Tensor,
        # 注释： 声明字段 `size`，表示 size；dataclass 会把它纳入日志对象。
        size: int,
        # 注释： 声明字段 `noise_std`，表示 noise_std；dataclass 会把它纳入日志对象。
        noise_std: float,
        # 注释： 声明字段 `seed`，表示 seed；dataclass 会把它纳入日志对象。
        seed: int,
        # 注释： 声明字段 `factor_bins`，表示 factor_bins；dataclass 会把它纳入日志对象。
        factor_bins: int,
        # 注释： 声明字段 `split_offset`，表示 split_offset；dataclass 会把它纳入日志对象。
        split_offset: float,
        # 注释： 声明字段 `local_probability`，表示 local_probability；dataclass 会把它纳入日志对象。
        local_probability: float,
        # 注释： 声明字段 `split_mode`，表示 split_mode；dataclass 会把它纳入日志对象。
        split_mode: str,
    # 注释： 执行 `) -> None:`；这是当前实验流程中的必要操作。
    ) -> None:
        # 注释： 检查条件 `phase not in {"s1", "t2", "local_split"}`；该条件决定是否进入当前实验分支。
        if phase not in {"s1", "t2", "local_split"}:
            # 注释： 抛出 `ValueError(f"Unknown controlled phase: {phase}")`，用于明确暴露非法参数或子进程失败。
            raise ValueError(f"Unknown controlled phase: {phase}")
        # 注释： 设置 `self.phase`（phase）为 `phase`；这样后续步骤可以复用该状态而不是重复计算。
        self.phase = phase
        # 注释： 设置 `self.embedding`（embedding）为 `embedding.float()`；这样后续步骤可以复用该状态而不是重复计算。
        self.embedding = embedding.float()
        # 注释： 设置 `self.size`（size）为 `int(size)`；这样后续步骤可以复用该状态而不是重复计算。
        self.size = int(size)
        # 注释： 设置 `self.noise_std`（noise_std）为 `float(noise_std)`；这样后续步骤可以复用该状态而不是重复计算。
        self.noise_std = float(noise_std)
        # 注释： 设置 `self.seed`（seed）为 `int(seed)`；这样后续步骤可以复用该状态而不是重复计算。
        self.seed = int(seed)
        # 注释： 设置 `self.factor_bins`（factor_bins）为 `int(factor_bins)`；这样后续步骤可以复用该状态而不是重复计算。
        self.factor_bins = int(factor_bins)
        # 注释： 设置 `self.split_offset`（split_offset）为 `float(split_offset)`；这样后续步骤可以复用该状态而不是重复计算。
        self.split_offset = float(split_offset)
        # 注释： 设置 `self.local_probability`（local_probability）为 `float(local_probability)`；这样后续步骤可以复用该状态而不是重复计算。
        self.local_probability = float(local_probability)
        # 注释： 设置 `self.split_mode`（split_mode）为 `split_mode`；这样后续步骤可以复用该状态而不是重复计算。
        self.split_mode = split_mode

    # 注释： 定义函数 `__len__`：封装当前脚本中的专用实验逻辑，方便复用和测试。
    def __len__(self) -> int:
        # 注释： 返回 `self.size`，把当前计算结果交给调用方继续训练、记录或判断。
        return self.size

    # 注释： 定义函数 `__getitem__`：根据样本编号 index 生成一个确定的受控流形样本。
    # 这里不是从磁盘读固定图片，而是“按 index 可复现地现场生成”角度、局部分裂标签和噪声。
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # 注释： 为当前样本创建独立随机数发生器，并用 `manual_seed(...)` 固定这个样本的随机性。
        # `manual_seed` 会把 generator 重置到确定状态，因此同一个 `(self.seed, index)` 总会生成同一个样本。
        # 种子写成 `self.seed * 1_000_003 + int(index)` 是为了同时编码两件事：
        # 1. `self.seed`：数据集级随机种子，不同实验 seed 会得到不同但可复现的数据流；
        # 2. `index`：样本编号，保证同一个数据集内不同 index 不会生成同一组角度/噪声。
        # 1_000_003 是一个较大的质数，用作混合系数，让 seed 的主导部分来自数据集 seed，
        # 同时把 index 均匀错开，降低不同 `(seed, index)` 组合碰撞到相近随机序列的概率。
        # 这种写法还有一个实用好处：即使 DataLoader shuffle 了访问顺序，index 对应的样本内容也不变。
        generator = make_torch_generator(self.seed * 1_000_003 + int(index))
        # 注释： 设置 `theta`（theta）为 `2.0 * math.pi * torch.rand((), generator=generator)`；这样后续步骤可以复用该状态而不是重复计算。
        theta = 2.0 * math.pi * torch.rand((), generator=generator)
        # 注释： 设置 `phi`（phi）为 `2.0 * math.pi * torch.rand((), generator=generator)`；这样后续步骤可以复用该状态而不是重复计算。
        phi = 2.0 * math.pi * torch.rand((), generator=generator)
        # 注释： 设置 `coord`（coord）为 `torch.zeros(self.embedding.shape[1])`；这样后续步骤可以复用该状态而不是重复计算。
        coord = torch.zeros(self.embedding.shape[1])
        # 注释： 设置 `theta_bin`（theta_bin）为 `bin_angle(theta, self.factor_bins)`；这样后续步骤可以复用该状态而不是重复计算。
        theta_bin = bin_angle(theta, self.factor_bins)
        # 注释： 设置 `phi_bin`（phi_bin）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
        phi_bin = 0
        # 注释： 设置 `split_bin`（split_bin）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
        split_bin = 0
        # 注释： 设置 `stage_id`（stage_id）为 `{"s1": 0, "t2": 1, "local_split": 2}[self.phase]`；这样后续步骤可以复用该状态而不是重复计算。
        stage_id = {"s1": 0, "t2": 1, "local_split": 2}[self.phase]

        # 注释： 设置 `coord[0]`（coord[0]）为 `torch.cos(theta)`；这样后续步骤可以复用该状态而不是重复计算。
        coord[0] = torch.cos(theta)
        # 注释： 设置 `coord[1]`（coord[1]）为 `torch.sin(theta)`；这样后续步骤可以复用该状态而不是重复计算。
        coord[1] = torch.sin(theta)
        # 注释： 检查条件 `self.phase in {"t2", "local_split"}`；该条件决定是否进入当前实验分支。
        if self.phase in {"t2", "local_split"}:
            # 注释： 设置 `coord[2]`（coord[2]）为 `torch.cos(phi)`；这样后续步骤可以复用该状态而不是重复计算。
            coord[2] = torch.cos(phi)
            # 注释： 设置 `coord[3]`（coord[3]）为 `torch.sin(phi)`；这样后续步骤可以复用该状态而不是重复计算。
            coord[3] = torch.sin(phi)
            # 注释： 设置 `phi_bin`（phi_bin）为 `1 + bin_angle(phi, self.factor_bins)`；这样后续步骤可以复用该状态而不是重复计算。
            phi_bin = 1 + bin_angle(phi, self.factor_bins)

        # 注释： 检查条件 `self.phase == "local_split" and torch.rand((), generator=generator).item() < self.local_probability`；该条件决定是否进入当前实验分支。
        if self.phase == "local_split" and torch.rand((), generator=generator).item() < self.local_probability:
            # 注释： 检查条件 `self.split_mode == "discrete"`；该条件决定是否进入当前实验分支。
            if self.split_mode == "discrete":
                # 注释： 设置 `split_class`（split_class）为 `int(torch.clamp(torch.floor(phi / (2.0 * math.pi) * 4), 0, 3).item())`；这样后续步骤可以复用该状态而不是重复计算。
                split_class = int(torch.clamp(torch.floor(phi / (2.0 * math.pi) * 4), 0, 3).item())
                # 注释： 设置 `split_bin`（split_bin）为 `1 + split_class`；这样后续步骤可以复用该状态而不是重复计算。
                split_bin = 1 + split_class
                # 注释： 设置 `split_angle`（split_angle）为 `2.0 * math.pi * split_class / 4.0`；这样后续步骤可以复用该状态而不是重复计算。
                split_angle = 2.0 * math.pi * split_class / 4.0
            # 注释： 前一个条件不成立时继续检查 `self.split_mode == "continuous_local"`，用于区分不同数据集/阶段/模式。
            elif self.split_mode == "continuous_local":
                # 注释： 检查条件 `phi.item() < math.pi`；该条件决定是否进入当前实验分支。
                if phi.item() < math.pi:
                    # 注释： 设置 `split_bin`（split_bin）为 `1`；这样后续步骤可以复用该状态而不是重复计算。
                    split_bin = 1
                    # 注释： 设置 `split_angle`（split_angle）为 `2.0 * math.pi * torch.rand((), generator=generator).item()`；这样后续步骤可以复用该状态而不是重复计算。
                    split_angle = 2.0 * math.pi * torch.rand((), generator=generator).item()
                # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
                else:
                    # 注释： 设置 `split_angle`（split_angle）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
                    split_angle = None
            # 注释： 前一个条件不成立时继续检查 `self.split_mode == "continuous_local_4d"`，用于区分不同数据集/阶段/模式。
            elif self.split_mode == "continuous_local_4d":
                # 注释： 设置 `split_class`（split_class）为 `int(torch.clamp(torch.floor(phi / (2.0 * math.pi) * 4), 0, 3).item())`；这样后续步骤可以复用该状态而不是重复计算。
                split_class = int(torch.clamp(torch.floor(phi / (2.0 * math.pi) * 4), 0, 3).item())
                # 注释： 设置 `split_bin`（split_bin）为 `1 + split_class`；这样后续步骤可以复用该状态而不是重复计算。
                split_bin = 1 + split_class
                # 注释： 设置 `centers`（centers）为 `torch.eye(4, dtype=coord.dtype)`；这样后续步骤可以复用该状态而不是重复计算。
                centers = torch.eye(4, dtype=coord.dtype)
                # 注释： 设置 `jitter`（jitter）为 `0.1 * (2.0 * torch.rand(4, generator=generator) - 1.0)`；这样后续步骤可以复用该状态而不是重复计算。
                jitter = 0.1 * (2.0 * torch.rand(4, generator=generator) - 1.0)
                # 注释： 设置 `coord[4:8]`（coord[4）为 `self.split_offset * (centers[split_class] + jitter)`；这样后续步骤可以复用该状态而不是重复计算。
                coord[4:8] = self.split_offset * (centers[split_class] + jitter)
                # 注释： 设置 `split_angle`（split_angle）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
                split_angle = None
            # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
            else:
                # 注释： 抛出 `ValueError(f"Unknown split mode: {self.split_mode}")`，用于明确暴露非法参数或子进程失败。
                raise ValueError(f"Unknown split mode: {self.split_mode}")

            # Delta_s occupies exactly two intrinsic dimensions.
            # 注释： 检查条件 `split_angle is not None`；该条件决定是否进入当前实验分支。
            if split_angle is not None:
                # 注释： 设置 `coord[4]`（coord[4]）为 `self.split_offset * math.cos(split_angle)`；这样后续步骤可以复用该状态而不是重复计算。
                coord[4] = self.split_offset * math.cos(split_angle)
                # 注释： 设置 `coord[5]`（coord[5]）为 `self.split_offset * math.sin(split_angle)`；这样后续步骤可以复用该状态而不是重复计算。
                coord[5] = self.split_offset * math.sin(split_angle)

        # 注释： 设置 `observed`（observed）为 `self.embedding @ coord`；这样后续步骤可以复用该状态而不是重复计算。
        observed = self.embedding @ coord
        # 注释： 检查条件 `self.noise_std > 0.0`；该条件决定是否进入当前实验分支。
        if self.noise_std > 0.0:
            # 注释： 设置 `observed`（observed）为 `observed + self.noise_std * torch.randn(observed.shape, generator=generator)`；这样后续步骤可以复用该状态而不是重复计算。
            observed = observed + self.noise_std * torch.randn(observed.shape, generator=generator)
        # 注释： 设置 `labels`（labels）为 `torch.tensor([theta_bin, phi_bin, split_bin, stage_id], dtype=torch.long)`；这样后续步骤可以复用该状态而不是重复计算。
        labels = torch.tensor([theta_bin, phi_bin, split_bin, stage_id], dtype=torch.long)
        # 注释： 返回 `observed.float(), labels`，把当前计算结果交给调用方继续训练、记录或判断。
        return observed.float(), labels


# 注释： 定义函数 `bin_angle`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def bin_angle(angle: Tensor, bins: int) -> int:
    # 注释： 返回 `int(torch.clamp(torch.floor(angle / (2.0 * math.pi) * bins), 0, bins - 1).item())`，把当前计算结果交给调用方继续训练、记录或判断。
    return int(torch.clamp(torch.floor(angle / (2.0 * math.pi) * bins), 0, bins - 1).item())


# 注释： 定义函数 `parse_args`：解析命令行参数，决定数据集、A/B 模式、阈值模式和输出目录。
def parse_args() -> argparse.Namespace:
    # 注释： 设置 `parser`（命令行参数解析器；集中声明实验入口可调参数）为 `argparse.ArgumentParser(description="Precision-weighted residual signal experiment.")`；这样后续步骤可以复用该状态而不是重复计算。
    parser = argparse.ArgumentParser(description="Precision-weighted residual signal experiment.")
    # 注释： 注册命令行参数 `--dataset`：选择 controlled 或 cifar 数据集。，这样无需改代码即可切换实验设置。
    parser.add_argument("--dataset", choices=("controlled", "cifar"), required=True)
    # 注释： 注册命令行参数 `--output-dir`：设置实验结果输出目录。，这样无需改代码即可切换实验设置。
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "pw_signal")
    # 注释： 注册命令行参数 `--device`：设置训练设备。，这样无需改代码即可切换实验设置。
    parser.add_argument("--device", default="auto")
    # 注释： 注册命令行参数 `--seed`：--seed，这样无需改代码即可切换实验设置。
    parser.add_argument("--seed", type=int, default=0)
    # 注释： 注册命令行参数 `--mode`：选择 loss_only 或 loss_plus_xi。，这样无需改代码即可切换实验设置。
    parser.add_argument("--mode", choices=("loss_only", "loss_plus_xi"), default="loss_only")
    # 注释： 注册命令行参数 `--beta-kl`：--beta-kl，这样无需改代码即可切换实验设置。
    parser.add_argument("--beta-kl", type=float, default=0.001)
    # 注释： 注册命令行参数 `--lambda-xi`：设置 Xi 正则项权重。，这样无需改代码即可切换实验设置。
    parser.add_argument("--lambda-xi", type=float, default=1.0)
    # 注释： 注册 `--xi-signal`：选择当前训练使用哪一种 Xi 读数；`pw` 是旧精度加权谱信号，`absorbability` 是信号一，`fast_slow_divergence` 是信号二，`residual_predictive_info` 是信号三。
    parser.add_argument("--xi-signal", choices=("pw", "absorbability", "fast_slow_divergence", "residual_predictive_info"), default="pw")
    # 注释： 注册 `--abs-fast-steps`：设置可吸收性探测里的 n_fast，也就是临时连续适应多少步后再测 E_after；文档示例给出 20 步。
    parser.add_argument("--abs-fast-steps", type=int, default=20)
    # 注释： 注册 `--abs-fast-lr`：设置临时快适应优化器的学习率；默认 0 表示复用正式训练的 learning_rate，避免又多一个必须手动设置的超参。
    parser.add_argument("--abs-fast-lr", type=float, default=0.0)
    # 注释： 注册 `--abs-tau`：手动指定 tau_abs；若小于 0，则用校准阶段的 E_after 分位数自动估计。
    parser.add_argument("--abs-tau", type=float, default=-1.0)
    # 注释： 注册 `--abs-probe-every`：控制可吸收性信号每隔多少 step 做一次快适应探测；默认 1 表示每步都测，数值越大实验越快但曲线更稀疏。
    parser.add_argument("--abs-probe-every", type=int, default=1)
    # 注释： 注册 `--div-fast-gamma`：快慢分歧信号里的快记忆更新率 gamma_f；若保持默认负值，则自动取 `10 * div_slow_gamma`。
    parser.add_argument("--div-fast-gamma", type=float, default=-1.0)
    # 注释： 注册 `--div-slow-gamma`：快慢分歧信号里的慢记忆更新率 gamma_s；越小越代表长期旧常态。
    parser.add_argument("--div-slow-gamma", type=float, default=0.005)
    # 注释： 注册 `--div-alpha`：快慢分歧信号里的慢协方差 shrinkage 系数；用于保持历史实验脚本参数兼容。
    parser.add_argument("--div-alpha", type=float, default=0.01)
    # 注释： 注册 `--info-factor`：第三信号要让残差预测哪个已知生成因子；当前 controlled 实验使用 split，因为它代表 local_split 阶段新增而 latent 可能未编码的局部自由度。
    parser.add_argument("--info-factor", choices=("theta", "phi", "split", "stage"), default="split")
    # 注释： 注册 `--info-probe-steps`：第三信号内部小分类器训练多少步；它只读残差、不更新主模型，步数越大越接近残差可预测信息下界但越慢。
    parser.add_argument("--info-probe-steps", type=int, default=50)
    # 注释： 注册 `--info-probe-lr`：第三信号内部线性分类器学习率；该分类器是临时诊断器，不是 VAE 的一部分。
    parser.add_argument("--info-probe-lr", type=float, default=0.05)
    # 注释： 注册命令行参数 `--learning-rate`：设置 VAE 优化器学习率。，这样无需改代码即可切换实验设置。
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # 注释： 注册命令行参数 `--weight-decay`：--weight-decay，这样无需改代码即可切换实验设置。
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # 注释： 注册命令行参数 `--batch-size`：设置训练 batch size。，这样无需改代码即可切换实验设置。
    parser.add_argument("--batch-size", type=int, default=128)
    # 注释： 注册命令行参数 `--log-every`：设置 train_history.csv 记录间隔。，这样无需改代码即可切换实验设置。
    parser.add_argument("--log-every", type=int, default=20)
    # 注释： 注册命令行参数 `--quiet-console`：--quiet-console，这样无需改代码即可切换实验设置。
    parser.add_argument("--quiet-console", action="store_true")
    # 注释： 注册命令行参数 `--online-probe-every`：设置在线 probe 间隔。，这样无需改代码即可切换实验设置。
    parser.add_argument("--online-probe-every", type=int, default=0)
    # 注释： 注册一个命令行参数，用于从外部控制训练、校准或 probe 行为。
    parser.add_argument(
        # 注释： 加入字符串 `--online-probe-start-offset`：设置每阶段开始多少 step 后第一次 probe。
        "--online-probe-start-offset",
        # 注释： 设置 `type`（type）为 `int`；这样后续步骤可以复用该状态而不是重复计算。
        type=int,
        # 注释： 设置 `default`（default）为 `100`；这样后续步骤可以复用该状态而不是重复计算。
        default=100,
        # 注释： 设置 `help`（help）为 `"First stage-local step used for online probes. With every=200 and offset=100, probes run at 100, 300, ... inside each stage."`；这样后续步骤可以复用该状态而不是重复计算。
        help="First stage-local step used for online probes. With every=200 and offset=100, probes run at 100, 300, ... inside each stage.",
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    # 注释： 注册命令行参数 `--online-probe-max-samples`：设置在线 probe 使用的最大样本数。，这样无需改代码即可切换实验设置。
    parser.add_argument("--online-probe-max-samples", type=int, default=2048)
    # 注释： 注册命令行参数 `--online-probe-train-steps`：设置在线 probe 分类器最大训练步数。，这样无需改代码即可切换实验设置。
    parser.add_argument("--online-probe-train-steps", type=int, default=1000)
    # 注释： 注册命令行参数 `--online-probe-early-stopping-patience`：设置在线 probe 早停 patience。，这样无需改代码即可切换实验设置。
    parser.add_argument("--online-probe-early-stopping-patience", type=int, default=50)
    # 注释： 注册命令行参数 `--online-probe-early-stopping-min-delta`：设置在线 probe 早停最小改善量。，这样无需改代码即可切换实验设置。
    parser.add_argument("--online-probe-early-stopping-min-delta", type=float, default=1e-6)
    # 注释： 注册一个命令行参数，用于从外部控制训练、校准或 probe 行为。
    parser.add_argument(
        # 注释： 加入字符串 `--online-probe-tasks`：设置在线 probe 的预测任务。
        "--online-probe-tasks",
        # 注释： 设置 `nargs`（nargs）为 `"+"`；这样后续步骤可以复用该状态而不是重复计算。
        nargs="+",
        # 注释： 设置 `choices`（choices）为 `("theta", "phi", "split", "stage")`；这样后续步骤可以复用该状态而不是重复计算。
        choices=("theta", "phi", "split", "stage"),
        # 注释： 设置 `default`（default）为 `("theta",)`；这样后续步骤可以复用该状态而不是重复计算。
        default=("theta",),
        # 注释： 设置 `help`（help）为 `"Controlled-data online probe tasks. The final experiment uses theta as the primary shared factor."`；这样后续步骤可以复用该状态而不是重复计算。
        help="Controlled-data online probe tasks. The final experiment uses theta as the primary shared factor.",
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )

    # 注释： 注册命令行参数 `--pw-window-samples`：--pw-window-samples，这样无需改代码即可切换实验设置。
    parser.add_argument("--pw-window-samples", type=int, default=64)
    # 注释： 注册命令行参数 `--pw-gamma`：--pw-gamma，这样无需改代码即可切换实验设置。
    parser.add_argument("--pw-gamma", type=float, default=0.005)
    # 注释： 注册命令行参数 `--pw-alpha`：--pw-alpha，这样无需改代码即可切换实验设置。
    parser.add_argument("--pw-alpha", type=float, default=0.1)
    # 注释： 注册命令行参数 `--pw-null-quantile`：设置 empirical tau_null 校准分位数。，这样无需改代码即可切换实验设置。
    parser.add_argument("--pw-null-quantile", type=float, default=0.95)
    # 注释： 注册命令行参数 `--pw-calibration-warmup-steps`：--pw-calibration-warmup-steps，这样无需改代码即可切换实验设置。
    parser.add_argument("--pw-calibration-warmup-steps", type=int, default=500)
    # 注释： 注册命令行参数 `--pw-calibration-steps`：--pw-calibration-steps，这样无需改代码即可切换实验设置。
    parser.add_argument("--pw-calibration-steps", type=int, default=500)
    # 注释： 注册一个命令行参数，用于从外部控制训练、校准或 probe 行为。
    parser.add_argument(
        # 注释： 加入字符串 `--pw-threshold-mode`：设置 Xi 阈值类型，empirical 或 mp_edge。
        "--pw-threshold-mode",
        # 注释： 设置 `choices`（choices）为 `("empirical", "mp_edge")`；这样后续步骤可以复用该状态而不是重复计算。
        choices=("empirical", "mp_edge"),
        # 注释： 设置 `default`（default）为 `"empirical"`；这样后续步骤可以复用该状态而不是重复计算。
        default="empirical",
        # 注释： 设置 `help`（help）为 `"empirical uses calibrated tau_null; mp_edge implements scheme A with the Marchenko-Pastur right edge."`；这样后续步骤可以复用该状态而不是重复计算。
        help="empirical uses calibrated tau_null; mp_edge implements scheme A with the Marchenko-Pastur right edge.",
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    parser.add_argument(
        "--pw-spectral-readout",
        choices=("edge", "mass", "gap"),
        default="edge",
        help=(
            "Compact spectral residual readout. edge uses "
            "[lambda_1/lambda_edge-1]_+; mass sums all super-edge "
            "eigenvalue excesses; gap uses the isolated top spike gap."
        ),
    )
    # 注释： 注册一个命令行参数，用于从外部控制训练、校准或 probe 行为。
    parser.add_argument(
        # 注释： 加入字符串 `--pw-baseline-mode`：设置 baseline 模式，previous_stage 或 state_conditioned。
        "--pw-baseline-mode",
        # 注释： 设置 `choices`（choices）为 `("previous_stage", "state_conditioned")`；这样后续步骤可以复用该状态而不是重复计算。
        choices=("previous_stage", "state_conditioned"),
        # 注释： 设置 `default`（default）为 `"previous_stage"`；这样后续步骤可以复用该状态而不是重复计算。
        default="previous_stage",
        # 注释： 设置 `help`（help）为 `"previous_stage keeps the old reference-stage logic; state_conditioned implements scheme B with one baseline per current state."`；这样后续步骤可以复用该状态而不是重复计算。
        help="previous_stage keeps the old reference-stage logic; state_conditioned implements scheme B with one baseline per current state.",
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    parser.add_argument(
        "--pw-readout-mode",
        choices=("current", "legacy_previous_stage"),
        default="current",
        help=(
            "current keeps the newer read-before-update routing; "
            "legacy_previous_stage matches Scheme 2 by reading Xi from "
            "the previous-stage baseline with the current batch included "
            "in the residual window."
        ),
    )
    # 注释： 注册一个命令行参数，用于从外部控制训练、校准或 probe 行为。
    parser.add_argument(
        # 注释： 加入字符串 `--pw-baseline-warmup-batches`：设置 baseline 初始化需要累计的 batch 数。
        "--pw-baseline-warmup-batches",
        # 注释： 设置 `type`（type）为 `int`；这样后续步骤可以复用该状态而不是重复计算。
        type=int,
        # 注释： 设置 `default`（default）为 `20`；这样后续步骤可以复用该状态而不是重复计算。
        default=20,
        # 注释： 设置 `help`（help）为 `"Number of batches used to initialize mu_0 and Sigma_0 before Xi is emitted."`；这样后续步骤可以复用该状态而不是重复计算。
        help="Number of batches used to initialize mu_0 and Sigma_0 before Xi is emitted.",
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    # 注释： 注册命令行参数 `--residual-feature-dim`：--residual-feature-dim，这样无需改代码即可切换实验设置。
    parser.add_argument("--residual-feature-dim", type=int, default=128)

    # 注释： 注册命令行参数 `--observation-dim`：--observation-dim，这样无需改代码即可切换实验设置。
    parser.add_argument("--observation-dim", type=int, default=64)
    # 注释： 注册命令行参数 `--controlled-latent-dim`：设置 controlled VAE latent 维度。，这样无需改代码即可切换实验设置。
    parser.add_argument("--controlled-latent-dim", type=int, default=32)
    # 注释： 注册命令行参数 `--controlled-hidden-dim`：设置 controlled MLP hidden 维度。，这样无需改代码即可切换实验设置。
    parser.add_argument("--controlled-hidden-dim", type=int, default=256)
    # 注释： 注册命令行参数 `--controlled-hidden-layers`：设置 controlled MLP hidden 层数。，这样无需改代码即可切换实验设置。
    parser.add_argument("--controlled-hidden-layers", type=int, default=2)
    # 注释： 注册命令行参数 `--controlled-noise-std`：--controlled-noise-std，这样无需改代码即可切换实验设置。
    parser.add_argument("--controlled-noise-std", type=float, default=0.02)
    # 注释： 注册命令行参数 `--controlled-stage-steps`：设置 controlled 三个阶段各自训练步数。，这样无需改代码即可切换实验设置。
    parser.add_argument("--controlled-stage-steps", type=int, nargs=3, default=(500, 500, 500))
    # 注释： 注册命令行参数 `--controlled-stage-size`：--controlled-stage-size，这样无需改代码即可切换实验设置。
    parser.add_argument("--controlled-stage-size", type=int, default=4096)
    # 注释： 注册命令行参数 `--factor-bins`：--factor-bins，这样无需改代码即可切换实验设置。
    parser.add_argument("--factor-bins", type=int, default=8)
    # 注释： 注册命令行参数 `--split-offset`：设置 local split 偏移强度。，这样无需改代码即可切换实验设置。
    parser.add_argument("--split-offset", type=float, default=0.35)
    # 注释： 注册命令行参数 `--local-probability`：设置 local split 自由度出现概率。，这样无需改代码即可切换实验设置。
    parser.add_argument("--local-probability", type=float, default=0.7)
    # 注释： 注册命令行参数 `--intrinsic-dim`：设置受控流形构造中的内在维度。，这样无需改代码即可切换实验设置。
    parser.add_argument("--intrinsic-dim", type=int, default=6)
    # 注释： 注册命令行参数 `--split-mode`：设置 local split 构造模式。，这样无需改代码即可切换实验设置。
    parser.add_argument("--split-mode", choices=("discrete", "continuous_local", "continuous_local_4d"), default="discrete")

    # 注释： 注册命令行参数 `--cifar10-root`：--cifar10-root，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar10-root", type=Path, default=ROOT / "data")
    # 注释： 注册命令行参数 `--cifarc-root`：--cifarc-root，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifarc-root", type=Path, default=ROOT / "data" / "CIFAR-C")
    # 注释： 注册命令行参数 `--cifar-full-c`：--cifar-full-c，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-full-c", action="store_true")
    # 注释： 注册命令行参数 `--cifar-mixed-size`：--cifar-mixed-size，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-mixed-size", type=int, default=50_000)
    # 注释： 注册命令行参数 `--cifar-stage-steps`：设置 CIFAR clean/CIFAR-C 两阶段训练步数。，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-stage-steps", type=int, nargs=2, default=(500, 500))
    # 注释： 注册命令行参数 `--cifar-latent-dim`：--cifar-latent-dim，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-latent-dim", type=int, default=32)
    # 注释： 注册命令行参数 `--cifar-hidden-dim`：--cifar-hidden-dim，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-hidden-dim", type=int, default=512)
    # 注释： 注册命令行参数 `--cifar-model`：--cifar-model，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-model", choices=("standard", "strong"), default="strong")
    # 注释： 注册命令行参数 `--cifar-base-channels`：--cifar-base-channels，这样无需改代码即可切换实验设置。
    parser.add_argument("--cifar-base-channels", type=int, default=64)
    # 注释： 注册命令行参数 `--download-cifar10`：--download-cifar10，这样无需改代码即可切换实验设置。
    parser.add_argument("--download-cifar10", action="store_true")
    # 注释： 返回 `parser.parse_args()`，把当前计算结果交给调用方继续训练、记录或判断。
    return parser.parse_args()


# 注释： 定义函数 `main`：组织完整方案 A/B 网格实验，遍历维度和训练模式。
def main() -> None:
    # 注释： 设置 `args`（命令行解析后的实验参数；后续训练、校准、probe 都从这里读取配置）为 `parse_args()`；这样后续步骤可以复用该状态而不是重复计算。
    args = parse_args()
    # 注释： 执行 `set_seed(args.seed)`；这是当前实验流程中的必要操作。
    set_seed(args.seed)
    # 注释： 设置 `device`（device）为 `resolve_device(args.device)`；这样后续步骤可以复用该状态而不是重复计算。
    device = resolve_device(args.device)
    # 注释： 设置 `run_dir`（单次训练 run 的输出目录；保存 train_history、probe 和 checkpoint）为 `args.output_dir / args.dataset / f"{args.mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"`；这样后续步骤可以复用该状态而不是重复计算。
    run_dir = args.output_dir / args.dataset / f"{args.mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    # 注释： 设置 `run_dir.mkdir(parents`（mkdir(parents）为 `True, exist_ok=True)`；这样后续步骤可以复用该状态而不是重复计算。
    run_dir.mkdir(parents=True, exist_ok=True)
    # 注释： 执行 `write_json(run_dir / "resolved_args.json", vars_for_json(args))`；这是当前实验流程中的必要操作。
    write_json(run_dir / "resolved_args.json", vars_for_json(args))

    # 注释： 设置 `calibration`（校准阶段得到的阈值字典；empirical 模式使用，MP 模式主要保留日志兼容性）为 `run_calibration(args, device, run_dir)`；这样后续步骤可以复用该状态而不是重复计算。
    calibration = run_calibration(args, device, run_dir)
    # 注释： 执行 `write_json(run_dir / "calibration_summary.json", calibration)`；这是当前实验流程中的必要操作。
    write_json(run_dir / "calibration_summary.json", calibration)
    # 注释： 设置 `model, stages, phase_boundaries`（model, stages, phase_boundaries）为 `build_experiment(args, device)`；这样后续步骤可以复用该状态而不是重复计算。
    model, stages, phase_boundaries = build_experiment(args, device)
    # 注释： 设置 `model`（当前训练的 VAE 模型；固定拓扑不做结构编辑）为 `model.to(device)`；这样后续步骤可以复用该状态而不是重复计算。
    model = model.to(device)
    # 注释： 执行 `train_formal(args, device, run_dir, model, stages, phase_boundaries, calibration)`；这是当前实验流程中的必要操作。
    train_formal(args, device, run_dir, model, stages, phase_boundaries, calibration)
    # 注释： 保存模型、参数和校准信息，方便复现实验或后续加载 probe。
    torch.save({"model_state": model.state_dict(), "args": vars_for_json(args), "calibration": calibration}, run_dir / "last.pt")
    # 注释： 执行 `plot_curves(run_dir, phase_boundaries)`；这是当前实验流程中的必要操作。
    plot_curves(run_dir, phase_boundaries)
    # 注释： 检查条件 `not args.quiet_console`；该条件决定是否进入当前实验分支。
    if not args.quiet_console:
        # 注释： 执行 `print(f"[done] {run_dir}")`；这是当前实验流程中的必要操作。
        print(f"[done] {run_dir}")


# 注释： 定义函数 `run_calibration`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    # 注释： 复制当前模型参数；可吸收性探测需要临时训练再回滚，clone 后的快照不会被后续原地更新改坏。
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_state(model: nn.Module, state: dict[str, Tensor]) -> None:
    # 注释： 把临时快适应前的参数快照完整加载回模型，确保探测过程不改变正式训练轨迹。
    model.load_state_dict(state, strict=True)


def _projected_residual_features(model: nn.Module, projector: ResidualFeatureProjector, batches: list[Tensor]) -> Tensor:
    # 注释： 保存每个 batch 的 h(epsilon)；最后拼成文档中最近 W 个残差向量形成的残差云。
    pieces: list[Tensor] = []
    # 注释： 逐批重新计算残差，保证快适应前后都在同一批样本上比较。
    for batch in batches:
        # 注释： 用当前模型重构输入；mean/logvar 不参与残差信号本身，所以用下划线占位。
        reconstruction, _mean, _logvar = forward_vae(model, batch)
        # 注释： 计算 epsilon=o-o_hat，这就是文档中所有信号共同使用的残差向量。
        residual = batch - reconstruction
        # 注释： 通过固定读出 h 把残差转成 d_epsilon 维特征，避免高维图像残差协方差过大。
        pieces.append(projector(residual))
    # 注释： 沿样本维拼接，得到形状 `[N, d_epsilon]` 的残差特征窗口。
    return torch.cat(pieces, dim=0)


def _top_direction_and_energy(features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # 注释： 样本数少于 2 时协方差没有统计意义；返回零值让调用方把窗口视为未 ready。
    if features.shape[0] < 2:
        # 注释： 创建与残差特征维度一致的零方向，避免后续矩阵乘法尺寸错误。
        direction = features.new_zeros(features.shape[1])
        # 注释： 返回零最大特征值和零能量，表示当前没有可用主残差方向。
        return direction, features.new_tensor(0.0), features.new_tensor(0.0)
    # 注释： 按文档公式 C_hat_t=1/W sum epsilon epsilon^T 估计窗口协方差。
    cov = features.transpose(0, 1) @ features / float(max(1, features.shape[0]))
    # 注释： 对协方差做特征分解；最大特征向量就是残差云的最强方向 u。
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # 注释： 取升序特征值中的最后一列作为最大特征值对应的方向。
    direction = eigenvectors[:, -1]
    # 注释： 最大特征值保留下来用于日志审计，对应旧信号里常看的 lambda_max。
    lambda_max = eigenvalues[-1].clamp_min(0.0)
    # 注释： 计算 E_before=mean((u^T epsilon)^2)，也就是快适应前在最强方向上的残差能量。
    e_before = (features @ direction).pow(2).mean().clamp_min(1e-8)
    # 注释： direction/lambda/E_before 都作为探测参照，不需要参与主模型反向传播。
    return direction.detach(), lambda_max.detach(), e_before.detach()


def _fast_adapt_model(args: argparse.Namespace, model: nn.Module, batches: list[Tensor]) -> None:
    # 注释： 若 abs_fast_lr 未单独设置，则复用正式 learning_rate，减少额外调参负担。
    fast_lr = args.learning_rate if args.abs_fast_lr <= 0 else args.abs_fast_lr
    # 注释： 使用独立 AdamW，避免临时快适应污染正式 optimizer 的动量和二阶矩状态。
    fast_optimizer = AdamW(model.parameters(), lr=fast_lr, weight_decay=args.weight_decay)
    # 注释： 切到训练模式，使快适应语义和正式连续参数训练一致。
    model.train()
    # 注释： 外层循环就是文档中的 n_fast，表示临时连续适应步数。
    for _ in range(max(0, int(args.abs_fast_steps))):
        # 注释： 在近期窗口 batch 上轮流快适应；batch_size>=W 时通常只有当前 batch。
        for batch in batches:
            # 注释： 前向计算临时模型当前重构和后验参数。
            reconstruction, mean, logvar = forward_vae(model, batch)
            # 注释： 快适应只优化普通 VAE 目标 rec+beta_kl*KL，用来测试连续参数能否吸收残差。
            objective = reconstruction_loss(reconstruction, batch) + args.beta_kl * kl_standard_normal(mean, logvar)
            # 注释： 清空临时优化器梯度，避免不同快适应步之间梯度累积。
            fast_optimizer.zero_grad(set_to_none=True)
            # 注释： 对临时目标反向传播；这些梯度只属于探测过程。
            objective.backward()
            # 注释： 复用正式训练的裁剪阈值，防止快适应偶发大梯度破坏探测。
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # 注释： 更新临时参数；外层函数随后会恢复快照。
            fast_optimizer.step()


def absorbability_signal(
    args: argparse.Namespace,
    model: nn.Module,
    projector: ResidualFeatureProjector,
    batches: list[Tensor],
    tau_abs: float,
) -> tuple[Tensor, PWLog]:
    # 注释： 没有 batch 时无法构造残差窗口；返回零信号作为保护。
    if not batches:
        # 注释： 构造与模型设备一致的零张量，避免和 objective 相加时报设备错误。
        zero = next(model.parameters()).new_tensor(0.0)
        # 注释： 返回空日志并把 threshold_mode 标为 absorbability，方便后续审计。
        return zero, PWLog(0.0, 0.0, tau_abs, 0.0, 0, False, threshold_mode="absorbability", tau_abs=tau_abs)
    # 注释： 保存正式模型快照，保证快适应探测不会提交参数改变。
    snapshot = _clone_model_state(model)
    # 注释： 快适应前计算同一窗口残差特征和最强方向 u。
    with torch.no_grad():
        # 注释： before_features 对应文档中的最近 W 个 epsilon_t。
        before_features = _projected_residual_features(model, projector, batches)
        # 注释： 从 C_hat_t 取主方向，并得到 E_before。
        direction, lambda_max, e_before = _top_direction_and_energy(before_features)
    # 注释： 若窗口样本不足，则信号不可靠，直接回滚并返回零。
    if before_features.shape[0] < 2 or float(e_before.item()) <= 0.0:
        # 注释： 恢复模型，保持函数的“探测不提交”约定。
        _restore_model_state(model, snapshot)
        # 注释： 构造零信号。
        zero = next(model.parameters()).new_tensor(0.0)
        # 注释： 日志写入 window_ready=False，说明该 step 没有有效 Xi_abs。
        return zero, PWLog(0.0, 0.0, tau_abs, 0.0, int(before_features.shape[1]), False, threshold_mode="absorbability", tau_abs=tau_abs)
    # 注释： 临时执行 n_fast 步连续适应；这一步用于测试残差方向能否被普通训练压掉。
    _fast_adapt_model(args, model, batches)
    # 注释： 在临时适应后的模型上计算 E_after。
    with torch.no_grad():
        # 注释： 仍然使用同一批样本和同一个 projector，保证前后比较公平。
        after_features = _projected_residual_features(model, projector, batches)
        # 注释： 在快适应前选定的同一方向 u 上投影，得到文档里的 E_after。
        e_after = (after_features @ direction.to(after_features.device)).pow(2).mean().clamp_min(0.0)
        # 注释： 计算 rho=(E_before-E_after)/E_before，表示连续适应压掉残差能量的比例。
        rho = (e_before.to(e_after.device) - e_after) / e_before.to(e_after.device).clamp_min(1e-8)
        # 注释： rho 作为比例写日志时裁剪到 [0,1]；如果快适应变差，记为 0 表示没有吸收。
        rho_clamped = rho.clamp(0.0, 1.0)
        # 注释： 文档公式等价于 Xi_abs=[E_after-tau_abs]_+，表示快适应后仍压不掉的残留能量。
        xi = torch.relu(e_after - e_after.new_tensor(float(tau_abs)))
    # 注释： 恢复正式模型参数，确保快适应只是读数过程。
    _restore_model_state(model, snapshot)
    # 注释： 构造兼容旧 CSV 的日志；xi_pw 列在该模式下表示 Xi_abs。
    log = PWLog(
        lambda_max=float(lambda_max.item()),
        xi_pw=float(xi.item()),
        tau_null=float(tau_abs),
        trace_baseline=0.0,
        feature_dim=int(before_features.shape[1]),
        window_ready=True,
        lambda_edge=0.0,
        spike_rank=0,
        baseline_ready=True,
        baseline_warmup_batches=0,
        threshold_mode="absorbability",
        abs_e_before=float(e_before.item()),
        abs_e_after=float(e_after.item()),
        abs_rho=float(rho_clamped.item()),
        tau_abs=float(tau_abs),
    )
    # 注释： 当前实现把 Xi_abs 作为探测读数返回，不做高阶可微快适应展开。
    return xi.detach(), log


def calibrate_absorbability_on_stage(
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    optimizer: AdamW,
    projector: ResidualFeatureProjector,
    stage: Stage,
    name: str,
    run_dir: Path,
) -> float:
    # 注释： 如果用户手动指定了 tau_abs，就直接使用该值；这用于快速消融或复现实验。
    if args.abs_tau >= 0:
        # 注释： 手动阈值仍写一条极简日志，方便之后知道该 run 没有自动校准。
        append_csv(run_dir / f"calibration_abs_{name}.csv", {"step": 0, "calibration_stage": name, "tau_abs": float(args.abs_tau), "source": "manual"})
        # 注释： 返回手动阈值作为当前阶段的 tau_abs。
        return float(args.abs_tau)
    # 注释： 创建当前校准阶段的数据迭代器；校准数据来自稳定阶段，用来估计正常可吸收残留能量。
    iterator = cycle_loader(stage.loader)
    # 注释： 收集校准期的 E_after；tau_abs 会取这些值的分位数。
    after_values: list[float] = []
    # 注释： 校准日志单独保存，避免和正式训练 train_history 混在一起。
    csv_path = run_dir / f"calibration_abs_{name}.csv"
    # 注释： 总校准步数由 warmup_steps+calibration_steps 决定，沿用旧实验的校准预算。
    total_steps = args.pw_calibration_warmup_steps + args.pw_calibration_steps
    # 注释： 逐步训练校准模型并在 warmup 后执行可吸收性探测。
    for step in range(1, total_steps + 1):
        # 注释： 读取一个校准 batch。
        batch, _labels = next(iterator)
        # 注释： 把 batch 放到训练设备上。
        batch = batch.to(device, non_blocking=True)
        # 注释： 先让校准模型按普通 VAE 目标适应该稳定阶段。
        reconstruction, mean, logvar = forward_vae(model, batch)
        # 注释： 计算重构误差。
        rec = reconstruction_loss(reconstruction, batch)
        # 注释： 计算 VAE KL 项。
        kl = kl_standard_normal(mean, logvar)
        # 注释： 校准期只优化普通 VAE 目标，不加入 Xi。
        objective = rec + args.beta_kl * kl
        # 注释： 清空正式校准优化器梯度。
        optimizer.zero_grad(set_to_none=True)
        # 注释： 反向传播校准目标。
        objective.backward()
        # 注释： 裁剪梯度，保持和正式训练一致。
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # 注释： 更新校准模型参数。
        optimizer.step()
        # 注释： warmup 前不收集 tau_abs 样本，因为模型还没进入相对稳定状态。
        if step <= args.pw_calibration_warmup_steps:
            # 注释： 跳过收集，只继续下一步校准。
            continue
        # 注释： 在当前校准 batch 上做可吸收性探测；tau=0 是为了直接读出 E_after。
        _xi, log = absorbability_signal(args, model, projector, [batch.detach()], tau_abs=0.0)
        # 注释： 把快适应后的残留能量加入分位数样本。
        after_values.append(float(log.abs_e_after))
        # 注释： 写出每一步校准读数，便于之后检查 tau_abs 来源。
        append_csv(
            csv_path,
            {
                "step": step,
                "calibration_stage": name,
                "phase": stage.name,
                "reconstruction": float(rec.detach().item()),
                "kl": float(kl.detach().item()),
                "abs_e_before": log.abs_e_before,
                "abs_e_after": log.abs_e_after,
                "abs_rho": log.abs_rho,
                "tau_abs": "",
                "source": "quantile_sample",
            },
        )
    # 注释： 按文档里的“容差阈值”语义，用稳定校准期 E_after 的分位数作为 tau_abs。
    tau_abs = float(np.quantile(np.asarray(after_values, dtype=np.float64), args.pw_null_quantile)) if after_values else 0.0
    # 注释： 额外写一条 summary 行，明确最终 tau_abs 数值和样本数。
    append_csv(csv_path, {"step": "summary", "calibration_stage": name, "phase": stage.name, "reconstruction": "", "kl": "", "abs_e_before": "", "abs_e_after": "", "abs_rho": "", "tau_abs": tau_abs, "source": f"q{args.pw_null_quantile}"})
    # 注释： 控制台输出校准结果，方便长实验时确认阈值不是 0 或 NaN。
    if not args.quiet_console:
        print(f"[calibrated-abs] {name} tau_abs={tau_abs:.6f} samples={len(after_values)}")
    # 注释： 返回当前阶段的 tau_abs。
    return tau_abs


def residual_predictive_info_signal(args: argparse.Namespace, features: Tensor, labels: Tensor | None) -> tuple[Tensor, PWLog]:
    # 注释： 第三信号需要已知生成因子标签 f；如果当前数据没有 labels，就返回 0，并在 threshold_mode 中明确记录原因。
    if labels is None:
        zero = features.new_tensor(0.0)
        return zero, PWLog(0.0, 0.0, 0.0, 0.0, int(features.shape[1]), True, threshold_mode="residual_predictive_info_no_labels")
    # 注释： ControlledManifoldDataset 的标签列固定为 theta、phi、split、stage；第三信号默认使用 split，因为它对应 local_split 阶段新增的局部自由度。
    factor_to_column = {"theta": 0, "phi": 1, "split": 2, "stage": 3}
    # 注释： 取出用户指定的因子列；这样后续如果要测 theta 或 stage，不必改代码，只改 `--info-factor`。
    factor_column = factor_to_column[str(args.info_factor)]
    # 注释： 把因子标签搬到残差特征所在设备，并转成 long，满足 PyTorch 交叉熵分类目标的输入要求。
    y_raw = labels[:, factor_column].to(device=features.device, dtype=torch.long)
    # 注释： 统计当前 batch 中实际出现的类别；只有一个类别时 H(f)=0，残差无法提供额外类别信息。
    unique = torch.unique(y_raw)
    if unique.numel() < 2:
        zero = features.new_tensor(0.0)
        return zero, PWLog(
            lambda_max=0.0,
            xi_pw=0.0,
            tau_null=0.0,
            trace_baseline=0.0,
            feature_dim=int(features.shape[1]),
            window_ready=True,
            threshold_mode="residual_predictive_info_constant",
            info_entropy=0.0,
            info_cond_entropy=0.0,
            info_accuracy=1.0,
            info_classes=int(unique.numel()),
        )
    # 注释： 将可能不连续的原始类别号重映射到 0..K-1；例如 split 在 local_split 阶段是 1..4，交叉熵需要连续编号。
    remapped = torch.empty_like(y_raw)
    for new_index, old_value in enumerate(unique):
        remapped[y_raw == old_value] = int(new_index)
    # 注释： 第三信号的探针只读残差，不更新主模型；因此 features 必须 detach，符合文档中“分类器独立训练、不回传主模型”的提醒。
    x = features.detach()
    # 注释： 标签也 detach，保证这个小分类问题完全独立于 VAE 的反向传播图。
    y = remapped.detach()
    # 注释： 计算经验熵 H(f)=-sum p log p；它是残差最多能携带多少该因子信息的上界。
    counts = torch.bincount(y, minlength=int(unique.numel())).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs[probs > 0.0] * torch.log(probs[probs > 0.0])).sum()
    # 注释： 构造临时线性分类器 q_phi(f|epsilon)；它是文档里“因子探针”的可计算代理。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(args.seed) + 8_191)
        probe = nn.Linear(x.shape[1], int(unique.numel())).to(device=features.device, dtype=features.dtype)
    # 注释： 使用独立 AdamW 优化器训练探针，避免污染主 VAE optimizer 的动量和权重衰减状态。
    probe_optimizer = AdamW(probe.parameters(), lr=float(args.info_probe_lr), weight_decay=0.0)
    # 注释： 分类交叉熵作为 H(f|epsilon) 的代理；探针越能从残差读出 f，条件熵越低。
    loss_fn = nn.CrossEntropyLoss()
    # 注释： 在当前 batch 上训练少量步数；这是轻量 readout，不是正式下游 probe，所以默认步数比在线 probe 少很多。
    for _ in range(max(1, int(args.info_probe_steps))):
        probe_optimizer.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = loss_fn(logits, y)
        loss.backward()
        probe_optimizer.step()
    # 注释： 训练后重新读出条件熵和准确率；准确率只作审计，Xi_info 使用熵差。
    with torch.no_grad():
        final_logits = probe(x)
        conditional_entropy = loss_fn(final_logits, y)
        accuracy = (final_logits.argmax(dim=1) == y).float().mean()
    # 注释： 互信息下界为 H(f)-H(f|epsilon)；用 ReLU 截断负值，避免探针欠拟合造成“负信息”。
    xi = torch.relu(entropy - conditional_entropy)
    # 注释： 统一写入 PWLog；兼容列 `xi_pw` 在第三信号下表示 Xi_info，`lambda_max/tau_null` 分别复用为 H(f)/H(f|epsilon)。
    log = PWLog(
        lambda_max=float(entropy.detach().item()),
        xi_pw=float(xi.detach().item()),
        tau_null=float(conditional_entropy.detach().item()),
        trace_baseline=0.0,
        feature_dim=int(features.shape[1]),
        window_ready=True,
        threshold_mode="residual_predictive_info",
        info_entropy=float(entropy.detach().item()),
        info_cond_entropy=float(conditional_entropy.detach().item()),
        info_accuracy=float(accuracy.detach().item()),
        info_classes=int(unique.numel()),
    )
    # 注释： 返回 detach 后的 Xi_info；当前实现是文档所说的独立分类器下界，不让 Xi_info 梯度进入主模型。
    return xi.detach(), log


def run_calibration(args: argparse.Namespace, device: torch.device, run_dir: Path) -> dict[str, float]:
    # 注释： 设置 `model, stages, _boundaries`（model, stages, _boundaries）为 `build_experiment(args, device)`；这样后续步骤可以复用该状态而不是重复计算。
    model, stages, _boundaries = build_experiment(args, device)
    # 注释： 设置 `model`（当前训练的 VAE 模型；固定拓扑不做结构编辑）为 `model.to(device)`；这样后续步骤可以复用该状态而不是重复计算。
    model = model.to(device)
    # 注释： 设置 `optimizer`（AdamW 优化器；更新固定拓扑 VAE 参数）为 `AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)`；这样后续步骤可以复用该状态而不是重复计算。
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 注释： 创建残差随机投影器 h，用于把高维残差压到 `args.residual_feature_dim=d_epsilon` 后再计算 Xi。
    # `seed=args.seed + 777` 中的 `args.seed` 是整次实验的基础随机种子；
    # `+777` 是给“残差读出投影矩阵”单独分配的随机流偏移量。
    # 这样正交数据嵌入、样本生成、模型初始化、残差投影不会共用同一段随机序列。
    # 777 本身不是论文参数，也不是需要调的超参；它只是一个固定命名空间编号，
    # 用来保证同一实验 seed 下投影矩阵可复现，同时和其他随机组件区分开。
    projector = ResidualFeatureProjector(args.residual_feature_dim, seed=args.seed + 777)
    # 注释： 设置 `result: dict[str, float]`（result）为 `{}`；这样后续步骤可以复用该状态而不是重复计算。
    result: dict[str, float] = {}

    # 注释： 如果当前选择第二或第三信号，则不需要校准 tau_null/tau_abs；第二信号在线更新快慢 EMA，第三信号直接训练临时因子探针。
    if args.xi_signal in {"fast_slow_divergence", "residual_predictive_info"}:
        # 注释： 直接返回空校准字典；后续 `train_formal` 会创建对应 readout 并把诊断量写入 train_history。
        return result

    # 注释： 检查 `args.xi_signal == "absorbability"`，如果当前实验选择的是新文档第一信号“可吸收性”，校准阶段就不再估计旧的 lambda_max/tau_null，而是估计快适应后残留能量阈值 tau_abs。
    if args.xi_signal == "absorbability":
        # 注释： 检查 `args.pw_baseline_mode == "state_conditioned"`，方案 B 会给每个数据状态分别校准自己的 tau_abs，避免把新状态的正常残差误当成旧状态异常。
        if args.pw_baseline_mode == "state_conditioned":
            # 注释： 遍历 `stages` 中的每个数据流阶段；每个阶段都可以对应一个独立状态 key，例如 controlled 里的 s1、s2/local_split。
            for stage in stages:
                # 注释： 通过 `state_key(args.dataset, stage.name)` 把阶段名映射成校准字典里的 key；没有 key 的阶段会被跳过。
                key = state_key(args.dataset, stage.name)
                # 注释： 只有 key 不为空时才做校准；这样 CIFAR 或 controlled 中不需要单独状态的阶段不会产生无意义阈值。
                if key is not None:
                    # 注释： 写入 `tau_{key}`，其数值来自 `calibrate_absorbability_on_stage`，即校准期 E_after 的指定分位数。
                    result[f"tau_{key}"] = calibrate_absorbability_on_stage(args, device, model, optimizer, projector, stage, key, run_dir)
        # 注释： 如果是 controlled 数据但不是 state_conditioned，则沿用 previous_stage 逻辑：先校准 S1，再让同一个校准模型适应 S2 并校准 S2。
        elif args.dataset == "controlled":
            # 注释： 校准第一阶段圆流形 S1 的 tau_abs，作为第一段和后续 previous-stage fallback 的参考阈值。
            result["tau_s1"] = calibrate_absorbability_on_stage(args, device, model, optimizer, projector, stages[0], "s1", run_dir)
            # 注释： 继续用同一个校准模型在第二阶段 torus 上适应并估计 tau_s2，保持和旧校准流程一致。
            result["tau_s2"] = calibrate_absorbability_on_stage(args, device, model, optimizer, projector, stages[1], "s2", run_dir)
        # 注释： 其他数据集使用 clean 阶段校准 tau_abs；例如 CIFAR 场景下 clean 表示未扰动基准流。
        else:
            # 注释： 写入 `tau_clean`，后续正式训练在 clean/corruption 读数里都可通过 `tau_for_key` 读取。
            result["tau_clean"] = calibrate_absorbability_on_stage(args, device, model, optimizer, projector, stages[0], "clean", run_dir)
        # 注释： 可吸收性校准已经完成，直接返回结果，避免继续进入旧 PW/tau_null 校准分支。
        return result

    # 注释： 检查条件 `args.pw_baseline_mode == "state_conditioned"`；该条件决定是否进入当前实验分支。
    if args.pw_baseline_mode == "state_conditioned":
        # 注释： 遍历 `stage in stages`，用于逐个处理 batch、阶段或 sweep 组合。
        for stage in stages:
            # 注释： 设置 `key`（key）为 `state_key(args.dataset, stage.name)`；这样后续步骤可以复用该状态而不是重复计算。
            key = state_key(args.dataset, stage.name)
            # 注释： 检查条件 `key is not None`；该条件决定是否进入当前实验分支。
            if key is not None:
                # 注释： 设置 `result[f"tau_{key}"]`（result[f"tau_{key}"]）为 `calibrate_on_stage(args, device, model, optimizer, projector, stage, key, run_dir)`；这样后续步骤可以复用该状态而不是重复计算。
                result[f"tau_{key}"] = calibrate_on_stage(args, device, model, optimizer, projector, stage, key, run_dir)
    # 注释： 前一个条件不成立时继续检查 `args.dataset == "controlled"`，用于区分不同数据集/阶段/模式。
    elif args.dataset == "controlled":
        # 注释： 设置 `result["tau_s1"]`（result["tau_s1"]）为 `calibrate_on_stage(args, device, model, optimizer, projector, stages[0], "s1", run_dir)`；这样后续步骤可以复用该状态而不是重复计算。
        result["tau_s1"] = calibrate_on_stage(args, device, model, optimizer, projector, stages[0], "s1", run_dir)
        # Adapt the same calibration model to torus before estimating the S2 null distribution.
        # 注释： 设置 `result["tau_s2"]`（result["tau_s2"]）为 `calibrate_on_stage(args, device, model, optimizer, projector, stages[1], "s2", run_dir)`；这样后续步骤可以复用该状态而不是重复计算。
        result["tau_s2"] = calibrate_on_stage(args, device, model, optimizer, projector, stages[1], "s2", run_dir)
    # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
    else:
        # 注释： 设置 `result["tau_clean"]`（result["tau_clean"]）为 `calibrate_on_stage(args, device, model, optimizer, projector, stages[0], "clean", run_dir)`；这样后续步骤可以复用该状态而不是重复计算。
        result["tau_clean"] = calibrate_on_stage(args, device, model, optimizer, projector, stages[0], "clean", run_dir)
    # 注释： 返回 `result`，把当前计算结果交给调用方继续训练、记录或判断。
    return result


# 注释： 定义函数 `calibrate_on_stage`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def calibrate_on_stage(
    # 注释： 声明字段 `args`，表示 命令行解析后的实验参数；后续训练、校准、probe 都从这里读取配置；dataclass 会把它纳入日志对象。
    args: argparse.Namespace,
    # 注释： 声明字段 `device`，表示 device；dataclass 会把它纳入日志对象。
    device: torch.device,
    # 注释： 声明字段 `model`，表示 当前训练的 VAE 模型；固定拓扑不做结构编辑；dataclass 会把它纳入日志对象。
    model: nn.Module,
    # 注释： 声明字段 `optimizer`，表示 AdamW 优化器；更新固定拓扑 VAE 参数；dataclass 会把它纳入日志对象。
    optimizer: AdamW,
    # 注释： 声明字段 `projector`，表示 残差随机投影器 h；把高维残差压到 d_epsilon 维后计算 Xi；dataclass 会把它纳入日志对象。
    projector: ResidualFeatureProjector,
    # 注释： 声明字段 `stage`，表示 当前数据流阶段对象；提供名称、数据加载器和训练步数；dataclass 会把它纳入日志对象。
    stage: Stage,
    # 注释： 声明字段 `name`，表示 name；dataclass 会把它纳入日志对象。
    name: str,
    # 注释： 声明字段 `run_dir`，表示 单次训练 run 的输出目录；保存 train_history、probe 和 checkpoint；dataclass 会把它纳入日志对象。
    run_dir: Path,
# 注释： 执行 `) -> float:`；这是当前实验流程中的必要操作。
) -> float:
    # 注释： 设置 `iterator`（iterator）为 `cycle_loader(stage.loader)`；这样后续步骤可以复用该状态而不是重复计算。
    iterator = cycle_loader(stage.loader)
    # 注释： 设置 `signal: PrecisionWeightedResidualSignal | None`（signal）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
    signal: PrecisionWeightedResidualSignal | None = None
    # 注释： 设置 `lambda_values: list[float]`（lambda_values）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    lambda_values: list[float] = []
    # 注释： 设置 `total_steps`（total_steps）为 `args.pw_calibration_warmup_steps + args.pw_calibration_steps`；这样后续步骤可以复用该状态而不是重复计算。
    total_steps = args.pw_calibration_warmup_steps + args.pw_calibration_steps
    # 注释： 设置 `csv_path`（csv_path）为 `run_dir / f"calibration_{name}.csv"`；这样后续步骤可以复用该状态而不是重复计算。
    csv_path = run_dir / f"calibration_{name}.csv"
    # 注释： 遍历 `step in range(1, total_steps + 1)`，用于逐个处理 batch、阶段或 sweep 组合。
    for step in range(1, total_steps + 1):
        # 注释： 设置 `batch, _labels`（batch, _labels）为 `next(iterator)`；这样后续步骤可以复用该状态而不是重复计算。
        batch, _labels = next(iterator)
        # 注释： 设置 `batch`（batch）为 `batch.to(device, non_blocking=True)`；这样后续步骤可以复用该状态而不是重复计算。
        batch = batch.to(device, non_blocking=True)
        # 注释： 设置 `reconstruction, mean, logvar`（reconstruction, mean, logvar）为 `forward_vae(model, batch)`；这样后续步骤可以复用该状态而不是重复计算。
        reconstruction, mean, logvar = forward_vae(model, batch)
        # 注释： 设置 `residual`（输入与重构的残差 epsilon；Xi 的原始信号来源）为 `batch - reconstruction`；这样后续步骤可以复用该状态而不是重复计算。
        residual = batch - reconstruction
        # 注释： 设置 `rec`（rec）为 `reconstruction_loss(reconstruction, batch)`；这样后续步骤可以复用该状态而不是重复计算。
        rec = reconstruction_loss(reconstruction, batch)
        # 注释： 设置 `kl`（VAE KL 散度；约束 latent 后验接近先验）为 `kl_standard_normal(mean, logvar)`；这样后续步骤可以复用该状态而不是重复计算。
        kl = kl_standard_normal(mean, logvar)
        # 注释： 设置 `objective`（训练目标函数；loss_plus_xi 时包含 lambda_xi * Xi）为 `rec + args.beta_kl * kl`；这样后续步骤可以复用该状态而不是重复计算。
        objective = rec + args.beta_kl * kl
        # 注释： 设置 `optimizer.zero_grad(set_to_none`（zero_grad(set_to_none）为 `True)`；这样后续步骤可以复用该状态而不是重复计算。
        optimizer.zero_grad(set_to_none=True)
        # 注释： 对 objective 反向传播；loss_plus_xi 下 Xi 会通过当前 batch 残差影响模型。
        objective.backward()
        # 注释： 执行 `nn.utils.clip_grad_norm_(model.parameters(), 5.0)`；这是当前实验流程中的必要操作。
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # 注释： 执行 AdamW 参数更新，使模型根据当前 objective 继续适应数据流。
        optimizer.step()

        # 注释： 打开上下文 `torch.no_grad()`，确保文件句柄或资源在结束后自动释放。
        with torch.no_grad():
            # 注释： 设置 `features`（投影后的残差特征 epsilon_h；Xi 不直接用原始像素残差而用这个特征）为 `projector(residual.detach())`；这样后续步骤可以复用该状态而不是重复计算。
            features = projector(residual.detach())
            # 注释： 检查条件 `signal is None`；该条件决定是否进入当前实验分支。
            if signal is None:
                # 注释： 设置 `signal`（signal）为 `make_signal(args, features.shape[1], tau_null=0.0)`；这样后续步骤可以复用该状态而不是重复计算。
                signal = make_signal(args, features.shape[1], tau_null=0.0)
            # 注释： 执行 `signal.update_baseline(features)`；这是当前实验流程中的必要操作。
            signal.update_baseline(features)
            # 注释： 执行 `signal.update_window(features)`；这是当前实验流程中的必要操作。
            signal.update_window(features)
            # 注释： 设置 `_, log`（_, log）为 `signal.window_signal(features)`；这样后续步骤可以复用该状态而不是重复计算。
            _, log = signal.window_signal(features)
            # 注释： 设置 `window_lmax`（window_lmax）为 `signal.window_lambda_max(device)`；这样后续步骤可以复用该状态而不是重复计算。
            window_lmax = signal.window_lambda_max(device)
            # 注释： 检查条件 `step > args.pw_calibration_warmup_steps`；该条件决定是否进入当前实验分支。
            if step > args.pw_calibration_warmup_steps:
                # 注释： 执行 `lambda_values.append(window_lmax)`；这是当前实验流程中的必要操作。
                lambda_values.append(window_lmax)
            # 注释： 把当前训练、校准或 probe 记录追加到 CSV；这样中途停止也不会丢失已完成结果。
            append_csv(
                # 注释： 在当前多行结构中加入 `csv_path`；多行写法便于审计实验参数。
                csv_path,
                # 注释： 执行 `{`；这是当前实验流程中的必要操作。
                {
                    # 注释： 写入日志字段 `step`（step），用于后续画图、审计和汇总；字段值来自 `step`。
                    "step": step,
                    # 注释： 写入日志字段 `calibration_stage`（calibration_stage），用于后续画图、审计和汇总；字段值来自 `name`。
                    "calibration_stage": name,
                    # 注释： 写入日志字段 `phase`（phase），用于后续画图、审计和汇总；字段值来自 `stage.name`。
                    "phase": stage.name,
                    # 注释： 写入日志字段 `reconstruction`（重构误差；衡量模型对输入的拟合质量），用于后续画图、审计和汇总；字段值来自 `float(rec.detach().item())`。
                    "reconstruction": float(rec.detach().item()),
                    # 注释： 写入日志字段 `kl`（VAE KL 散度；约束 latent 后验接近先验），用于后续画图、审计和汇总；字段值来自 `float(kl.detach().item())`。
                    "kl": float(kl.detach().item()),
                    # 注释： 写入日志字段 `lambda_max_batch`（lambda_max_batch），用于后续画图、审计和汇总；字段值来自 `log.lambda_max`。
                    "lambda_max_batch": log.lambda_max,
                    # 注释： 写入日志字段 `lambda_max_window`（lambda_max_window），用于后续画图、审计和汇总；字段值来自 `window_lmax`。
                    "lambda_max_window": window_lmax,
                    # 注释： 写入日志字段 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移），用于后续画图、审计和汇总；字段值来自 `log.trace_baseline`。
                    "trace_baseline": log.trace_baseline,
                    # 注释： 写入日志字段 `lambda_edge`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向），用于后续画图、审计和汇总；字段值来自 `log.lambda_edge`。
                    "lambda_edge": log.lambda_edge,
                    # 注释： 写入日志字段 `spike_rank`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度），用于后续画图、审计和汇总；字段值来自 `log.spike_rank`。
                    "spike_rank": log.spike_rank,
                    # 注释： 写入日志字段 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照），用于后续画图、审计和汇总；字段值来自 `int(log.baseline_ready)`。
                    "baseline_ready": int(log.baseline_ready),
                    # 注释： 写入日志字段 `window_ready`（残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi），用于后续画图、审计和汇总；字段值来自 `int(log.window_ready)`。
                    "window_ready": int(log.window_ready),
                    # 注释： 写入日志字段 `threshold_mode`（Xi 阈值模式；用于切换经验 tau_null 与理论 MP edge 两种读数），用于后续画图、审计和汇总；字段值来自 `log.threshold_mode`。
                    "threshold_mode": log.threshold_mode,
                # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
                },
            # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
            )
    # 注释： 设置 `tau`（tau）为 `float(np.quantile(np.asarray(lambda_values, dtype=np.float64), args.pw_null_quantile)) if lambda_values else 0.0`；这样后续步骤可以复用该状态而不是重复计算。
    tau = float(np.quantile(np.asarray(lambda_values, dtype=np.float64), args.pw_null_quantile)) if lambda_values else 0.0
    # 注释： 检查条件 `not args.quiet_console`；该条件决定是否进入当前实验分支。
    if not args.quiet_console:
        # 注释： 设置 `print(f"[calibrated] {name} tau_null`（print(f"[calibrated] {name} tau_null）为 `{tau:.6f} samples={len(lambda_values)}")`；这样后续步骤可以复用该状态而不是重复计算。
        print(f"[calibrated] {name} tau_null={tau:.6f} samples={len(lambda_values)}")
    # 注释： 返回 `tau`，把当前计算结果交给调用方继续训练、记录或判断。
    return tau


# 注释： 定义函数 `train_formal`：执行正式固定拓扑训练循环，记录 rec、Xi、spike_rank 和在线 probe。
def train_formal(
    # 注释： 声明字段 `args`，表示 命令行解析后的实验参数；后续训练、校准、probe 都从这里读取配置；dataclass 会把它纳入日志对象。
    args: argparse.Namespace,
    # 注释： 声明字段 `device`，表示 device；dataclass 会把它纳入日志对象。
    device: torch.device,
    # 注释： 声明字段 `run_dir`，表示 单次训练 run 的输出目录；保存 train_history、probe 和 checkpoint；dataclass 会把它纳入日志对象。
    run_dir: Path,
    # 注释： 声明字段 `model`，表示 当前训练的 VAE 模型；固定拓扑不做结构编辑；dataclass 会把它纳入日志对象。
    model: nn.Module,
    # 注释： 声明字段 `stages`，表示 stages；dataclass 会把它纳入日志对象。
    stages: list[Stage],
    # 注释： 声明字段 `phase_boundaries`，表示 phase_boundaries；dataclass 会把它纳入日志对象。
    phase_boundaries: list[int],
    # 注释： 声明字段 `calibration`，表示 校准阶段得到的阈值字典；empirical 模式使用，MP 模式主要保留日志兼容性；dataclass 会把它纳入日志对象。
    calibration: dict[str, float],
# 注释： 执行 `) -> None:`；这是当前实验流程中的必要操作。
) -> None:
    # 注释： 设置 `optimizer`（AdamW 优化器；更新固定拓扑 VAE 参数）为 `AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)`；这样后续步骤可以复用该状态而不是重复计算。
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 注释： 创建正式训练阶段使用的残差随机投影器 h。
    # 它必须和校准阶段使用同样的 seed 规则 `args.seed + 777`，否则校准得到的 tau/null 或 MP 读数
    # 会对应另一个残差特征空间，正式训练时的 Xi 就不可比。
    # `args.seed` 负责实验级复现，`+777` 固定表示“残差投影器这一路随机性”；
    # 真正的投影矩阵还会在 `ResidualFeatureProjector` 内部继续混入 input_dim 和 feature_dim，
    # 因而不同观测维度或不同 d_epsilon 不会误用同一矩阵。
    projector = ResidualFeatureProjector(args.residual_feature_dim, seed=args.seed + 777)
    # 注释： 设置 `signals: dict[str, PrecisionWeightedResidualSignal]`（按状态保存的 PrecisionWeightedResidualSignal 字典；方案 B 为每个状态维护独立 baseline）为 `{}`；这样后续步骤可以复用该状态而不是重复计算。
    signals: dict[str, PrecisionWeightedResidualSignal] = {}
    # 注释： 根据 `pw_window_samples / batch_size` 计算可吸收性信号需要缓存多少个最近 batch；这样 E_before/E_after 使用的样本量和旧 PW 窗口处在同一尺度。
    absorb_window_batches = max(1, math.ceil(int(args.pw_window_samples) / max(1, int(args.batch_size))))
    # 注释： 创建 `absorb_recent_batches` 双端队列，用来保存最近若干个输入 batch；可吸收性信号必须在同一个近期窗口上比较快适应前后的残差能量。
    absorb_recent_batches: deque[Tensor] = deque(maxlen=absorb_window_batches)
    # 注释： 计算快慢分歧信号的快记忆更新率 gamma_f；默认使用 `10 * gamma_s`，对应文档中“gamma_f 远大于 gamma_s”的建议。
    div_gamma_fast = float(args.div_fast_gamma) if float(args.div_fast_gamma) > 0 else 10.0 * float(args.div_slow_gamma)
    # 注释： 创建快慢分歧信号对象；它会在整个数据流上连续维护 C_f/C_s，不在阶段边界重置。
    divergence_signal = FastSlowDivergenceSignal(
        # 注释： 初始 feature_dim 使用命令行 residual_feature_dim；第一次看到真实残差特征后会自动按实际维度校正。
        feature_dim=int(args.residual_feature_dim or 0),
        # 注释： 快 EMA 更新率 gamma_f，控制快记忆对新残差形状的响应速度。
        gamma_fast=div_gamma_fast,
        # 注释： 慢 EMA 更新率 gamma_s，控制慢记忆追踪长期常态的速度。
        gamma_slow=float(args.div_slow_gamma),
        # 注释： 慢协方差 shrinkage 系数；保留参数以兼容已保存的运行命令。
        alpha=float(args.div_alpha),
    )
    # 注释： 设置 `history_path`（history_path）为 `run_dir / "train_history.csv"`；这样后续步骤可以复用该状态而不是重复计算。
    history_path = run_dir / "train_history.csv"
    # 注释： 设置 `probe_history_path`（probe_history_path）为 `run_dir / "probe_history.csv"`；这样后续步骤可以复用该状态而不是重复计算。
    probe_history_path = run_dir / "probe_history.csv"
    # 注释： 设置 `probe_loss_history_path`（probe_loss_history_path）为 `run_dir / "probe_loss_history.csv"`；这样后续步骤可以复用该状态而不是重复计算。
    probe_loss_history_path = run_dir / "probe_loss_history.csv"
    # 注释： 设置 `global_step`（跨阶段累计的全局训练 step；用于对齐阶段边界和曲线横轴）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
    global_step = 0
    # 注释： 设置 `previous_objective: float | None`（上一条记录的 objective；用于计算目标函数变化量 delta_objective）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
    previous_objective: float | None = None

    # 注释： 遍历 `stage_index, stage in enumerate(stages, start=1)`，用于逐个处理 batch、阶段或 sweep 组合。
    for stage_index, stage in enumerate(stages, start=1):
        # 注释： 设置 `iterator`（iterator）为 `cycle_loader(stage.loader)`；这样后续步骤可以复用该状态而不是重复计算。
        iterator = cycle_loader(stage.loader)
        # 注释： 遍历 `step_in_stage in range(1, stage.steps + 1)`，用于逐个处理 batch、阶段或 sweep 组合。
        for step_in_stage in range(1, stage.steps + 1):
            # 注释： 设置 `global_step +`（global_step +）为 `1`；这样后续步骤可以复用该状态而不是重复计算。
            global_step += 1
            # 注释： 设置 `batch, labels` 为当前训练 batch 及其 controlled 因子标签；第三信号需要 labels 里的 split 因子来估计残差预测信息。
            batch, labels = next(iterator)
            # 注释： 设置 `batch`（batch）为 `batch.to(device, non_blocking=True)`；这样后续步骤可以复用该状态而不是重复计算。
            batch = batch.to(device, non_blocking=True)
            # 注释： 设置 `reconstruction, mean, logvar`（reconstruction, mean, logvar）为 `forward_vae(model, batch)`；这样后续步骤可以复用该状态而不是重复计算。
            reconstruction, mean, logvar = forward_vae(model, batch)
            # 注释： 设置 `residual`（输入与重构的残差 epsilon；Xi 的原始信号来源）为 `batch - reconstruction`；这样后续步骤可以复用该状态而不是重复计算。
            residual = batch - reconstruction
            # 注释： 设置 `features`（投影后的残差特征 epsilon_h；Xi 不直接用原始像素残差而用这个特征）为 `projector(residual)`；这样后续步骤可以复用该状态而不是重复计算。
            features = projector(residual)
            # 注释： 设置 `rec`（rec）为 `reconstruction_loss(reconstruction, batch)`；这样后续步骤可以复用该状态而不是重复计算。
            rec = reconstruction_loss(reconstruction, batch)
            # 注释： 设置 `kl`（VAE KL 散度；约束 latent 后验接近先验）为 `kl_standard_normal(mean, logvar)`；这样后续步骤可以复用该状态而不是重复计算。
            kl = kl_standard_normal(mean, logvar)
            # 注释： 设置 `target_signal_key`（当前阶段理论上应该使用的状态 baseline；方案 B 用它区分 s1/t2/local_split）为 `formal_signal_key(args, stage.name)`；这样后续步骤可以复用该状态而不是重复计算。
            target_signal_key = formal_signal_key(args, stage.name)
            # 注释： 设置 `signal_key, baseline_source`（signal_key, baseline_source）为 `select_signal_key_for_readout(args, stage.name, signals, target_signal_key)`；这样后续步骤可以复用该状态而不是重复计算。
            signal_key, baseline_source = select_signal_key_for_readout(args, stage.name, signals, target_signal_key)
            # 注释： 设置 `xi`（当前 step 的 Xi 张量；loss_plus_xi 模式下会进入反向传播）为 `batch.new_tensor(0.0)`；这样后续步骤可以复用该状态而不是重复计算。
            xi = batch.new_tensor(0.0)
            # 注释： 设置 `pw_log`（pw_log）为 `None`；这样后续步骤可以复用该状态而不是重复计算。
            pw_log = None
            legacy_readout_window_key = None
            legacy_readout_window_updated = False
            # 注释： 把历史缓存和当前 batch 组成可吸收性探测窗口；当前 batch 先 `detach`，因为快适应探测只读数据，不需要把输入张量接入主训练计算图。
            absorb_batches = list(absorb_recent_batches) + [batch.detach()]
            # 注释： 检查 `args.xi_signal == "fast_slow_divergence"`；如果启用第二信号，就比较快慢协方差记忆而不是使用阈值窗口。
            if args.xi_signal == "fast_slow_divergence":
                # 注释： 调用 `divergence_signal.update_and_signal(features)`；它会更新 C_f/C_s 并返回 Xi_div=tr(A_s^{-1}C_f)-d+logdet(A_s)-logdet(C_f)。
                xi, pw_log = divergence_signal.update_and_signal(features)
                # 注释： 把 baseline_source 标记为 `fast_slow_ema`，表示该读数来自快慢 EMA 协方差差异。
                baseline_source = "fast_slow_ema"
            # 注释： 检查 `args.xi_signal == "residual_predictive_info"`；如果启用第三信号，就用残差预测 split 因子的互信息下界作为 Xi_info。
            elif args.xi_signal == "residual_predictive_info":
                # 注释： 调用独立因子探针；它训练临时分类器 q(f|epsilon)，返回 Xi_info=max(H(f)-H(f|epsilon),0)，不向主模型回传梯度。
                xi, pw_log = residual_predictive_info_signal(args, features, labels)
                # 注释： 把 baseline_source 标为 `factor_probe`，表示该读数不是 baseline/窗口谱，而是文档第三信号的分类器代理。
                baseline_source = f"factor_probe:{args.info_factor}"
            # 注释： 检查 `args.xi_signal == "absorbability"`，如果启用新文档第一信号，就用“快适应后残留能量”替代旧 PW 谱读数。
            elif args.xi_signal == "absorbability":
                # 注释： 检查 `signal_key is not None` 且达到 `abs_probe_every` 间隔；可吸收性探测要临时训练再回滚，计算较贵，所以允许按间隔采样。
                if signal_key is not None and global_step % max(1, int(args.abs_probe_every)) == 0:
                    # 注释： 从校准结果中读取当前状态对应的 `tau_abs`；字段仍复用 `tau_for_key`，因此兼容 `tau_s1/tau_s2/tau_clean` 命名。
                    tau_abs = tau_for_key(calibration, signal_key)
                    # 注释： 调用 `absorbability_signal` 计算 Xi_abs；内部会保存模型快照、做 n_fast 步连续适应、测 E_after，然后恢复原模型参数。
                    xi, pw_log = absorbability_signal(args, model, projector, absorb_batches, tau_abs)
                    # 注释： 重新执行正式训练前向传播；原因是 `absorbability_signal` 会临时更新再回滚模型参数，必须重新建立干净的 autograd 图，避免 PyTorch 参数版本号不一致。
                    reconstruction, mean, logvar = forward_vae(model, batch)
                    # 注释： 重新计算残差 `batch - reconstruction`，保证后续 baseline/window 更新和日志使用的是回滚后正式模型的残差。
                    residual = batch - reconstruction
                    # 注释： 重新投影残差得到 `features`，确保旧 PW 兼容字段和 CSV 中的 lambda/baseline 信息来自当前正式模型。
                    features = projector(residual)
                    # 注释： 重新计算重构误差；这是正式 objective 的第一项，不能复用快适应 probe 前旧计算图里的张量。
                    rec = reconstruction_loss(reconstruction, batch)
                    # 注释： 重新计算 KL 项；这是正式 objective 的第二项，同样必须属于回滚后的新计算图。
                    kl = kl_standard_normal(mean, logvar)
                    # 注释： 把 baseline_source 标为 `absorbability_probe`，方便后续审计这一步不是旧 baseline 读数，而是快适应探测读数。
                    baseline_source = "absorbability_probe"
                # 注释： 如果本 step 没有执行快适应探测，则写入一个零信号日志；这样 CSV 字段稳定，曲线中未探测点不会沿用旧值造成误解。
                else:
                    # 注释： 根据当前 signal_key 尽量读取对应阈值；没有 key 时阈值为 0，表示当前阶段不产生可吸收性读数。
                    tau_abs = tau_for_key(calibration, signal_key) if signal_key is not None else 0.0
                    # 注释： 构造一个兼容日志对象，`xi_pw=0` 表示本 step 没有实际 probe，`threshold_mode="absorbability_skip"` 用于区分。
                    pw_log = PWLog(0.0, 0.0, tau_abs, 0.0, int(features.shape[1]), False, threshold_mode="absorbability_skip", tau_abs=tau_abs)
            # 注释： 旧路径：如果没有启用可吸收性信号，则继续按原来的 PW/MP 残差窗口信号读取 Xi。
            elif signal_key is not None:
                # 注释： 设置 `signal`（signal）为 `signals.get(signal_key)`；这样后续步骤可以复用该状态而不是重复计算。
                signal = signals.get(signal_key)
                # 注释： 检查条件 `signal is None`；该条件决定是否进入当前实验分支。
                if signal is None:
                    # 注释： 设置 `tau`（tau）为 `tau_for_key(calibration, signal_key)`；这样后续步骤可以复用该状态而不是重复计算。
                    tau = tau_for_key(calibration, signal_key)
                    # 注释： 设置 `signal`（signal）为 `make_signal(args, features.shape[1], tau_null=tau)`；这样后续步骤可以复用该状态而不是重复计算。
                    signal = make_signal(args, features.shape[1], tau_null=tau)
                    # 注释： 设置 `signals[signal_key]`（signals[signal_key]）为 `signal`；这样后续步骤可以复用该状态而不是重复计算。
                    signals[signal_key] = signal
                # Scheme 2 reads the previous-stage baseline on a window that
                # already contains the current batch.  This is different from
                # the later scheme-B read-before-update routing.
                if use_legacy_scheme2_readout(args):
                    xi, pw_log = signal.differentiable_window_signal(features)
                    baseline_source = "legacy_previous_stage"
                    legacy_readout_window_key = signal_key
                # The current routing defines Xi from the already-stored recent
                # whitened-residual window C_hat_t; loss_plus_xi uses a
                # differentiable current-batch readout so Xi can affect theta.
                # 注释： 检查条件 `args.mode == "loss_plus_xi"`；该条件决定是否进入当前实验分支。
                elif args.mode == "loss_plus_xi":
                    # 注释： 设置 `xi, pw_log`（xi, pw_log）为 `signal.differentiable_window_signal(features)`；这样后续步骤可以复用该状态而不是重复计算。
                    xi, pw_log = signal.differentiable_window_signal(features)
                # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
                else:
                    # 注释： 设置 `xi, pw_log`（xi, pw_log）为 `signal.window_signal(features)`；这样后续步骤可以复用该状态而不是重复计算。
                    xi, pw_log = signal.window_signal(features)
            # 注释： 设置 `objective`（训练目标函数；loss_plus_xi 时包含 lambda_xi * Xi）为 `rec + args.beta_kl * kl`；这样后续步骤可以复用该状态而不是重复计算。
            objective = rec + args.beta_kl * kl
            # 注释： 检查条件 `args.mode == "loss_plus_xi"`；该条件决定是否进入当前实验分支。
            if args.mode == "loss_plus_xi":
                # 注释： 设置 `objective`（训练目标函数；loss_plus_xi 时包含 lambda_xi * Xi）为 `objective + args.lambda_xi * xi`；这样后续步骤可以复用该状态而不是重复计算。
                objective = objective + args.lambda_xi * xi
            # 注释： 设置 `optimizer.zero_grad(set_to_none`（zero_grad(set_to_none）为 `True)`；这样后续步骤可以复用该状态而不是重复计算。
            optimizer.zero_grad(set_to_none=True)
            # 注释： 对 objective 反向传播；loss_plus_xi 下 Xi 会通过当前 batch 残差影响模型。
            objective.backward()
            # 注释： 设置 `grad_norm`（grad_norm）为 `float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))`；这样后续步骤可以复用该状态而不是重复计算。
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            # 注释： 执行 AdamW 参数更新，使模型根据当前 objective 继续适应数据流。
            optimizer.step()

            # Update baselines after the parameter update. S1 learns on S1; S2 starts learning during T2.
            # 注释： 打开上下文 `torch.no_grad()`，确保文件句柄或资源在结束后自动释放。
            with torch.no_grad():
                # 注释： 如果当前使用可吸收性信号，则先更新最近 batch 缓存；后面的旧 PW baseline 更新可以继续执行作为兼容性状态，不会参与 Xi_abs 读数。
                if args.xi_signal == "absorbability":
                    # 注释： 保存当前 batch 的 detach 版本，作为后续 E_before/E_after 窗口的一部分；队列满时会自动丢弃最旧 batch。
                    absorb_recent_batches.append(batch.detach())
                    # 注释： 不在这里 `continue`，因为本 step 仍然需要写 train_history、执行在线 probe，并保持旧 CSV 字段完整。
                    pass
                # 注释： 如果当前使用快慢分歧信号或残差预测信息信号，则不需要更新旧 PW baseline；但为了保持日志和 probe 流程，不能跳出训练循环。
                if args.xi_signal in {"fast_slow_divergence", "residual_predictive_info"}:
                    # 注释： 不执行旧 baseline 更新；第二信号已在线更新 C_f/C_s，第三信号只用临时因子探针读当前残差。
                    pass
                # 注释： 如果当前使用第二或第三信号，则跳过旧 PW baseline/window 更新块，避免额外创建无用的 PrecisionWeightedResidualSignal。
                if args.xi_signal in {"fast_slow_divergence", "residual_predictive_info"}:
                    update_keys = []
                    detached_features = features.detach()
                else:
                    # 注释： 设置 `update_keys`（本 step 需要更新 baseline/window 的状态 key；方案 B 中通常是当前阶段对应状态）为 `baseline_update_keys(args, stage.name)`；这样后续步骤可以复用该状态而不是重复计算。
                    update_keys = baseline_update_keys(args, stage.name)
                    # 注释： 设置 `detached_features`（停止梯度后的当前 batch 残差特征；用于更新 baseline/window，避免把日志状态接进计算图）为 `features.detach()`；这样后续步骤可以复用该状态而不是重复计算。
                    detached_features = features.detach()
                if legacy_readout_window_key is not None:
                    readout_signal = signals.get(legacy_readout_window_key)
                    if readout_signal is not None and readout_signal.baseline_ready():
                        readout_signal.update_window(detached_features)
                        legacy_readout_window_updated = True
                # 注释： 遍历 `key in update_keys`，用于逐个处理 batch、阶段或 sweep 组合。
                for key in update_keys:
                    # 注释： 设置 `signal`（signal）为 `signals.get(key)`；这样后续步骤可以复用该状态而不是重复计算。
                    signal = signals.get(key)
                    # 注释： 检查条件 `signal is None`；该条件决定是否进入当前实验分支。
                    if signal is None:
                        # 注释： 设置 `signal`（signal）为 `make_signal(args, detached_features.shape[1], tau_null=tau_for_key(calibration, key))`；这样后续步骤可以复用该状态而不是重复计算。
                        signal = make_signal(args, detached_features.shape[1], tau_null=tau_for_key(calibration, key))
                        # 注释： 设置 `signals[key]`（signals[key]）为 `signal`；这样后续步骤可以复用该状态而不是重复计算。
                        signals[key] = signal
                    # 注释： 执行 `signal.update_baseline(detached_features)`；这是当前实验流程中的必要操作。
                    signal.update_baseline(detached_features)
                    # 注释： 执行 `signal.update_window(detached_features)`；这是当前实验流程中的必要操作。
                    if not (legacy_readout_window_updated and key == legacy_readout_window_key):
                        signal.update_window(detached_features)
                # 注释： 检查条件 `(`；该条件决定是否进入当前实验分支。
                if (
                    # 注释： 设置 `args.pw_baseline_mode`（pw_baseline_mode）为 `= "state_conditioned"`；这样后续步骤可以复用该状态而不是重复计算。
                    args.pw_baseline_mode == "state_conditioned"
                    # 注释： 执行 `and signal_key is not None`；这是当前实验流程中的必要操作。
                    and signal_key is not None
                    # 注释： 执行 `and signal_key not in update_keys`；这是当前实验流程中的必要操作。
                    and signal_key not in update_keys
                # 注释： 执行 `):`；这是当前实验流程中的必要操作。
                ):
                    # Fallback readout: keep the previous state's mu/Sigma fixed,
                    # but feed the new residuals into its window so boundary
                    # anisotropy is visible instead of being skipped during
                    # current-state warmup.
                    # 注释： 设置 `readout_signal`（fallback 读数使用的旧状态 signal；只更新窗口不更新旧状态 mu/Sigma）为 `signals.get(signal_key)`；这样后续步骤可以复用该状态而不是重复计算。
                    readout_signal = signals.get(signal_key)
                    # 注释： 检查条件 `readout_signal is not None and readout_signal.baseline_ready()`；该条件决定是否进入当前实验分支。
                    if readout_signal is not None and readout_signal.baseline_ready():
                        # 注释： 执行 `readout_signal.update_window(detached_features)`；这是当前实验流程中的必要操作。
                        readout_signal.update_window(detached_features)

            # 注释： 检查条件 `global_step % args.log_every == 0 or step_in_stage == 1`；该条件决定是否进入当前实验分支。
            if global_step % args.log_every == 0 or step_in_stage == 1:
                # 注释： 设置 `row`（准备写入 CSV 的单步日志；包含 rec、Xi、lambda_max、spike_rank 和 baseline 来源）为 `{`；这样后续步骤可以复用该状态而不是重复计算。
                row = {
                    # 注释： 写入日志字段 `step`（step），用于后续画图、审计和汇总；字段值来自 `global_step`。
                    "step": global_step,
                    # 注释： 写入日志字段 `stage_index`（当前数据流阶段编号；用于日志区分 s1/t2/local_split），用于后续画图、审计和汇总；字段值来自 `stage_index`。
                    "stage_index": stage_index,
                    # 注释： 写入日志字段 `stage`（当前数据流阶段对象；提供名称、数据加载器和训练步数），用于后续画图、审计和汇总；字段值来自 `stage.name`。
                    "stage": stage.name,
                    # 注释： 写入日志字段 `step_in_stage`（当前阶段内部 step；用于按阶段触发在线 probe），用于后续画图、审计和汇总；字段值来自 `step_in_stage`。
                    "step_in_stage": step_in_stage,
                    # 注释： 写入日志字段 `objective`（训练目标函数；loss_plus_xi 时包含 lambda_xi * Xi），用于后续画图、审计和汇总；字段值来自 `float(objective.detach().item())`。
                    "objective": float(objective.detach().item()),
                    # 注释： 写入日志字段 `reconstruction`（重构误差；衡量模型对输入的拟合质量），用于后续画图、审计和汇总；字段值来自 `float(rec.detach().item())`。
                    "reconstruction": float(rec.detach().item()),
                    # 注释： 写入日志字段 `kl`（VAE KL 散度；约束 latent 后验接近先验），用于后续画图、审计和汇总；字段值来自 `float(kl.detach().item())`。
                    "kl": float(kl.detach().item()),
                    # 注释： 写入日志字段 `xi_pw`（精度加权残差各向异性信号 Xi；用于衡量残差是否集中到少数方向），用于后续画图、审计和汇总；字段值来自 `float(xi.detach().item())`。
                    "xi_pw": float(xi.detach().item()),
                    # 注释： 写入日志字段 `xi_signal`，记录本次运行实际使用 `pw` 还是 `absorbability`；后续四代信号合图时可以避免把不同读数混淆。
                    "xi_signal": args.xi_signal,
                    # 注释： 写入日志字段 `lambda_max_batch`（lambda_max_batch），用于后续画图、审计和汇总；字段值来自 `pw_log.lambda_max if pw_log else 0.0`。
                    "lambda_max_batch": pw_log.lambda_max if pw_log else 0.0,
                    # 注释： 写入日志字段 `tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代），用于后续画图、审计和汇总；字段值来自 `pw_log.tau_null if pw_log else 0.0`。
                    "tau_null": pw_log.tau_null if pw_log else 0.0,
                    # 注释： 写入日志字段 `lambda_edge`（Marchenko-Pastur 右边缘阈值；超过它的谱方向被视为非随机异常方向），用于后续画图、审计和汇总；字段值来自 `pw_log.lambda_edge if pw_log else 0.0`。
                    "lambda_edge": pw_log.lambda_edge if pw_log else 0.0,
                    # 注释： 写入日志字段 `spike_rank`（超过 MP edge 的特征值个数 k_t；表示异常方向数量而不只是最强异常幅度），用于后续画图、审计和汇总；字段值来自 `pw_log.spike_rank if pw_log else 0`。
                    "spike_rank": pw_log.spike_rank if pw_log else 0,
                    # 注释： 写入日志字段 `baseline_ready`（baseline 的 mu_0 和 Sigma_0 是否已经由 warmup 数据初始化；未 ready 时白化没有可信参照），用于后续画图、审计和汇总；字段值来自 `int(pw_log.baseline_ready) if pw_log else 0`。
                    "baseline_ready": int(pw_log.baseline_ready) if pw_log else 0,
                    # 注释： 写入日志字段 `window_ready`（残差窗口是否已满；窗口不满时谱估计不可靠，所以不能输出有效 Xi），用于后续画图、审计和汇总；字段值来自 `int(pw_log.window_ready) if pw_log else 0`。
                    "window_ready": int(pw_log.window_ready) if pw_log else 0,
                    # 注释： 加入字符串 `threshold_mode": pw_log.threshold_mode if pw_log else "`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                    "threshold_mode": pw_log.threshold_mode if pw_log else "",
                    # 注释： 写入日志字段 `trace_baseline`（baseline 协方差矩阵的迹；用于监控正常残差尺度是否漂移），用于后续画图、审计和汇总；字段值来自 `pw_log.trace_baseline if pw_log else 0.0`。
                    "trace_baseline": pw_log.trace_baseline if pw_log else 0.0,
                    # 注释： 写入日志字段 `abs_e_before`，表示快适应前残差沿最强方向 u 的能量 E_before；旧 PW 信号下该值为 0。
                    "abs_e_before": pw_log.abs_e_before if pw_log else 0.0,
                    # 注释： 写入日志字段 `abs_e_after`，表示临时快适应 n_fast 步后同一方向 u 上的残留能量 E_after；Xi_abs 实际就是扣阈值后的该残留。
                    "abs_e_after": pw_log.abs_e_after if pw_log else 0.0,
                    # 注释： 写入日志字段 `abs_rho`，表示可吸收比例 rho=(E_before-E_after)/E_before；越大说明连续参数越能吸收该残差方向。
                    "abs_rho": pw_log.abs_rho if pw_log else 0.0,
                    # 注释： 写入日志字段 `tau_abs`，表示可吸收性信号使用的残留能量阈值；由校准期 E_after 分位数或手动参数给出。
                    "tau_abs": pw_log.tau_abs if pw_log else 0.0,
                    # 注释： 写入日志字段 `div_trace`，表示快慢分歧公式中的迹项 tr(A_s^{-1}C_f)；旧信号下为 0。
                    "div_trace": pw_log.div_trace if pw_log else 0.0,
                    # 注释： 写入日志字段 `div_logdet`，表示快慢分歧公式中的 logdet(A_s)-logdet(C_f)；旧信号下为 0。
                    "div_logdet": pw_log.div_logdet if pw_log else 0.0,
                    # 注释： 写入日志字段 `div_gamma_fast`，记录快记忆 EMA 更新率 gamma_f，便于复现实验。
                    "div_gamma_fast": pw_log.div_gamma_fast if pw_log else 0.0,
                    # 注释： 写入日志字段 `div_gamma_slow`，记录慢记忆 EMA 更新率 gamma_s，便于复现实验。
                    "div_gamma_slow": pw_log.div_gamma_slow if pw_log else 0.0,
                    # 注释： 写入日志字段 `div_trace_fast`，记录快协方差 C_f 的 trace，用于判断快记忆总体能量。
                    "div_trace_fast": pw_log.div_trace_fast if pw_log else 0.0,
                    # 注释： 写入日志字段 `div_trace_slow`，记录慢协方差 C_s 的 trace，用于判断慢记忆是否追上快记忆。
                    "div_trace_slow": pw_log.div_trace_slow if pw_log else 0.0,
                    # 注释： 写入日志字段 `info_entropy`，表示第三信号中的经验熵 H(f)；它是残差最多能携带多少目标因子信息的上界。
                    "info_entropy": pw_log.info_entropy if pw_log else 0.0,
                    # 注释： 写入日志字段 `info_cond_entropy`，表示第三信号分类器估计的 H(f|epsilon)；越低说明残差越能预测目标因子。
                    "info_cond_entropy": pw_log.info_cond_entropy if pw_log else 0.0,
                    # 注释： 写入日志字段 `info_accuracy`，表示第三信号临时分类器在当前 batch 上预测目标因子的准确率，用于审计 Xi_info 是否可信。
                    "info_accuracy": pw_log.info_accuracy if pw_log else 0.0,
                    # 注释： 写入日志字段 `info_classes`，表示第三信号当前 batch 中目标因子的有效类别数；类别少于 2 时 Xi_info 必然为 0。
                    "info_classes": pw_log.info_classes if pw_log else 0,
                    # 注释： 写入日志字段 `grad_norm`（grad_norm），用于后续画图、审计和汇总；字段值来自 `grad_norm`。
                    "grad_norm": grad_norm,
                    # 注释： 写入日志字段 `delta_objective`（delta_objective），用于后续画图、审计和汇总；字段值来自 `"" if previous_objective is None else float(objective.detach().item()) - previous_objective`。
                    "delta_objective": "" if previous_objective is None else float(objective.detach().item()) - previous_objective,
                    # 注释： 写入日志字段 `mode`（当前训练模式；决定 Xi 是否进入训练目标），用于后续画图、审计和汇总；字段值来自 `args.mode`。
                    "mode": args.mode,
                    # 注释： 写入日志字段 `dataset`（当前实验数据集名称；决定使用 controlled 还是 CIFAR 数据流），用于后续画图、审计和汇总；字段值来自 `args.dataset`。
                    "dataset": args.dataset,
                    # 注释： 加入字符串 `target_signal_key": target_signal_key or "`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                    "target_signal_key": target_signal_key or "",
                    # 注释： 加入字符串 `signal_key": signal_key or "`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                    "signal_key": signal_key or "",
                    # 注释： 写入日志字段 `baseline_source`（baseline 来源说明；用于审计本步是 state、fallback 还是 warming），用于后续画图、审计和汇总；字段值来自 `baseline_source`。
                    "baseline_source": baseline_source,
                # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
                }
                # 注释： 设置 `previous_objective`（上一条记录的 objective；用于计算目标函数变化量 delta_objective）为 `float(objective.detach().item())`；这样后续步骤可以复用该状态而不是重复计算。
                previous_objective = float(objective.detach().item())
                # 注释： 把当前训练、校准或 probe 记录追加到 CSV；这样中途停止也不会丢失已完成结果。
                append_csv(history_path, row)
                # 注释： 检查条件 `not args.quiet_console`；该条件决定是否进入当前实验分支。
                if not args.quiet_console:
                    # 注释： 执行 `print(`；这是当前实验流程中的必要操作。
                    print(
                        # 注释： 加入字符串 `[step] {global_step} {stage.name} obj={row['objective']:.4f} rec={row['reconstruction']:.4f} `：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                        f"[step] {global_step} {stage.name} obj={row['objective']:.4f} rec={row['reconstruction']:.4f} "
                        # 注释： 加入字符串 `kl={row['kl']:.4f} xi_pw={row['xi_pw']:.4f} lmax={row['lambda_max_batch']:.4f} `：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                        f"kl={row['kl']:.4f} xi_pw={row['xi_pw']:.4f} lmax={row['lambda_max_batch']:.4f} "
                        # 注释： 加入字符串 `threshold={row['tau_null']:.4f} k={row['spike_rank']} baseline={row['baseline_source']}`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                        f"threshold={row['tau_null']:.4f} k={row['spike_rank']} baseline={row['baseline_source']}",
                        # 注释： 设置 `flush`（flush）为 `True`；这样后续步骤可以复用该状态而不是重复计算。
                        flush=True,
                    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
                    )

            # 注释： 检查条件 `should_run_online_probe(args, step_in_stage)`；该条件决定是否进入当前实验分支。
            if should_run_online_probe(args, step_in_stage):
                # 注释： 设置 `probe_rows`（probe_rows）为 `run_online_controlled_probe(`；这样后续步骤可以复用该状态而不是重复计算。
                probe_rows = run_online_controlled_probe(
                    # 注释： 设置 `args`（命令行解析后的实验参数；后续训练、校准、probe 都从这里读取配置）为 `args`；这样后续步骤可以复用该状态而不是重复计算。
                    args=args,
                    # 注释： 设置 `model`（当前训练的 VAE 模型；固定拓扑不做结构编辑）为 `model`；这样后续步骤可以复用该状态而不是重复计算。
                    model=model,
                    # 注释： 设置 `stage`（当前数据流阶段对象；提供名称、数据加载器和训练步数）为 `stage`；这样后续步骤可以复用该状态而不是重复计算。
                    stage=stage,
                    # 注释： 设置 `global_step`（跨阶段累计的全局训练 step；用于对齐阶段边界和曲线横轴）为 `global_step`；这样后续步骤可以复用该状态而不是重复计算。
                    global_step=global_step,
                    # 注释： 设置 `stage_index`（当前数据流阶段编号；用于日志区分 s1/t2/local_split）为 `stage_index`；这样后续步骤可以复用该状态而不是重复计算。
                    stage_index=stage_index,
                    # 注释： 设置 `step_in_stage`（当前阶段内部 step；用于按阶段触发在线 probe）为 `step_in_stage`；这样后续步骤可以复用该状态而不是重复计算。
                    step_in_stage=step_in_stage,
                    # 注释： 设置 `device`（device）为 `device`；这样后续步骤可以复用该状态而不是重复计算。
                    device=device,
                # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
                )
                # 注释： 遍历 `probe_row in probe_rows`，用于逐个处理 batch、阶段或 sweep 组合。
                for probe_row in probe_rows:
                    # 注释： 设置 `row_type`（row_type）为 `probe_row.pop("row_type")`；这样后续步骤可以复用该状态而不是重复计算。
                    row_type = probe_row.pop("row_type")
                    # 注释： 设置 `append_csv(probe_loss_history_path if row_type`（append_csv(probe_loss_history_path if row_type）为 `= "loss" else probe_history_path, probe_row)`；这样后续步骤可以复用该状态而不是重复计算。
                    append_csv(probe_loss_history_path if row_type == "loss" else probe_history_path, probe_row)

    # 注释： 设置 `summary`（summary）为 `{`；这样后续步骤可以复用该状态而不是重复计算。
    summary = {
        # 注释： 写入日志字段 `steps`（steps），用于后续画图、审计和汇总；字段值来自 `global_step`。
        "steps": global_step,
        # 注释： 写入日志字段 `phase_boundaries`（phase_boundaries），用于后续画图、审计和汇总；字段值来自 `phase_boundaries`。
        "phase_boundaries": phase_boundaries,
        # 注释： 写入日志字段 `calibration`（校准阶段得到的阈值字典；empirical 模式使用，MP 模式主要保留日志兼容性），用于后续画图、审计和汇总；字段值来自 `calibration`。
        "calibration": calibration,
        # 注释： 写入日志字段 `history`（train_history.csv 读出的训练日志；最终 summary 从最后一行提取指标），用于后续画图、审计和汇总；字段值来自 `str(history_path)`。
        "history": str(history_path),
        # 注释： 写入日志字段 `probe_history`（probe_history），用于后续画图、审计和汇总；字段值来自 `str(probe_history_path) if probe_history_path.exists() else None`。
        "probe_history": str(probe_history_path) if probe_history_path.exists() else None,
        # 注释： 写入日志字段 `probe_loss_history`（probe_loss_history），用于后续画图、审计和汇总；字段值来自 `str(probe_loss_history_path) if probe_loss_history_path.exists() else None`。
        "probe_loss_history": str(probe_loss_history_path) if probe_loss_history_path.exists() else None,
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    }
    # 注释： 执行 `write_json(run_dir / "summary.json", summary)`；这是当前实验流程中的必要操作。
    write_json(run_dir / "summary.json", summary)


# 注释： 定义函数 `should_run_online_probe`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def should_run_online_probe(args: argparse.Namespace, step_in_stage: int) -> bool:
    # 注释： 检查条件 `args.dataset != "controlled" or args.online_probe_every <= 0`；该条件决定是否进入当前实验分支。
    if args.dataset != "controlled" or args.online_probe_every <= 0:
        # 注释： 返回 `False`，把当前计算结果交给调用方继续训练、记录或判断。
        return False
    # 注释： 设置 `offset`（offset）为 `max(1, int(args.online_probe_start_offset))`；这样后续步骤可以复用该状态而不是重复计算。
    offset = max(1, int(args.online_probe_start_offset))
    # 注释： 检查条件 `step_in_stage < offset`；该条件决定是否进入当前实验分支。
    if step_in_stage < offset:
        # 注释： 返回 `False`，把当前计算结果交给调用方继续训练、记录或判断。
        return False
    # 注释： 返回 `(step_in_stage - offset) % int(args.online_probe_every) == 0`，把当前计算结果交给调用方继续训练、记录或判断。
    return (step_in_stage - offset) % int(args.online_probe_every) == 0


# 注释： 定义函数 `run_online_controlled_probe`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def run_online_controlled_probe(
    # 注释： 声明字段 `args`，表示 命令行解析后的实验参数；后续训练、校准、probe 都从这里读取配置；dataclass 会把它纳入日志对象。
    args: argparse.Namespace,
    # 注释： 声明字段 `model`，表示 当前训练的 VAE 模型；固定拓扑不做结构编辑；dataclass 会把它纳入日志对象。
    model: nn.Module,
    # 注释： 声明字段 `stage`，表示 当前数据流阶段对象；提供名称、数据加载器和训练步数；dataclass 会把它纳入日志对象。
    stage: Stage,
    # 注释： 声明字段 `global_step`，表示 跨阶段累计的全局训练 step；用于对齐阶段边界和曲线横轴；dataclass 会把它纳入日志对象。
    global_step: int,
    # 注释： 声明字段 `stage_index`，表示 当前数据流阶段编号；用于日志区分 s1/t2/local_split；dataclass 会把它纳入日志对象。
    stage_index: int,
    # 注释： 声明字段 `step_in_stage`，表示 当前阶段内部 step；用于按阶段触发在线 probe；dataclass 会把它纳入日志对象。
    step_in_stage: int,
    # 注释： 声明字段 `device`，表示 device；dataclass 会把它纳入日志对象。
    device: torch.device,
# 注释： 执行 `) -> list[dict[str, object]]:`；这是当前实验流程中的必要操作。
) -> list[dict[str, object]]:
    # 注释： 设置 `previous_training`（previous_training）为 `model.training`；这样后续步骤可以复用该状态而不是重复计算。
    previous_training = model.training
    # 注释： 执行 `model.eval()`；这是当前实验流程中的必要操作。
    model.eval()
    # 注释： 设置 `features: list[Tensor]`（投影后的残差特征 epsilon_h；Xi 不直接用原始像素残差而用这个特征）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    features: list[Tensor] = []
    # 注释： 设置 `labels: list[Tensor]`（labels）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    labels: list[Tensor] = []
    # 注释： 设置 `loader`（loader）为 `DataLoader(`；这样后续步骤可以复用该状态而不是重复计算。
    loader = DataLoader(
        # 注释： 在当前多行结构中加入 `stage.loader.dataset`；多行写法便于审计实验参数。
        stage.loader.dataset,
        # 注释： 设置 `batch_size`（batch_size）为 `args.batch_size`；这样后续步骤可以复用该状态而不是重复计算。
        batch_size=args.batch_size,
        # 注释： 设置 `shuffle`（shuffle）为 `False`；这样后续步骤可以复用该状态而不是重复计算。
        shuffle=False,
        # 注释： 设置 `drop_last`（drop_last）为 `False`；这样后续步骤可以复用该状态而不是重复计算。
        drop_last=False,
        # 注释： 设置 `num_workers`（num_workers）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
        num_workers=0,
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    # 注释： 设置 `seen`（seen）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
    seen = 0
    # 注释： 打开上下文 `torch.no_grad()`，确保文件句柄或资源在结束后自动释放。
    with torch.no_grad():
        # 注释： 遍历 `batch, target in loader`，用于逐个处理 batch、阶段或 sweep 组合。
        for batch, target in loader:
            # 注释： 检查条件 `seen >= args.online_probe_max_samples`；该条件决定是否进入当前实验分支。
            if seen >= args.online_probe_max_samples:
                # 注释： 执行 `break`；这是当前实验流程中的必要操作。
                break
            # 注释： 设置 `take`（take）为 `min(batch.shape[0], args.online_probe_max_samples - seen)`；这样后续步骤可以复用该状态而不是重复计算。
            take = min(batch.shape[0], args.online_probe_max_samples - seen)
            # 注释： 设置 `batch`（batch）为 `batch[:take].to(device, non_blocking=True)`；这样后续步骤可以复用该状态而不是重复计算。
            batch = batch[:take].to(device, non_blocking=True)
            # 注释： 设置 `_reconstruction, mean, _logvar`（_reconstruction, mean, _logvar）为 `forward_vae(model, batch)`；这样后续步骤可以复用该状态而不是重复计算。
            _reconstruction, mean, _logvar = forward_vae(model, batch)
            # 注释： 执行 `features.append(mean.cpu())`；这是当前实验流程中的必要操作。
            features.append(mean.cpu())
            # 注释： 执行 `labels.append(target[:take].cpu())`；这是当前实验流程中的必要操作。
            labels.append(target[:take].cpu())
            # 注释： 设置 `seen +`（seen +）为 `take`；这样后续步骤可以复用该状态而不是重复计算。
            seen += take
    # 注释： 检查条件 `previous_training`；该条件决定是否进入当前实验分支。
    if previous_training:
        # 注释： 执行 `model.train()`；这是当前实验流程中的必要操作。
        model.train()
    # 注释： 检查条件 `not features`；该条件决定是否进入当前实验分支。
    if not features:
        # 注释： 返回 `[]`，把当前计算结果交给调用方继续训练、记录或判断。
        return []
    # 注释： 设置 `x`（x）为 `torch.cat(features, dim=0)`；这样后续步骤可以复用该状态而不是重复计算。
    x = torch.cat(features, dim=0)
    # 注释： 设置 `y`（y）为 `torch.cat(labels, dim=0)`；这样后续步骤可以复用该状态而不是重复计算。
    y = torch.cat(labels, dim=0)
    # 注释： 设置 `all_tasks`（all_tasks）为 `[`；这样后续步骤可以复用该状态而不是重复计算。
    all_tasks = [
        # 注释： 在当前多行结构中加入 `("theta", 0, int(args.factor_bins))`；多行写法便于审计实验参数。
        ("theta", 0, int(args.factor_bins)),
        # 注释： 在当前多行结构中加入 `("phi", 1, int(args.factor_bins) + 1)`；多行写法便于审计实验参数。
        ("phi", 1, int(args.factor_bins) + 1),
        # 注释： 在当前多行结构中加入 `("split", 2, 5)`；多行写法便于审计实验参数。
        ("split", 2, 5),
        # 注释： 在当前多行结构中加入 `("stage", 3, 3)`；多行写法便于审计实验参数。
        ("stage", 3, 3),
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    ]
    # 注释： 设置 `requested_tasks`（requested_tasks）为 `set(args.online_probe_tasks)`；这样后续步骤可以复用该状态而不是重复计算。
    requested_tasks = set(args.online_probe_tasks)
    # 注释： 设置 `tasks`（tasks）为 `[task for task in all_tasks if task[0] in requested_tasks]`；这样后续步骤可以复用该状态而不是重复计算。
    tasks = [task for task in all_tasks if task[0] in requested_tasks]
    # 注释： 设置 `rows: list[dict[str, object]]`（已经收集到的 summary 行；用于断点续跑时保留已有结果）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    rows: list[dict[str, object]] = []
    # 注释： 设置 `accuracies: list[float]`（accuracies）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    accuracies: list[float] = []
    # 注释： 遍历 `task_offset, (task, index, class_count) in enumerate(tasks)`，用于逐个处理 batch、阶段或 sweep 组合。
    for task_offset, (task, index, class_count) in enumerate(tasks):
        # 注释： 设置 `acc, loss_rows`（acc, loss_rows）为 `train_online_linear_probe(`；这样后续步骤可以复用该状态而不是重复计算。
        acc, loss_rows = train_online_linear_probe(
            # 注释： 在当前多行结构中加入 `x`；多行写法便于审计实验参数。
            x,
            # 注释： 在当前多行结构中加入 `y[:, index]`；多行写法便于审计实验参数。
            y[:, index],
            # 注释： 在当前多行结构中加入 `class_count`；多行写法便于审计实验参数。
            class_count,
            # 注释： 设置 `seed`（seed）为 `int(args.seed) + 10_000 + global_step + task_offset`；这样后续步骤可以复用该状态而不是重复计算。
            seed=int(args.seed) + 10_000 + global_step + task_offset,
            # 注释： 设置 `train_steps`（train_steps）为 `int(args.online_probe_train_steps)`；这样后续步骤可以复用该状态而不是重复计算。
            train_steps=int(args.online_probe_train_steps),
            # 注释： 设置 `patience`（patience）为 `int(args.online_probe_early_stopping_patience)`；这样后续步骤可以复用该状态而不是重复计算。
            patience=int(args.online_probe_early_stopping_patience),
            # 注释： 设置 `min_delta`（min_delta）为 `float(args.online_probe_early_stopping_min_delta)`；这样后续步骤可以复用该状态而不是重复计算。
            min_delta=float(args.online_probe_early_stopping_min_delta),
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
        # 注释： 检查条件 `not math.isnan(acc)`；该条件决定是否进入当前实验分支。
        if not math.isnan(acc):
            # 注释： 执行 `accuracies.append(acc)`；这是当前实验流程中的必要操作。
            accuracies.append(acc)
        # 注释： 遍历 `loss_row in loss_rows`，用于逐个处理 batch、阶段或 sweep 组合。
        for loss_row in loss_rows:
            # 注释： 执行 `rows.append(`；这是当前实验流程中的必要操作。
            rows.append(
                # 注释： 执行 `{`；这是当前实验流程中的必要操作。
                {
                    # 注释： 加入字符串 `row_type": "loss`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                    "row_type": "loss",
                    # 注释： 写入日志字段 `step`（step），用于后续画图、审计和汇总；字段值来自 `global_step`。
                    "step": global_step,
                    # 注释： 写入日志字段 `stage_index`（当前数据流阶段编号；用于日志区分 s1/t2/local_split），用于后续画图、审计和汇总；字段值来自 `stage_index`。
                    "stage_index": stage_index,
                    # 注释： 写入日志字段 `stage`（当前数据流阶段对象；提供名称、数据加载器和训练步数），用于后续画图、审计和汇总；字段值来自 `stage.name`。
                    "stage": stage.name,
                    # 注释： 写入日志字段 `step_in_stage`（当前阶段内部 step；用于按阶段触发在线 probe），用于后续画图、审计和汇总；字段值来自 `step_in_stage`。
                    "step_in_stage": step_in_stage,
                    # 注释： 写入日志字段 `task`（task），用于后续画图、审计和汇总；字段值来自 `task`。
                    "task": task,
                    # 注释： 在当前多行结构中加入 `**loss_row`；多行写法便于审计实验参数。
                    **loss_row,
                    # 注释： 写入日志字段 `mode`（当前训练模式；决定 Xi 是否进入训练目标），用于后续画图、审计和汇总；字段值来自 `args.mode`。
                    "mode": args.mode,
                    # 注释： 写入日志字段 `dataset`（当前实验数据集名称；决定使用 controlled 还是 CIFAR 数据流），用于后续画图、审计和汇总；字段值来自 `args.dataset`。
                    "dataset": args.dataset,
                # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
                }
            # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
            )
        # 注释： 执行 `rows.append(`；这是当前实验流程中的必要操作。
        rows.append(
            # 注释： 执行 `{`；这是当前实验流程中的必要操作。
            {
                # 注释： 加入字符串 `row_type": "accuracy`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
                "row_type": "accuracy",
                # 注释： 写入日志字段 `step`（step），用于后续画图、审计和汇总；字段值来自 `global_step`。
                "step": global_step,
                # 注释： 写入日志字段 `stage_index`（当前数据流阶段编号；用于日志区分 s1/t2/local_split），用于后续画图、审计和汇总；字段值来自 `stage_index`。
                "stage_index": stage_index,
                # 注释： 写入日志字段 `stage`（当前数据流阶段对象；提供名称、数据加载器和训练步数），用于后续画图、审计和汇总；字段值来自 `stage.name`。
                "stage": stage.name,
                # 注释： 写入日志字段 `step_in_stage`（当前阶段内部 step；用于按阶段触发在线 probe），用于后续画图、审计和汇总；字段值来自 `step_in_stage`。
                "step_in_stage": step_in_stage,
                # 注释： 写入日志字段 `task`（task），用于后续画图、审计和汇总；字段值来自 `task`。
                "task": task,
                # 注释： 写入日志字段 `accuracy`（accuracy），用于后续画图、审计和汇总；字段值来自 `acc`。
                "accuracy": acc,
                # 注释： 写入日志字段 `mode`（当前训练模式；决定 Xi 是否进入训练目标），用于后续画图、审计和汇总；字段值来自 `args.mode`。
                "mode": args.mode,
                # 注释： 写入日志字段 `dataset`（当前实验数据集名称；决定使用 controlled 还是 CIFAR 数据流），用于后续画图、审计和汇总；字段值来自 `args.dataset`。
                "dataset": args.dataset,
            # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
            }
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
    # 注释： 执行 `rows.append(`；这是当前实验流程中的必要操作。
    rows.append(
        # 注释： 执行 `{`；这是当前实验流程中的必要操作。
        {
            # 注释： 加入字符串 `row_type": "accuracy`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
            "row_type": "accuracy",
            # 注释： 写入日志字段 `step`（step），用于后续画图、审计和汇总；字段值来自 `global_step`。
            "step": global_step,
            # 注释： 写入日志字段 `stage_index`（当前数据流阶段编号；用于日志区分 s1/t2/local_split），用于后续画图、审计和汇总；字段值来自 `stage_index`。
            "stage_index": stage_index,
            # 注释： 写入日志字段 `stage`（当前数据流阶段对象；提供名称、数据加载器和训练步数），用于后续画图、审计和汇总；字段值来自 `stage.name`。
            "stage": stage.name,
            # 注释： 写入日志字段 `step_in_stage`（当前阶段内部 step；用于按阶段触发在线 probe），用于后续画图、审计和汇总；字段值来自 `step_in_stage`。
            "step_in_stage": step_in_stage,
            # 注释： 加入字符串 `task": "mean`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
            "task": "mean",
            # 注释： 写入日志字段 `accuracy`（accuracy），用于后续画图、审计和汇总；字段值来自 `sum(accuracies) / len(accuracies) if accuracies else float("nan")`。
            "accuracy": sum(accuracies) / len(accuracies) if accuracies else float("nan"),
            # 注释： 写入日志字段 `mode`（当前训练模式；决定 Xi 是否进入训练目标），用于后续画图、审计和汇总；字段值来自 `args.mode`。
            "mode": args.mode,
            # 注释： 写入日志字段 `dataset`（当前实验数据集名称；决定使用 controlled 还是 CIFAR 数据流），用于后续画图、审计和汇总；字段值来自 `args.dataset`。
            "dataset": args.dataset,
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        }
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    # 注释： 返回 `rows`，把当前计算结果交给调用方继续训练、记录或判断。
    return rows


# 注释： 定义函数 `train_online_linear_probe`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def train_online_linear_probe(
    # 注释： 声明字段 `x`，表示 x；dataclass 会把它纳入日志对象。
    x: Tensor,
    # 注释： 声明字段 `y`，表示 y；dataclass 会把它纳入日志对象。
    y: Tensor,
    # 注释： 声明字段 `num_classes`，表示 num_classes；dataclass 会把它纳入日志对象。
    num_classes: int,
    # 注释： 声明字段 `seed`，表示 seed；dataclass 会把它纳入日志对象。
    seed: int,
    # 注释： 声明字段 `train_steps`，表示 train_steps；dataclass 会把它纳入日志对象。
    train_steps: int,
    # 注释： 声明字段 `patience`，表示 patience；dataclass 会把它纳入日志对象。
    patience: int,
    # 注释： 声明字段 `min_delta`，表示 min_delta；dataclass 会把它纳入日志对象。
    min_delta: float,
# 注释： 执行 `) -> tuple[float, list[dict[str, float | int]]]:`；这是当前实验流程中的必要操作。
) -> tuple[float, list[dict[str, float | int]]]:
    # 注释： 设置 `x`（x）为 `x.float()`；这样后续步骤可以复用该状态而不是重复计算。
    x = x.float()
    # 注释： 设置 `y`（y）为 `y.long()`；这样后续步骤可以复用该状态而不是重复计算。
    y = y.long()
    # 注释： 检查条件 `x.numel() == 0 or len(torch.unique(y)) < 2`；该条件决定是否进入当前实验分支。
    if x.numel() == 0 or len(torch.unique(y)) < 2:
        # 注释： 返回 `float("nan"), []`，把当前计算结果交给调用方继续训练、记录或判断。
        return float("nan"), []
    # 注释： 设置 `generator`（生成固定随机投影矩阵的随机数发生器；用确定性 seed 保证实验可复现）为 `torch.Generator().manual_seed(seed)`；这样后续步骤可以复用该状态而不是重复计算。
    generator = make_torch_generator(seed)
    # 注释： 设置 `permutation`（permutation）为 `torch.randperm(x.shape[0], generator=generator)`；这样后续步骤可以复用该状态而不是重复计算。
    permutation = torch.randperm(x.shape[0], generator=generator)
    # 注释： 设置 `x`（x）为 `x[permutation]`；这样后续步骤可以复用该状态而不是重复计算。
    x = x[permutation]
    # 注释： 设置 `y`（y）为 `y[permutation]`；这样后续步骤可以复用该状态而不是重复计算。
    y = y[permutation]
    # 注释： 设置 `split`（split）为 `max(1, int(0.8 * x.shape[0]))`；这样后续步骤可以复用该状态而不是重复计算。
    split = max(1, int(0.8 * x.shape[0]))
    # 注释： 设置 `train_x, test_x`（train_x, test_x）为 `x[:split], x[split:]`；这样后续步骤可以复用该状态而不是重复计算。
    train_x, test_x = x[:split], x[split:]
    # 注释： 设置 `train_y, test_y`（train_y, test_y）为 `y[:split], y[split:]`；这样后续步骤可以复用该状态而不是重复计算。
    train_y, test_y = y[:split], y[split:]
    # 注释： 检查条件 `test_x.shape[0] == 0`；该条件决定是否进入当前实验分支。
    if test_x.shape[0] == 0:
        # 注释： 设置 `test_x, test_y`（test_x, test_y）为 `train_x, train_y`；这样后续步骤可以复用该状态而不是重复计算。
        test_x, test_y = train_x, train_y
    # 注释： 设置 `mean`（mean）为 `train_x.mean(dim=0, keepdim=True)`；这样后续步骤可以复用该状态而不是重复计算。
    mean = train_x.mean(dim=0, keepdim=True)
    # 注释： 设置 `std`（std）为 `train_x.std(dim=0, keepdim=True).clamp_min(1e-6)`；这样后续步骤可以复用该状态而不是重复计算。
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    # 注释： 设置 `train_x`（train_x）为 `(train_x - mean) / std`；这样后续步骤可以复用该状态而不是重复计算。
    train_x = (train_x - mean) / std
    # 注释： 设置 `test_x`（test_x）为 `(test_x - mean) / std`；这样后续步骤可以复用该状态而不是重复计算。
    test_x = (test_x - mean) / std
    # 注释： 设置 `probe`（probe）为 `nn.Linear(train_x.shape[1], num_classes)`；这样后续步骤可以复用该状态而不是重复计算。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 17)
        probe = nn.Linear(train_x.shape[1], num_classes)
    # 注释： 设置 `optimizer`（AdamW 优化器；更新固定拓扑 VAE 参数）为 `AdamW(probe.parameters(), lr=5e-3, weight_decay=1e-4)`；这样后续步骤可以复用该状态而不是重复计算。
    optimizer = AdamW(probe.parameters(), lr=5e-3, weight_decay=1e-4)
    # 注释： 设置 `loss_fn`（loss_fn）为 `nn.CrossEntropyLoss()`；这样后续步骤可以复用该状态而不是重复计算。
    loss_fn = nn.CrossEntropyLoss()
    # 注释： 设置 `train_loader`（train_loader）为 `DataLoader(TensorDataset(train_x, train_y), batch_size=min(256, train_x.shape[0]), shuffle=True)`；这样后续步骤可以复用该状态而不是重复计算。
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(256, train_x.shape[0]),
        shuffle=True,
        generator=make_torch_generator(int(seed) + 31),
    )
    # 注释： 设置 `val_loader`（val_loader）为 `DataLoader(TensorDataset(test_x, test_y), batch_size=min(256, test_x.shape[0]), shuffle=False)`；这样后续步骤可以复用该状态而不是重复计算。
    val_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=min(256, test_x.shape[0]), shuffle=False)
    # 注释： 设置 `trainer`（trainer）为 `Engine(lambda _engine, _batch: None)`；这样后续步骤可以复用该状态而不是重复计算。
    trainer = Engine(lambda _engine, _batch: None)
    # 注释： 设置 `evaluator`（evaluator）为 `Engine(lambda _engine, _batch: None)`；这样后续步骤可以复用该状态而不是重复计算。
    evaluator = Engine(lambda _engine, _batch: None)

    # 注释： 定义函数 `score_function`：封装当前脚本中的专用实验逻辑，方便复用和测试。
    def score_function(engine: Engine) -> float:
        # 注释： 返回 `-float(engine.state.metrics["val_loss"])`，把当前计算结果交给调用方继续训练、记录或判断。
        return -float(engine.state.metrics["val_loss"])

    # 注释： 设置 `early_stopping`（early_stopping）为 `EarlyStopping(`；这样后续步骤可以复用该状态而不是重复计算。
    early_stopping = EarlyStopping(
        # 注释： 设置 `patience`（patience）为 `max(1, int(patience))`；这样后续步骤可以复用该状态而不是重复计算。
        patience=max(1, int(patience)),
        # 注释： 设置 `min_delta`（min_delta）为 `float(min_delta)`；这样后续步骤可以复用该状态而不是重复计算。
        min_delta=float(min_delta),
        # 注释： 设置 `score_function`（score_function）为 `score_function`；这样后续步骤可以复用该状态而不是重复计算。
        score_function=score_function,
        # 注释： 设置 `trainer`（trainer）为 `trainer`；这样后续步骤可以复用该状态而不是重复计算。
        trainer=trainer,
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    # 注释： 执行 `evaluator.add_event_handler(Events.COMPLETED, early_stopping)`；这是当前实验流程中的必要操作。
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    # 注释： 设置 `history: list[dict[str, float | int]]`（train_history.csv 读出的训练日志；最终 summary 从最后一行提取指标）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    history: list[dict[str, float | int]] = []
    # 注释： 遍历 `epoch in range(1, max(1, int(train_steps)) + 1)`，用于逐个处理 batch、阶段或 sweep 组合。
    for epoch in range(1, max(1, int(train_steps)) + 1):
        # 注释： 执行 `probe.train()`；这是当前实验流程中的必要操作。
        probe.train()
        # 注释： 设置 `train_loss_total`（train_loss_total）为 `0.0`；这样后续步骤可以复用该状态而不是重复计算。
        train_loss_total = 0.0
        # 注释： 设置 `train_count`（train_count）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
        train_count = 0
        # 注释： 遍历 `batch_x, batch_y in train_loader`，用于逐个处理 batch、阶段或 sweep 组合。
        for batch_x, batch_y in train_loader:
            # 注释： 设置 `logits`（logits）为 `probe(batch_x)`；这样后续步骤可以复用该状态而不是重复计算。
            logits = probe(batch_x)
            # 注释： 设置 `loss`（loss）为 `loss_fn(logits, batch_y)`；这样后续步骤可以复用该状态而不是重复计算。
            loss = loss_fn(logits, batch_y)
            # 注释： 设置 `optimizer.zero_grad(set_to_none`（zero_grad(set_to_none）为 `True)`；这样后续步骤可以复用该状态而不是重复计算。
            optimizer.zero_grad(set_to_none=True)
            # 注释： 执行 `loss.backward()`；这是当前实验流程中的必要操作。
            loss.backward()
            # 注释： 执行 AdamW 参数更新，使模型根据当前 objective 继续适应数据流。
            optimizer.step()
            # 注释： 设置 `train_loss_total +`（train_loss_total +）为 `float(loss.detach().item()) * int(batch_x.shape[0])`；这样后续步骤可以复用该状态而不是重复计算。
            train_loss_total += float(loss.detach().item()) * int(batch_x.shape[0])
            # 注释： 设置 `train_count +`（train_count +）为 `int(batch_x.shape[0])`；这样后续步骤可以复用该状态而不是重复计算。
            train_count += int(batch_x.shape[0])
        # 注释： 执行 `probe.eval()`；这是当前实验流程中的必要操作。
        probe.eval()
        # 注释： 设置 `val_loss_total`（val_loss_total）为 `0.0`；这样后续步骤可以复用该状态而不是重复计算。
        val_loss_total = 0.0
        # 注释： 设置 `val_count`（val_count）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
        val_count = 0
        # 注释： 打开上下文 `torch.no_grad()`，确保文件句柄或资源在结束后自动释放。
        with torch.no_grad():
            # 注释： 遍历 `batch_x, batch_y in val_loader`，用于逐个处理 batch、阶段或 sweep 组合。
            for batch_x, batch_y in val_loader:
                # 注释： 设置 `val_loss`（val_loss）为 `loss_fn(probe(batch_x), batch_y)`；这样后续步骤可以复用该状态而不是重复计算。
                val_loss = loss_fn(probe(batch_x), batch_y)
                # 注释： 设置 `val_loss_total +`（val_loss_total +）为 `float(val_loss.detach().item()) * int(batch_x.shape[0])`；这样后续步骤可以复用该状态而不是重复计算。
                val_loss_total += float(val_loss.detach().item()) * int(batch_x.shape[0])
                # 注释： 设置 `val_count +`（val_count +）为 `int(batch_x.shape[0])`；这样后续步骤可以复用该状态而不是重复计算。
                val_count += int(batch_x.shape[0])
        # 注释： 设置 `train_loss`（train_loss）为 `train_loss_total / max(1, train_count)`；这样后续步骤可以复用该状态而不是重复计算。
        train_loss = train_loss_total / max(1, train_count)
        # 注释： 设置 `val_loss_value`（val_loss_value）为 `val_loss_total / max(1, val_count)`；这样后续步骤可以复用该状态而不是重复计算。
        val_loss_value = val_loss_total / max(1, val_count)
        # 注释： 执行 `history.append({"probe_epoch": epoch, "train_loss": train_loss, "val_loss": val_loss_value})`；这是当前实验流程中的必要操作。
        history.append({"probe_epoch": epoch, "train_loss": train_loss, "val_loss": val_loss_value})
        # 注释： 设置 `evaluator.state.metrics`（metrics）为 `{"val_loss": val_loss_value}`；这样后续步骤可以复用该状态而不是重复计算。
        evaluator.state.metrics = {"val_loss": val_loss_value}
        # 注释： 执行 `evaluator.fire_event(Events.COMPLETED)`；这是当前实验流程中的必要操作。
        evaluator.fire_event(Events.COMPLETED)
        # 注释： 检查条件 `trainer.should_terminate`；该条件决定是否进入当前实验分支。
        if trainer.should_terminate:
            # 注释： 执行 `break`；这是当前实验流程中的必要操作。
            break
    # 注释： 打开上下文 `torch.no_grad()`，确保文件句柄或资源在结束后自动释放。
    with torch.no_grad():
        # 注释： 设置 `predictions`（predictions）为 `probe(test_x).argmax(dim=1)`；这样后续步骤可以复用该状态而不是重复计算。
        predictions = probe(test_x).argmax(dim=1)
    # 注释： 返回 `float((predictions == test_y).float().mean().item()), history`，把当前计算结果交给调用方继续训练、记录或判断。
    return float((predictions == test_y).float().mean().item()), history


# 注释： 定义函数 `formal_signal_key`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def formal_signal_key(args: argparse.Namespace, stage_name: str) -> str | None:
    # 注释： 检查条件 `args.pw_baseline_mode == "state_conditioned"`；该条件决定是否进入当前实验分支。
    if args.pw_baseline_mode == "state_conditioned":
        # 注释： 返回 `state_key(args.dataset, stage_name)`，把当前计算结果交给调用方继续训练、记录或判断。
        return state_key(args.dataset, stage_name)
    # 注释： 检查条件 `args.dataset == "controlled"`；该条件决定是否进入当前实验分支。
    if args.dataset == "controlled":
        # 注释： 检查条件 `stage_name == "s1"`；该条件决定是否进入当前实验分支。
        if stage_name == "s1":
            # 注释： 返回 `"s1"`，把当前计算结果交给调用方继续训练、记录或判断。
            return "s1"
        # 注释： 检查条件 `stage_name == "t2"`；该条件决定是否进入当前实验分支。
        if stage_name == "t2":
            # 注释： 返回 `"s1"`，把当前计算结果交给调用方继续训练、记录或判断。
            return "s1"
        # 注释： 检查条件 `stage_name == "local_split"`；该条件决定是否进入当前实验分支。
        if stage_name == "local_split":
            # 注释： 返回 `"s2"`，把当前计算结果交给调用方继续训练、记录或判断。
            return "s2"
        # 注释： 返回 `None`，把当前计算结果交给调用方继续训练、记录或判断。
        return None
    # 注释： 检查条件 `stage_name == "clean_cifar10"`；该条件决定是否进入当前实验分支。
    if stage_name == "clean_cifar10":
        # 注释： 返回 `"clean"`，把当前计算结果交给调用方继续训练、记录或判断。
        return "clean"
    # 注释： 检查条件 `stage_name != "clean_cifar10"`；该条件决定是否进入当前实验分支。
    if stage_name != "clean_cifar10":
        # 注释： 返回 `"clean"`，把当前计算结果交给调用方继续训练、记录或判断。
        return "clean"
    # 注释： 返回 `None`，把当前计算结果交给调用方继续训练、记录或判断。
    return None


# 注释： 定义函数 `select_signal_key_for_readout`：为方案 B 选择当前 Xi 使用的 state baseline；新状态未 ready 时退回 previous ready baseline。
def select_signal_key_for_readout(
    # 注释： 声明字段 `args`，表示 命令行解析后的实验参数；后续训练、校准、probe 都从这里读取配置；dataclass 会把它纳入日志对象。
    args: argparse.Namespace,
    # 注释： 声明字段 `stage_name`，表示 stage_name；dataclass 会把它纳入日志对象。
    stage_name: str,
    # 注释： 声明字段 `signals`，表示 按状态保存的 PrecisionWeightedResidualSignal 字典；方案 B 为每个状态维护独立 baseline；dataclass 会把它纳入日志对象。
    signals: dict[str, PrecisionWeightedResidualSignal],
    # 注释： 声明字段 `target_key`，表示 target_key；dataclass 会把它纳入日志对象。
    target_key: str | None,
# 注释： 执行 `) -> tuple[str | None, str]:`；这是当前实验流程中的必要操作。
) -> tuple[str | None, str]:
    """Choose which baseline actually computes Xi for the current batch.

    Scheme B uses a current-state baseline, but a new state needs warmup before
    it can emit a window-spectrum signal. During that warmup, use the most
    recent ready previous-state baseline so the transition response is not
    silently replaced by Xi=0.
    """
    # 注释： 检查条件 `args.pw_baseline_mode != "state_conditioned" or target_key is None`；该条件决定是否进入当前实验分支。
    if args.pw_baseline_mode != "state_conditioned" or target_key is None:
        # 注释： 返回 `target_key, "reference" if target_key is not None else "none"`，把当前计算结果交给调用方继续训练、记录或判断。
        return target_key, "reference" if target_key is not None else "none"

    # 注释： 设置 `target_signal`（target_signal）为 `signals.get(target_key)`；这样后续步骤可以复用该状态而不是重复计算。
    target_signal = signals.get(target_key)
    # 注释： 检查条件 `target_signal is not None and target_signal.readout_ready()`；该条件决定是否进入当前实验分支。
    if target_signal is not None and target_signal.readout_ready():
        # 注释： 返回 `target_key, f"state:{target_key}"`，把当前计算结果交给调用方继续训练、记录或判断。
        return target_key, f"state:{target_key}"

    # 注释： 设置 `fallback_key`（fallback_key）为 `fallback_signal_key(args.dataset, stage_name, signals)`；这样后续步骤可以复用该状态而不是重复计算。
    fallback_key = fallback_signal_key(args.dataset, stage_name, signals)
    # 注释： 检查条件 `fallback_key is not None`；该条件决定是否进入当前实验分支。
    if fallback_key is not None:
        # 注释： 返回 `fallback_key, f"fallback:{fallback_key}->target:{target_key}"`，把当前计算结果交给调用方继续训练、记录或判断。
        return fallback_key, f"fallback:{fallback_key}->target:{target_key}"

    # 注释： 返回 `target_key, f"warming:{target_key}"`，把当前计算结果交给调用方继续训练、记录或判断。
    return target_key, f"warming:{target_key}"


# 注释： 定义函数 `fallback_signal_key`：根据当前阶段寻找最近可用的 previous-stage baseline，避免边界处 Xi 被跳过。
def fallback_signal_key(
    # 注释： 声明字段 `dataset`，表示 当前实验数据集名称；决定使用 controlled 还是 CIFAR 数据流；dataclass 会把它纳入日志对象。
    dataset: str,
    # 注释： 声明字段 `stage_name`，表示 stage_name；dataclass 会把它纳入日志对象。
    stage_name: str,
    # 注释： 声明字段 `signals`，表示 按状态保存的 PrecisionWeightedResidualSignal 字典；方案 B 为每个状态维护独立 baseline；dataclass 会把它纳入日志对象。
    signals: dict[str, PrecisionWeightedResidualSignal],
# 注释： 执行 `) -> str | None:`；这是当前实验流程中的必要操作。
) -> str | None:
    """Return the nearest previous ready baseline for Scheme B fallback."""
    # 注释： 检查条件 `dataset == "controlled"`；该条件决定是否进入当前实验分支。
    if dataset == "controlled":
        # 注释： 检查条件 `stage_name == "t2"`；该条件决定是否进入当前实验分支。
        if stage_name == "t2":
            # 注释： 设置 `candidates`（candidates）为 `["s1"]`；这样后续步骤可以复用该状态而不是重复计算。
            candidates = ["s1"]
        # 注释： 前一个条件不成立时继续检查 `stage_name == "local_split"`，用于区分不同数据集/阶段/模式。
        elif stage_name == "local_split":
            # 注释： 设置 `candidates`（candidates）为 `["t2", "s1"]`；这样后续步骤可以复用该状态而不是重复计算。
            candidates = ["t2", "s1"]
        # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
        else:
            # 注释： 设置 `candidates`（candidates）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
            candidates = []
    # 注释： 前一个条件不成立时继续检查 `dataset == "cifar"`，用于区分不同数据集/阶段/模式。
    elif dataset == "cifar":
        # 注释： 设置 `candidates`（candidates）为 `["clean_cifar10"] if stage_name != "clean_cifar10" else []`；这样后续步骤可以复用该状态而不是重复计算。
        candidates = ["clean_cifar10"] if stage_name != "clean_cifar10" else []
    # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
    else:
        # 注释： 设置 `candidates`（candidates）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
        candidates = []

    # 注释： 遍历 `key in candidates`，用于逐个处理 batch、阶段或 sweep 组合。
    for key in candidates:
        # 注释： 设置 `signal`（signal）为 `signals.get(key)`；这样后续步骤可以复用该状态而不是重复计算。
        signal = signals.get(key)
        # 注释： 检查条件 `signal is not None and signal.readout_ready()`；该条件决定是否进入当前实验分支。
        if signal is not None and signal.readout_ready():
            # 注释： 返回 `key`，把当前计算结果交给调用方继续训练、记录或判断。
            return key
    # 注释： 返回 `None`，把当前计算结果交给调用方继续训练、记录或判断。
    return None


# 注释： 定义函数 `baseline_update_keys`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def baseline_update_keys(args: argparse.Namespace, stage_name: str) -> list[str]:
    # 注释： 检查条件 `args.pw_baseline_mode == "state_conditioned"`；该条件决定是否进入当前实验分支。
    if args.pw_baseline_mode == "state_conditioned":
        # 注释： 设置 `key`（key）为 `state_key(args.dataset, stage_name)`；这样后续步骤可以复用该状态而不是重复计算。
        key = state_key(args.dataset, stage_name)
        # 注释： 返回 `[key] if key is not None else []`，把当前计算结果交给调用方继续训练、记录或判断。
        return [key] if key is not None else []
    # 注释： 检查条件 `args.dataset == "controlled"`；该条件决定是否进入当前实验分支。
    if args.dataset == "controlled":
        # 注释： 检查条件 `stage_name == "s1"`；该条件决定是否进入当前实验分支。
        if stage_name == "s1":
            # 注释： 返回 `["s1"]`，把当前计算结果交给调用方继续训练、记录或判断。
            return ["s1"]
        # 注释： 检查条件 `stage_name == "t2"`；该条件决定是否进入当前实验分支。
        if stage_name == "t2":
            # 注释： 返回 `["s2"]`，把当前计算结果交给调用方继续训练、记录或判断。
            return ["s2"]
        # 注释： 返回 `[]`，把当前计算结果交给调用方继续训练、记录或判断。
        return []
    # 注释： 检查条件 `stage_name == "clean_cifar10"`；该条件决定是否进入当前实验分支。
    if stage_name == "clean_cifar10":
        # 注释： 返回 `["clean"]`，把当前计算结果交给调用方继续训练、记录或判断。
        return ["clean"]
    # 注释： 返回 `[]`，把当前计算结果交给调用方继续训练、记录或判断。
    return []


def use_legacy_scheme2_readout(args: argparse.Namespace) -> bool:
    return args.xi_signal == "pw" and args.pw_readout_mode == "legacy_previous_stage"


# 注释： 定义函数 `state_key`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def state_key(dataset: str, stage_name: str) -> str | None:
    """State s_t used by scheme B.

    This implementation uses the stream stage as the state variable.  It is the
    natural state available in both controlled and CIFAR streams, and keeps the
    baseline conditioned on the current regime rather than on a previous-stage
    reference.  Finer states such as class labels can be added later by routing
    individual samples to per-label signals.
    """
    # 注释： 检查条件 `dataset == "controlled"`；该条件决定是否进入当前实验分支。
    if dataset == "controlled":
        # 注释： 返回 `stage_name if stage_name in {"s1", "t2", "local_split"} else None`，把当前计算结果交给调用方继续训练、记录或判断。
        return stage_name if stage_name in {"s1", "t2", "local_split"} else None
    # 注释： 检查条件 `dataset == "cifar"`；该条件决定是否进入当前实验分支。
    if dataset == "cifar":
        # 注释： 返回 `stage_name`，把当前计算结果交给调用方继续训练、记录或判断。
        return stage_name
    # 注释： 返回 `None`，把当前计算结果交给调用方继续训练、记录或判断。
    return None


# 注释： 定义函数 `tau_for_key`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def tau_for_key(calibration: dict[str, float], key: str) -> float:
    # 注释： 返回 `float(calibration.get(f"tau_{key}", 0.0))`，把当前计算结果交给调用方继续训练、记录或判断。
    return float(calibration.get(f"tau_{key}", 0.0))


# 注释： 定义函数 `make_signal`：根据参数创建 PrecisionWeightedResidualSignal，并注入窗口、阈值和 warmup 设置。
def make_signal(args: argparse.Namespace, feature_dim: int, tau_null: float) -> PrecisionWeightedResidualSignal:
    # 注释： 返回 `PrecisionWeightedResidualSignal(`，把当前计算结果交给调用方继续训练、记录或判断。
    return PrecisionWeightedResidualSignal(
        # 注释： 设置 `feature_dim`（残差经过随机投影 h(epsilon) 后的目标维度 d_epsilon；后续 MP edge 和谱协方差都依赖它）为 `feature_dim`；这样后续步骤可以复用该状态而不是重复计算。
        feature_dim=feature_dim,
        # 注释： 设置 `window_samples`（白化残差窗口长度 W；决定谱估计平滑程度，也决定 MP edge 中的 d_epsilon/W）为 `args.pw_window_samples`；这样后续步骤可以复用该状态而不是重复计算。
        window_samples=args.pw_window_samples,
        # 注释： 设置 `gamma`（baseline 均值和协方差的 EMA 更新率；小 gamma 让正常残差基线缓慢适应，避免突变被立刻吸收）为 `args.pw_gamma`；这样后续步骤可以复用该状态而不是重复计算。
        gamma=args.pw_gamma,
        # 注释： 设置 `alpha`（协方差 shrinkage 正则强度；防止高维小样本协方差病态或不可逆）为 `args.pw_alpha`；这样后续步骤可以复用该状态而不是重复计算。
        alpha=args.pw_alpha,
        # 注释： 设置 `tau_null`（空模型阈值；用于扣除有限窗口造成的随机谱虚高，MP 模式下实际被 MP edge 替代）为 `tau_null`；这样后续步骤可以复用该状态而不是重复计算。
        tau_null=tau_null,
        # 注释： 设置 `threshold_mode`（Xi 阈值模式；用于切换经验 tau_null 与理论 MP edge 两种读数）为 `args.pw_threshold_mode`；这样后续步骤可以复用该状态而不是重复计算。
        threshold_mode=args.pw_threshold_mode,
        # 注释： 设置 compact 谱读数类型：edge/mass/gap。训练和校准仍共用同一组残差白化与谱边缘，只在 Xi 公式处切换。
        spectral_readout=args.pw_spectral_readout,
        # 注释： 设置 `baseline_warmup_batches`（初始化 baseline 需要累计的 batch 数；用多批数据避免第一个 batch 偶然性过强）为 `args.pw_baseline_warmup_batches`；这样后续步骤可以复用该状态而不是重复计算。
        baseline_warmup_batches=args.pw_baseline_warmup_batches,
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )


# 注释： 定义函数 `build_experiment`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def build_experiment(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[Stage], list[int]]:
    # 注释： 检查条件 `args.dataset == "controlled"`；该条件决定是否进入当前实验分支。
    if args.dataset == "controlled":
        # 注释： 返回 `build_controlled(args, device)`，把当前计算结果交给调用方继续训练、记录或判断。
        return build_controlled(args, device)
    # 注释： 返回 `build_cifar(args, device)`，把当前计算结果交给调用方继续训练、记录或判断。
    return build_cifar(args, device)


# 注释： 定义函数 `build_controlled`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def build_controlled(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[Stage], list[int]]:
    # 注释： 生成受控流形从内禀空间到观测空间的固定正交嵌入矩阵。
    # `args.seed + 123` 的含义和上面的 `+777` 类似：`args.seed` 保证整次实验可复现，
    # `+123` 单独标记“受控数据正交嵌入”这一随机模块，避免它和样本噪声、残差投影等模块共用随机序列。
    # 这个嵌入在同一次 run 中固定不变，所以阶段变化来自流形结构变化，而不是观测坐标系变化。
    embedding = make_orthogonal_embedding(args.observation_dim, args.intrinsic_dim, args.seed + 123)
    # 注释： 设置 `phases`（phases）为 `["s1", "t2", "local_split"]`；这样后续步骤可以复用该状态而不是重复计算。
    phases = ["s1", "t2", "local_split"]
    # 注释： 设置 `stages: list[Stage]`（stages）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    stages: list[Stage] = []
    # 注释： 设置 `boundaries: list[int]`（boundaries）为 `[]`；这样后续步骤可以复用该状态而不是重复计算。
    boundaries: list[int] = []
    # 注释： 设置 `cumulative`（cumulative）为 `0`；这样后续步骤可以复用该状态而不是重复计算。
    cumulative = 0
    # 注释： 遍历 `offset, (phase, steps) in enumerate(zip(phases, args.controlled_stage_steps, strict=True))`，用于逐个处理 batch、阶段或 sweep 组合。
    for offset, (phase, steps) in enumerate(zip(phases, args.controlled_stage_steps, strict=True)):
        # 注释： 设置 `dataset`（当前实验数据集名称；决定使用 controlled 还是 CIFAR 数据流）为 `ControlledPhaseDataset(`；这样后续步骤可以复用该状态而不是重复计算。
        dataset = ControlledPhaseDataset(
            # 注释： 设置 `phase`（phase）为 `phase`；这样后续步骤可以复用该状态而不是重复计算。
            phase=phase,
            # 注释： 设置 `embedding`（embedding）为 `embedding`；这样后续步骤可以复用该状态而不是重复计算。
            embedding=embedding,
            # 注释： 设置 `size`（size）为 `args.controlled_stage_size`；这样后续步骤可以复用该状态而不是重复计算。
            size=args.controlled_stage_size,
            # 注释： 设置 `noise_std`（noise_std）为 `args.controlled_noise_std`；这样后续步骤可以复用该状态而不是重复计算。
            noise_std=args.controlled_noise_std,
            # 注释： 为每个阶段分配不同但可复现的数据集 seed。
            # `args.seed` 是实验级基础种子；`offset` 是阶段编号，s1/t2/local_split 分别为 0/1/2；
            # `offset * 101` 把不同阶段的数据随机流错开，避免三个阶段在相同 index 下采到完全相同的角度或噪声。
            # 101 是质数，只作为简单偏移系数使用，不代表实验超参数。
            seed=args.seed + offset * 101,
            # 注释： 设置 `factor_bins`（factor_bins）为 `args.factor_bins`；这样后续步骤可以复用该状态而不是重复计算。
            factor_bins=args.factor_bins,
            # 注释： 设置 `split_offset`（split_offset）为 `args.split_offset`；这样后续步骤可以复用该状态而不是重复计算。
            split_offset=args.split_offset,
            # 注释： 设置 `local_probability`（local_probability）为 `args.local_probability`；这样后续步骤可以复用该状态而不是重复计算。
            local_probability=args.local_probability,
            # 注释： 设置 `split_mode`（split_mode）为 `args.split_mode`；这样后续步骤可以复用该状态而不是重复计算。
            split_mode=args.split_mode,
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
        # 注释： 执行 `stages.append(`；这是当前实验流程中的必要操作。
        stages.append(
            # 注释： 执行 `Stage(`；这是当前实验流程中的必要操作。
            Stage(
                # 注释： 设置 `name`（name）为 `phase`；这样后续步骤可以复用该状态而不是重复计算。
                name=phase,
                # 注释： 设置 `steps`（steps）为 `int(steps)`；这样后续步骤可以复用该状态而不是重复计算。
                steps=int(steps),
                # 注释： 设置 `loader`（loader）为 `DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")`；这样后续步骤可以复用该状态而不是重复计算。
                loader=DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0,
                    pin_memory=device.type == "cuda",
                    generator=make_torch_generator(args.seed + offset * 10_007 + 2_001),
                ),
            # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
            )
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        )
        # 注释： 设置 `cumulative +`（cumulative +）为 `int(steps)`；这样后续步骤可以复用该状态而不是重复计算。
        cumulative += int(steps)
        # 注释： 执行 `boundaries.append(cumulative)`；这是当前实验流程中的必要操作。
        boundaries.append(cumulative)
    # 注释： 设置 `model`（当前训练的 VAE 模型；固定拓扑不做结构编辑）为 `MLPVAE(`；这样后续步骤可以复用该状态而不是重复计算。
    model = MLPVAE(
        # 注释： 在当前多行结构中加入 `args.observation_dim`；多行写法便于审计实验参数。
        args.observation_dim,
        # 注释： 在当前多行结构中加入 `args.controlled_latent_dim`；多行写法便于审计实验参数。
        args.controlled_latent_dim,
        # 注释： 在当前多行结构中加入 `args.controlled_hidden_dim`；多行写法便于审计实验参数。
        args.controlled_hidden_dim,
        # 注释： 设置 `hidden_layers`（hidden_layers）为 `args.controlled_hidden_layers`；这样后续步骤可以复用该状态而不是重复计算。
        hidden_layers=args.controlled_hidden_layers,
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    )
    # 注释： 返回 `model, stages, boundaries[:-1]`，把当前计算结果交给调用方继续训练、记录或判断。
    return model, stages, boundaries[:-1]


# 注释： 定义函数 `build_cifar`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def build_cifar(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[Stage], list[int]]:
    # 注释： 设置 `clean`（clean）为 `build_cifar10_dataset(args.cifar10_root, train=True, download=args.download_cifar10)`；这样后续步骤可以复用该状态而不是重复计算。
    clean = build_cifar10_dataset(args.cifar10_root, train=True, download=args.download_cifar10)
    # 注释： 构建 CIFAR-C 混合腐蚀数据流。
    # `seed=args.seed + 404` 用来控制“从不同 corruption 类型/强度中如何抽样”的随机性；
    # `+404` 单独标记 CIFAR-C 采样模块，避免它和模型初始化或受控流形采样共用随机序列。
    # 同一 seed 下，虚拟数据流的抽样顺序可复现；换 seed 时，腐蚀样本组合会随之改变。
    corrupt = MixedCifarCDataset(args.cifarc_root, virtual_size=args.cifar_mixed_size, seed=args.seed + 404)
    # 注释： 设置 `clean_steps, corrupt_steps`（clean_steps, corrupt_steps）为 `[int(v) for v in args.cifar_stage_steps]`；这样后续步骤可以复用该状态而不是重复计算。
    clean_steps, corrupt_steps = [int(v) for v in args.cifar_stage_steps]
    # 注释： 设置 `stages`（stages）为 `[`；这样后续步骤可以复用该状态而不是重复计算。
    stages = [
        # 注释： 执行 `Stage(`；这是当前实验流程中的必要操作。
        Stage(
            # 注释： 加入字符串 `clean_cifar10`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
            "clean_cifar10",
            # 注释： 在当前多行结构中加入 `clean_steps`；多行写法便于审计实验参数。
            clean_steps,
            # 注释： 设置 `DataLoader(clean, batch_size`（DataLoader(clean, batch_size）为 `args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")`；这样后续步骤可以复用该状态而不是重复计算。
            DataLoader(
                clean,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
                generator=make_torch_generator(args.seed + 3_001),
            ),
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        ),
        # 注释： 执行 `Stage(`；这是当前实验流程中的必要操作。
        Stage(
            # 注释： 加入字符串 `full_cifarc`：作为当前列表/命令参数的一项，保持命令参数顺序明确。
            "full_cifarc",
            # 注释： 在当前多行结构中加入 `corrupt_steps`；多行写法便于审计实验参数。
            corrupt_steps,
            # 注释： 设置 `DataLoader(corrupt, batch_size`（DataLoader(corrupt, batch_size）为 `args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")`；这样后续步骤可以复用该状态而不是重复计算。
            DataLoader(
                corrupt,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
                generator=make_torch_generator(args.seed + 3_101),
            ),
        # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
        ),
    # 注释： 结束当前多行结构，保持命令列表、字典或函数调用的边界清晰。
    ]
    # 注释： 检查条件 `args.cifar_model == "strong"`；该条件决定是否进入当前实验分支。
    if args.cifar_model == "strong":
        # 注释： 设置 `model`（当前训练的 VAE 模型；固定拓扑不做结构编辑）为 `StrongConvVAE(args.cifar_latent_dim, hidden_dim=max(args.cifar_hidden_dim, 512), base_channels=args.cifar_base_channels)`；这样后续步骤可以复用该状态而不是重复计算。
        model = StrongConvVAE(args.cifar_latent_dim, hidden_dim=max(args.cifar_hidden_dim, 512), base_channels=args.cifar_base_channels)
    # 注释： 处理前面条件都不满足的默认路径，保证流程有兜底行为。
    else:
        # 注释： 设置 `model`（当前训练的 VAE 模型；固定拓扑不做结构编辑）为 `ConvVAE(args.cifar_latent_dim, hidden_dim=args.cifar_hidden_dim)`；这样后续步骤可以复用该状态而不是重复计算。
        model = ConvVAE(args.cifar_latent_dim, hidden_dim=args.cifar_hidden_dim)
    # 注释： 返回 `model, stages, [clean_steps]`，把当前计算结果交给调用方继续训练、记录或判断。
    return model, stages, [clean_steps]


# 注释： 定义函数 `forward_vae`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def forward_vae(model: nn.Module, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # 注释： 设置 `output`（output）为 `model(batch)`；这样后续步骤可以复用该状态而不是重复计算。
    output = model(batch)
    # 注释： 检查条件 `isinstance(output, tuple)`；该条件决定是否进入当前实验分支。
    if isinstance(output, tuple):
        # 注释： 返回 `output`，把当前计算结果交给调用方继续训练、记录或判断。
        return output
    # 注释： 返回 `output.reconstruction, output.mean, output.logvar`，把当前计算结果交给调用方继续训练、记录或判断。
    return output.reconstruction, output.mean, output.logvar


# 注释： 定义函数 `vars_for_json`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def vars_for_json(args: argparse.Namespace) -> dict[str, object]:
    # 注释： 设置 `result`（result）为 `{}`；这样后续步骤可以复用该状态而不是重复计算。
    result = {}
    # 注释： 遍历 `key, value in vars(args).items()`，用于逐个处理 batch、阶段或 sweep 组合。
    for key, value in vars(args).items():
        # 注释： 设置 `result[key]`（result[key]）为 `str(value) if isinstance(value, Path) else value`；这样后续步骤可以复用该状态而不是重复计算。
        result[key] = str(value) if isinstance(value, Path) else value
    # 注释： 返回 `result`，把当前计算结果交给调用方继续训练、记录或判断。
    return result


# 注释： 定义函数 `plot_curves`：根据 train_history.csv 绘制 objective、lambda_max/tau 和 Xi 曲线。
def plot_curves(run_dir: Path, phase_boundaries: list[int]) -> None:
    # 注释： 设置 `history_path`（history_path）为 `run_dir / "train_history.csv"`；这样后续步骤可以复用该状态而不是重复计算。
    history_path = run_dir / "train_history.csv"
    # 注释： 检查条件 `not history_path.exists()`；该条件决定是否进入当前实验分支。
    if not history_path.exists():
        # 注释： 当前分支已经完成必要处理，直接结束函数避免继续执行。
        return
    # 注释： 设置 `rows`（已经收集到的 summary 行；用于断点续跑时保留已有结果）为 `read_csv(history_path)`；这样后续步骤可以复用该状态而不是重复计算。
    rows = read_csv(history_path)
    # 注释： 检查条件 `not rows`；该条件决定是否进入当前实验分支。
    if not rows:
        # 注释： 当前分支已经完成必要处理，直接结束函数避免继续执行。
        return
    # 注释： 设置 `steps`（steps）为 `[float(r["step"]) for r in rows]`；这样后续步骤可以复用该状态而不是重复计算。
    steps = [float(r["step"]) for r in rows]
    # 注释： 设置 `fig, axes`（fig, axes）为 `plt.subplots(3, 1, figsize=(10, 8), sharex=True)`；这样后续步骤可以复用该状态而不是重复计算。
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # 注释： 设置 `axes[0].plot(steps, [float(r["reconstruction"]) for r in rows], label`（plot(steps, [float(r["reconstruction"]) for r in rows], label）为 `"reconstruction")`；这样后续步骤可以复用该状态而不是重复计算。
    axes[0].plot(steps, [float(r["reconstruction"]) for r in rows], label="reconstruction")
    # 注释： 设置 `axes[0].plot(steps, [float(r["objective"]) for r in rows], label`（plot(steps, [float(r["objective"]) for r in rows], label）为 `"objective")`；这样后续步骤可以复用该状态而不是重复计算。
    axes[0].plot(steps, [float(r["objective"]) for r in rows], label="objective")
    # 注释： 设置 `axes[1].plot(steps, [float(r["lambda_max_batch"]) for r in rows], label`（plot(steps, [float(r["lambda_max_batch"]) for r in rows], label）为 `"lambda_max")`；这样后续步骤可以复用该状态而不是重复计算。
    axes[1].plot(steps, [float(r["lambda_max_batch"]) for r in rows], label="lambda_max")
    # 注释： 设置 `axes[1].plot(steps, [float(r["tau_null"]) for r in rows], label`（plot(steps, [float(r["tau_null"]) for r in rows], label）为 `"tau_null")`；这样后续步骤可以复用该状态而不是重复计算。
    axes[1].plot(steps, [float(r["tau_null"]) for r in rows], label="tau_null")
    # 注释： 读取第一行的 `xi_signal` 字段；旧日志没有该字段时默认显示为 `pw`，保证老结果仍然可以画图。
    xi_signal_name = rows[0].get("xi_signal", "pw")
    # 注释： 根据 `xi_signal_name` 决定曲线图例；兼容列 `xi_pw` 在不同新信号下分别保存 Xi_abs、Xi_div 或 Xi_info。
    xi_label = "Xi_abs" if xi_signal_name == "absorbability" else ("Xi_div" if xi_signal_name == "fast_slow_divergence" else ("Xi_info" if xi_signal_name == "residual_predictive_info" else "Xi_pw"))
    # 注释： 绘制 Xi 曲线；字段仍读取 `xi_pw` 是为了兼容旧报告脚本和历史 CSV。
    axes[2].plot(steps, [float(r["xi_pw"]) for r in rows], label=xi_label)
    # 注释： 遍历 `boundary in phase_boundaries`，用于逐个处理 batch、阶段或 sweep 组合。
    for boundary in phase_boundaries:
        # 注释： 遍历 `ax in axes`，用于逐个处理 batch、阶段或 sweep 组合。
        for ax in axes:
            # 注释： 设置 `ax.axvline(boundary, color`（axvline(boundary, color）为 `"black", linestyle="--", linewidth=1, alpha=0.45)`；这样后续步骤可以复用该状态而不是重复计算。
            ax.axvline(boundary, color="black", linestyle="--", linewidth=1, alpha=0.45)
    # 注释： 遍历 `ax in axes`，用于逐个处理 batch、阶段或 sweep 组合。
    for ax in axes:
        # 注释： 设置 `ax.grid(alpha`（grid(alpha）为 `0.25)`；这样后续步骤可以复用该状态而不是重复计算。
        ax.grid(alpha=0.25)
        # 注释： 绘图操作 `ax.legend()`；用于把训练过程中的 rec、Xi 或谱读数可视化。
        ax.legend()
    # 注释： 绘图操作 `axes[-1].set_xlabel("step")`；用于把训练过程中的 rec、Xi 或谱读数可视化。
    axes[-1].set_xlabel("step")
    # 注释： 绘图操作 `fig.tight_layout()`；用于把训练过程中的 rec、Xi 或谱读数可视化。
    fig.tight_layout()
    # 注释： 设置 `fig.savefig(run_dir / "pw_signal_curves.png", dpi`（png", dpi）为 `180)`；这样后续步骤可以复用该状态而不是重复计算。
    fig.savefig(run_dir / "pw_signal_curves.png", dpi=180)
    # 注释： 绘图操作 `plt.close(fig)`；用于把训练过程中的 rec、Xi 或谱读数可视化。
    plt.close(fig)
    # 注释： 执行 `plot_probe_loss_curves(run_dir, phase_boundaries)`；这是当前实验流程中的必要操作。
    plot_probe_loss_curves(run_dir, phase_boundaries)


# 注释： 定义函数 `plot_probe_loss_curves`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def plot_probe_loss_curves(run_dir: Path, phase_boundaries: list[int]) -> None:
    # 注释： 设置 `probe_loss_path`（probe_loss_path）为 `run_dir / "probe_loss_history.csv"`；这样后续步骤可以复用该状态而不是重复计算。
    probe_loss_path = run_dir / "probe_loss_history.csv"
    # 注释： 检查条件 `not probe_loss_path.exists()`；该条件决定是否进入当前实验分支。
    if not probe_loss_path.exists():
        # 注释： 当前分支已经完成必要处理，直接结束函数避免继续执行。
        return
    # 注释： 设置 `rows`（已经收集到的 summary 行；用于断点续跑时保留已有结果）为 `read_csv(probe_loss_path)`；这样后续步骤可以复用该状态而不是重复计算。
    rows = read_csv(probe_loss_path)
    # 注释： 检查条件 `not rows`；该条件决定是否进入当前实验分支。
    if not rows:
        # 注释： 当前分支已经完成必要处理，直接结束函数避免继续执行。
        return
    # 注释： 设置 `tasks`（tasks）为 `sorted({row["task"] for row in rows})`；这样后续步骤可以复用该状态而不是重复计算。
    tasks = sorted({row["task"] for row in rows})
    # 注释： 设置 `fig, axes`（fig, axes）为 `plt.subplots(len(tasks), 1, figsize=(10, max(3.0, 2.4 * len(tasks))), sharex=True)`；这样后续步骤可以复用该状态而不是重复计算。
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, max(3.0, 2.4 * len(tasks))), sharex=True)
    # 注释： 检查条件 `len(tasks) == 1`；该条件决定是否进入当前实验分支。
    if len(tasks) == 1:
        # 注释： 设置 `axes`（axes）为 `[axes]`；这样后续步骤可以复用该状态而不是重复计算。
        axes = [axes]
    # 注释： 遍历 `ax, task in zip(axes, tasks, strict=True)`，用于逐个处理 batch、阶段或 sweep 组合。
    for ax, task in zip(axes, tasks, strict=True):
        # 注释： 设置 `subset`（subset）为 `[row for row in rows if row["task"] == task]`；这样后续步骤可以复用该状态而不是重复计算。
        subset = [row for row in rows if row["task"] == task]
        # 注释： 设置 `steps`（steps）为 `[float(row["step"]) + 0.02 * float(row["probe_epoch"]) for row in subset]`；这样后续步骤可以复用该状态而不是重复计算。
        steps = [float(row["step"]) + 0.02 * float(row["probe_epoch"]) for row in subset]
        # 注释： 设置 `ax.plot(steps, [float(row["train_loss"]) for row in subset], label`（plot(steps, [float(row["train_loss"]) for row in subset], label）为 `f"{task} train", alpha=0.75)`；这样后续步骤可以复用该状态而不是重复计算。
        ax.plot(steps, [float(row["train_loss"]) for row in subset], label=f"{task} train", alpha=0.75)
        # 注释： 设置 `ax.plot(steps, [float(row["val_loss"]) for row in subset], label`（plot(steps, [float(row["val_loss"]) for row in subset], label）为 `f"{task} val", alpha=0.9)`；这样后续步骤可以复用该状态而不是重复计算。
        ax.plot(steps, [float(row["val_loss"]) for row in subset], label=f"{task} val", alpha=0.9)
        # 注释： 遍历 `boundary in phase_boundaries`，用于逐个处理 batch、阶段或 sweep 组合。
        for boundary in phase_boundaries:
            # 注释： 设置 `ax.axvline(boundary, color`（axvline(boundary, color）为 `"black", linestyle="--", linewidth=1, alpha=0.45)`；这样后续步骤可以复用该状态而不是重复计算。
            ax.axvline(boundary, color="black", linestyle="--", linewidth=1, alpha=0.45)
        # 注释： 绘图操作 `ax.set_ylabel("CE loss")`；用于把训练过程中的 rec、Xi 或谱读数可视化。
        ax.set_ylabel("CE loss")
        # 注释： 设置 `ax.grid(alpha`（grid(alpha）为 `0.25)`；这样后续步骤可以复用该状态而不是重复计算。
        ax.grid(alpha=0.25)
        # 注释： 设置 `ax.legend(fontsize`（legend(fontsize）为 `8)`；这样后续步骤可以复用该状态而不是重复计算。
        ax.legend(fontsize=8)
    # 注释： 绘图操作 `axes[-1].set_xlabel("training step + probe epoch offset")`；用于把训练过程中的 rec、Xi 或谱读数可视化。
    axes[-1].set_xlabel("training step + probe epoch offset")
    # 注释： 绘图操作 `fig.tight_layout()`；用于把训练过程中的 rec、Xi 或谱读数可视化。
    fig.tight_layout()
    # 注释： 设置 `fig.savefig(run_dir / "online_probe_loss_curves.png", dpi`（png", dpi）为 `180)`；这样后续步骤可以复用该状态而不是重复计算。
    fig.savefig(run_dir / "online_probe_loss_curves.png", dpi=180)
    # 注释： 绘图操作 `plt.close(fig)`；用于把训练过程中的 rec、Xi 或谱读数可视化。
    plt.close(fig)


# 注释： 定义函数 `read_csv`：封装当前脚本中的专用实验逻辑，方便复用和测试。
def read_csv(path: Path) -> list[dict[str, str]]:
    # 注释： 打开上下文 `path.open("r", newline="", encoding="utf-8") as handle`，确保文件句柄或资源在结束后自动释放。
    with path.open("r", newline="", encoding="utf-8") as handle:
        # 注释： 返回 `list(csv.DictReader(handle))`，把当前计算结果交给调用方继续训练、记录或判断。
        return list(csv.DictReader(handle))


# 注释： 判断当前文件是否作为脚本直接运行；这样被 import 时不会自动启动实验。
if __name__ == "__main__":
    # 注释： 执行 `main()`；这是当前实验流程中的必要操作。
    main()
