from __future__ import annotations  # 使用延迟类型注解。

import math  # 用于根据输入尺寸和卷积层数计算展平维度。

import torch  # PyTorch 主库。
from torch import Tensor, nn  # Tensor 用于类型标注，nn 用于搭建网络。
import torch.nn.functional as F  # 用于 BCE/MSE/KL 等损失计算。


class ConvVAE(nn.Module):  # 定义可配置 CNN VAE，适配 dSprites 和 CIFAR。
    def __init__(self, input_shape: tuple[int, int, int], channels: tuple[int, ...], latent_dim: int) -> None:  # 初始化模型。
        super().__init__()  # 初始化 nn.Module 基类。
        self.input_shape = tuple(int(v) for v in input_shape)  # 保存输入形状 C,H,W。
        self.channels = tuple(int(v) for v in channels)  # 保存每层卷积通道数。
        self.latent_dim = int(latent_dim)  # 保存 latent 维度。
        in_channels = self.input_shape[0]  # 当前卷积输入通道数初始为图像通道数。
        encoder_layers: list[nn.Module] = []  # 创建编码器层列表。
        for out_channels in self.channels:  # 逐层构建 stride=2 的下采样卷积。
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))  # 卷积使空间尺寸减半。
            encoder_layers.append(nn.GELU())  # GELU 非线性。
            in_channels = out_channels  # 下一层输入通道等于当前输出通道。
        self.encoder = nn.Sequential(*encoder_layers)  # 打包编码器。
        final_h = self.input_shape[1] // (2 ** len(self.channels))  # 计算编码后高度。
        final_w = self.input_shape[2] // (2 ** len(self.channels))  # 计算编码后宽度。
        if final_h < 1 or final_w < 1:  # 防止网络层数过深导致空间尺寸无效。
            raise ValueError(f"Too many conv layers for input shape {self.input_shape}.")  # 抛出明确错误。
        self.feature_shape = (self.channels[-1], final_h, final_w)  # 保存 encoder 输出特征图形状。
        self.feature_dim = int(math.prod(self.feature_shape))  # 计算展平后的特征维度。
        self.fc_mu = nn.Linear(self.feature_dim, self.latent_dim)  # 输出后验均值。
        self.fc_logvar = nn.Linear(self.feature_dim, self.latent_dim)  # 输出后验 log variance。
        self.fc_decode = nn.Linear(self.latent_dim, self.feature_dim)  # latent 映射回 decoder 初始特征。
        decoder_layers: list[nn.Module] = []  # 创建解码器层列表。
        reversed_channels = list(reversed(self.channels))  # 解码器通道顺序与编码器相反。
        for index, in_ch in enumerate(reversed_channels):  # 逐层构建反卷积。
            out_ch = self.input_shape[0] if index == len(reversed_channels) - 1 else reversed_channels[index + 1]  # 最后一层输出图像通道。
            decoder_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))  # 反卷积使空间尺寸翻倍。
            if index != len(reversed_channels) - 1:  # 最后一层保留 logits，不加非线性。
                decoder_layers.append(nn.GELU())  # 中间层使用 GELU。
        self.decoder = nn.Sequential(*decoder_layers)  # 打包解码器。

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:  # 编码输入图像，返回后验均值和 logvar。
        features = self.encoder(x)  # 通过 CNN encoder。
        flat = features.flatten(start_dim=1)  # 展平为 batch x feature_dim。
        mu = self.fc_mu(flat)  # 计算后验均值。
        logvar = self.fc_logvar(flat).clamp(min=-12.0, max=12.0)  # 计算并裁剪 logvar，避免数值爆炸。
        return mu, logvar  # 返回后验参数。

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:  # VAE 重参数采样。
        std = torch.exp(0.5 * logvar)  # 从 log variance 得到标准差。
        eps = torch.randn_like(std)  # 采样标准正态噪声。
        return mu + eps * std  # 返回 z = mu + sigma * eps。

    def decode(self, z: Tensor) -> Tensor:  # 从 latent 解码到图像 logits。
        flat = self.fc_decode(z)  # latent 映射到展平特征。
        features = flat.view(z.shape[0], *self.feature_shape)  # reshape 回特征图。
        return self.decoder(features)  # 输出重构 logits。

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # 完整 VAE 前向。
        mu, logvar = self.encode(x)  # 得到后验参数。
        z = self.reparameterize(mu, logvar)  # 采样 latent。
        reconstruction_logits = self.decode(z)  # 解码得到重构 logits。
        return reconstruction_logits, mu, logvar, z  # 返回重构和后验信息。


def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:  # 计算 q(z|x) 到标准正态先验的 KL。
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # 每个样本的 KL。
    return kl_per_sample.mean()  # 对 batch 求均值。


def reconstruction_loss(logits: Tensor, target: Tensor, likelihood: str) -> tuple[Tensor, Tensor]:  # 计算训练用重构项和像素 MSE 诊断。
    if likelihood == "bernoulli":  # dSprites 使用 Bernoulli/BCE。
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")  # BCEWithLogitsLoss。
        reconstruction = torch.sigmoid(logits)  # 将 logits 转为概率图，供诊断和残差使用。
    elif likelihood == "gaussian":  # CIFAR 使用 Gaussian/MSE。
        reconstruction = torch.sigmoid(logits)  # 将输出限制到 0-1 图像范围。
        loss = F.mse_loss(reconstruction, target, reduction="mean")  # 使用均方误差。
    else:  # 未知 likelihood 直接报错。
        raise ValueError(f"Unknown likelihood: {likelihood}")  # 给出明确错误。
    mse = F.mse_loss(reconstruction, target, reduction="mean")  # 统一记录 MSE 诊断。
    return loss, mse  # 返回优化重构项和 MSE 指标。
