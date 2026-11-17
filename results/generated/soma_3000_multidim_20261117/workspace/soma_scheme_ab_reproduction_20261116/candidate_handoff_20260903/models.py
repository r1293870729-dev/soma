from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    reconstruction: Tensor
    mean: Tensor
    logvar: Tensor
    latent: Tensor


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128 * 4 * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(x)
        return self.mean(features), self.logvar(features).clamp(min=-12.0, max=12.0)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + torch.randn_like(std) * std
        return mean

    def decode(self, z: Tensor) -> Tensor:
        hidden = self.decoder_input(z).view(z.shape[0], 128, 4, 4)
        return self.decoder(hidden)

    def forward(self, x: Tensor) -> VAEOutput:
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latent)
        return VAEOutput(reconstruction=reconstruction, mean=mean, logvar=logvar, latent=latent)


class StrongConvVAE(nn.Module):
    """用于 CIFAR 重构检查的增强版 VAE 骨干。

    它本身不是新的理论贡献；加入它是因为较小的三层 ConvVAE 在 CIFAR
    重构实验中明显欠拟合。
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512, base_channels: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c = int(base_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.GELU(),
            nn.Conv2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(c * 4 * 4 * 4, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c * 4 * 4 * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.ConvTranspose2d(c, c // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, 3, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(x)
        return self.mean(features), self.logvar(features).clamp(min=-12.0, max=12.0)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + torch.randn_like(std) * std
        return mean

    def decode(self, z: Tensor) -> Tensor:
        c4 = self.decoder[0].in_channels
        hidden = self.decoder_input(z).view(z.shape[0], c4, 4, 4)
        return self.decoder(hidden)

    def forward(self, x: Tensor) -> VAEOutput:
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latent)
        return VAEOutput(reconstruction=reconstruction, mean=mean, logvar=logvar, latent=latent)


def reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(reconstruction, target, reduction="mean")


def kl_standard_normal(mean: Tensor, logvar: Tensor) -> Tensor:
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
    return kl_per_sample.mean()
