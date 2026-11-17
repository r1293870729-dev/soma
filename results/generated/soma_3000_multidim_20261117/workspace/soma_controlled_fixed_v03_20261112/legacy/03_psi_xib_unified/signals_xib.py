from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from utils import make_generator


@dataclass(frozen=True)
class XiBLog:
    """One diagnostic readout of the Xi-B residual spectral mass."""

    xi_b: float
    lambda_edge: float
    spike_rank: int
    window_count: int
    baseline_ready: bool
    readout_source: str


class ResidualProjector:
    """Fixed random projection h(epsilon) from pixel residuals to d_epsilon."""

    def __init__(self, feature_dim: int, seed: int) -> None:
        # Keep only the target feature dimension here; the input dimension is known
        # only after the first residual batch arrives.
        self.feature_dim = int(feature_dim)
        # The seed makes the projection reproducible across machines.
        self.seed = int(seed)
        # The matrix is created lazily so the same class supports 8x8 vectors,
        # 64x64 images, and CIFAR images without separate code paths.
        self.matrix: Tensor | None = None

    def __call__(self, residual: Tensor) -> Tensor:
        # Flatten each residual image/vector into one row.
        flat = residual.flatten(start_dim=1)
        input_dim = int(flat.shape[1])
        if self.matrix is None or self.matrix.shape[0] != input_dim:
            # Mix input_dim and feature_dim into the seed so different observation
            # shapes do not accidentally reuse incompatible random projections.
            generator = make_generator(self.seed + input_dim * 1009 + self.feature_dim)
            matrix = torch.randn(input_dim, self.feature_dim, generator=generator)
            # Scale by sqrt(d_epsilon) to keep projected magnitudes comparable.
            self.matrix = matrix / (float(self.feature_dim) ** 0.5)
        return flat @ self.matrix.to(device=flat.device, dtype=flat.dtype)


class StageBaseline:
    """EMA baseline for one data-stream stage."""

    def __init__(self, feature_dim: int, gamma: float, alpha: float, warmup_batches: int, epsilon: float) -> None:
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.warmup_batches = int(warmup_batches)
        self.epsilon = float(epsilon)
        self.batch_count = 0
        self.mu: Tensor | None = None
        self.cov: Tensor | None = None

    def ready(self) -> bool:
        return self.batch_count >= self.warmup_batches and self.mu is not None and self.cov is not None

    def update(self, features: Tensor) -> None:
        # Baseline statistics are diagnostic state, not trainable model state.
        detached = features.detach()
        batch_mu = detached.mean(dim=0)
        centered = detached - batch_mu
        denom = max(1, int(detached.shape[0]) - 1)
        batch_cov = centered.transpose(0, 1) @ centered / float(denom)
        identity = torch.eye(self.feature_dim, device=features.device, dtype=features.dtype)
        trace = torch.trace(batch_cov).clamp_min(self.epsilon)
        isotropic_target = trace / float(self.feature_dim) * identity
        # Shrinkage prevents unstable inverse square roots when the batch
        # covariance has tiny or noisy eigenvalues.
        batch_cov = (1.0 - self.alpha) * batch_cov + self.alpha * isotropic_target
        batch_cov = batch_cov + self.epsilon * identity
        if self.mu is None or self.cov is None:
            self.mu = batch_mu
            self.cov = batch_cov
        else:
            self.mu = (1.0 - self.gamma) * self.mu + self.gamma * batch_mu
            self.cov = (1.0 - self.gamma) * self.cov + self.gamma * batch_cov
        self.batch_count += 1

    def whiten(self, features: Tensor) -> Tensor:
        if self.mu is None or self.cov is None:
            raise RuntimeError("Baseline is not initialized.")
        cov = self.cov.to(device=features.device, dtype=features.dtype)
        mu = self.mu.to(device=features.device, dtype=features.dtype)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.clamp_min(self.epsilon)
        invsqrt = eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.transpose(0, 1)
        return (features - mu) @ invsqrt.transpose(0, 1)


class XiBSignal:
    """Precision-weighted residual spectral mass, shared by all datasets."""

    def __init__(
        self,
        feature_dim: int,
        window_samples: int,
        gamma: float,
        alpha: float,
        warmup_batches: int,
        seed: int,
        epsilon: float = 1e-5,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.window_samples = int(window_samples)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.warmup_batches = int(warmup_batches)
        self.epsilon = float(epsilon)
        self.projector = ResidualProjector(feature_dim=self.feature_dim, seed=int(seed))
        self.baselines: dict[str, StageBaseline] = {}
        self.last_ready_stage: str | None = None
        self.window: deque[Tensor] = deque(maxlen=self.window_samples)

    def _baseline(self, stage_name: str) -> StageBaseline:
        if stage_name not in self.baselines:
            self.baselines[stage_name] = StageBaseline(
                self.feature_dim,
                self.gamma,
                self.alpha,
                self.warmup_batches,
                self.epsilon,
            )
        return self.baselines[stage_name]

    def update_and_measure(self, residual: Tensor, stage_name: str) -> XiBLog:
        # Use the same residual feature definition for every dataset.
        features = self.projector(residual)
        current = self._baseline(stage_name)
        # New stages use the most recent ready baseline until their own baseline
        # has warmed up, matching the fallback behavior used in recent Xi-B runs.
        readout_stage = stage_name if current.ready() else self.last_ready_stage
        readout = self.baselines.get(readout_stage) if readout_stage is not None else None
        log = self._measure(features, readout, readout_stage or "none")
        # Update after measuring so the current batch does not contaminate its
        # own readout.
        current.update(features)
        if current.ready():
            self.last_ready_stage = stage_name
        return log

    def _measure(self, features: Tensor, baseline: StageBaseline | None, source: str) -> XiBLog:
        edge_value = self.mp_edge()
        if baseline is None or not baseline.ready():
            return XiBLog(0.0, edge_value, 0, len(self.window), False, source)
        eta = baseline.whiten(features).detach()
        for row in eta:
            self.window.append(row.detach().cpu())
        if len(self.window) < self.window_samples:
            return XiBLog(0.0, edge_value, 0, len(self.window), True, source)
        matrix = torch.stack(list(self.window), dim=0).to(device=features.device, dtype=features.dtype)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        cov = matrix.transpose(0, 1) @ matrix / float(max(1, matrix.shape[0] - 1))
        trace = torch.trace(cov).clamp_min(self.epsilon)
        # Trace normalization makes Xi-B a shape signal rather than residual
        # energy itself.
        cov = cov * (float(self.feature_dim) / trace)
        eigenvalues = torch.linalg.eigvalsh(cov).flip(0)
        edge = torch.tensor(edge_value, device=features.device, dtype=features.dtype)
        excess = torch.clamp(eigenvalues / edge - 1.0, min=0.0)
        xi_b = excess.sum()
        spike_rank = int((eigenvalues > edge).sum().detach().cpu().item())
        return XiBLog(float(xi_b.detach().cpu()), edge_value, spike_rank, len(self.window), True, source)

    def mp_edge(self) -> float:
        ratio = float(self.feature_dim) / float(max(1, self.window_samples - 1))
        return float((1.0 + ratio ** 0.5) ** 2)
