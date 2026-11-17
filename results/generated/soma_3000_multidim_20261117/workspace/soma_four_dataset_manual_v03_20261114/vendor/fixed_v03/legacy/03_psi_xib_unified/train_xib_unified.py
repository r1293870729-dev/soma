from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from data import (
    StageSpec,
    build_cifar_stages,
    build_dsprites_stages,
    build_shapes3d_stages,
    make_loader,
)
from models import ConvVAE, kl_standard_normal, reconstruction_loss
from signals_xib import XiBSignal
from utils import append_csv, count_parameters, make_generator, resolve_device, set_seed, write_json


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
LOCAL_DSPRITES = DATA_ROOT / "dsprites" / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
LOCAL_SHAPES3D = DATA_ROOT / "shapes3d" / "3dshapes.h5"
LOCAL_CIFAR10 = DATA_ROOT / "cifar10"
LOCAL_CIFARC = DATA_ROOT / "CIFAR-C"


@dataclass(frozen=True)
class StreamBundle:
    train_stages: list[StageSpec]
    probe_stages: list[StageSpec]


class MLPVAE(nn.Module):
    """Small MLP VAE for controlled vector observations."""

    def __init__(self, observation_dim: int, latent_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        input_dim = int(observation_dim)
        for _ in range(max(1, int(hidden_layers))):
            encoder_layers.extend([nn.Linear(input_dim, int(hidden_dim)), nn.GELU()])
            input_dim = int(hidden_dim)
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(int(hidden_dim), int(latent_dim))
        self.fc_logvar = nn.Linear(int(hidden_dim), int(latent_dim))

        decoder_layers: list[nn.Module] = []
        input_dim = int(latent_dim)
        for _ in range(max(1, int(hidden_layers))):
            decoder_layers.extend([nn.Linear(input_dim, int(hidden_dim)), nn.GELU()])
            input_dim = int(hidden_dim)
        decoder_layers.append(nn.Linear(int(hidden_dim), int(observation_dim)))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(x)
        return self.fc_mu(features), self.fc_logvar(features).clamp(min=-12.0, max=12.0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z


class ControlledPhaseDataset(Dataset[dict[str, Tensor]]):
    """Controlled stream: circle -> torus -> torus with four local split degrees."""

    def __init__(
        self,
        phase: str,
        embedding: Tensor,
        size: int,
        noise_std: float,
        seed: int,
        factor_bins: int,
        split_offset: float,
        local_probability: float,
    ) -> None:
        if phase not in {"s1", "t2", "local_split"}:
            raise ValueError(f"Unknown controlled phase: {phase}")
        self.phase = phase
        self.embedding = embedding.float()
        self.size = int(size)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.factor_bins = int(factor_bins)
        self.split_offset = float(split_offset)
        self.local_probability = float(local_probability)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = make_generator(self.seed * 1_000_003 + int(index))
        theta = 2.0 * math.pi * torch.rand((), generator=generator)
        phi = 2.0 * math.pi * torch.rand((), generator=generator)
        coord = torch.zeros(self.embedding.shape[1])
        theta_bin = bin_angle(theta, self.factor_bins)
        phi_bin = 0
        split_bin = 0
        stage_id = {"s1": 0, "t2": 1, "local_split": 2}[self.phase]

        coord[0] = torch.cos(theta)
        coord[1] = torch.sin(theta)
        if self.phase in {"t2", "local_split"}:
            coord[2] = torch.cos(phi)
            coord[3] = torch.sin(phi)
            phi_bin = bin_angle(phi, self.factor_bins)
        if self.phase == "local_split" and torch.rand((), generator=generator).item() < self.local_probability:
            split_bin = int(torch.div(phi, math.pi / 2.0, rounding_mode="floor").clamp(0, 3).item()) + 1
            local = torch.zeros(4)
            local[split_bin - 1] = 1.0
            jitter = 0.1 * torch.randn(4, generator=generator)
            coord[4:8] = self.split_offset * (local + jitter)
        observation = coord @ self.embedding.transpose(0, 1)
        if self.noise_std > 0:
            observation = observation + self.noise_std * torch.randn(observation.shape, generator=generator)
        labels = torch.tensor([theta_bin, phi_bin, split_bin, stage_id], dtype=torch.long)
        return {"image": observation.float(), "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified loss-only Xi-B runner for controlled, dSprites, 3D Shapes, and CIFAR-C.")
    parser.add_argument("--dataset", choices=("controlled", "dsprites", "shapes3d", "cifar"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/xib_unified"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=None)
    parser.add_argument("--stage-steps", type=int, nargs=3, default=(1000, 1000, 1000))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--x-window", type=int, default=8)
    parser.add_argument("--residual-feature-dim", type=int, default=256)
    parser.add_argument("--baseline-warmup-batches", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta-kl", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--probe-every", type=int, default=200)
    parser.add_argument("--probe-start-offset", type=int, default=100)
    parser.add_argument("--probe-train-fraction", type=float, default=0.7)
    parser.add_argument("--probe-train-steps", type=int, default=1000)
    parser.add_argument("--probe-patience", type=int, default=50)
    parser.add_argument("--probe-min-delta", type=float, default=1e-6)
    parser.add_argument("--train-samples-per-stage", type=int, default=4096)
    parser.add_argument("--probe-samples-per-stage", type=int, default=4096)
    parser.add_argument("--probe-seed-offset", type=int, default=50_000)
    parser.add_argument("--dsprites-path", type=Path, default=LOCAL_DSPRITES)
    parser.add_argument("--shapes3d-path", type=Path, default=LOCAL_SHAPES3D)
    parser.add_argument("--cifar10-root", type=Path, default=LOCAL_CIFAR10)
    parser.add_argument("--cifarc-root", type=Path, default=LOCAL_CIFARC)
    parser.add_argument("--download-cifar10", action="store_true")
    parser.add_argument("--observation-dim", type=int, default=64)
    parser.add_argument("--intrinsic-dim", type=int, default=8)
    parser.add_argument("--controlled-latent-dim", type=int, default=4)
    parser.add_argument("--controlled-hidden-dim", type=int, default=32)
    parser.add_argument("--controlled-hidden-layers", type=int, default=1)
    parser.add_argument("--controlled-noise-std", type=float, default=0.02)
    parser.add_argument("--factor-bins", type=int, default=8)
    parser.add_argument("--split-offset", type=float, default=0.8)
    parser.add_argument("--local-probability", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-console", action="store_true")
    return parser.parse_args()


def bin_angle(angle: Tensor, bins: int) -> int:
    value = int(torch.floor((angle % (2.0 * math.pi)) / (2.0 * math.pi) * int(bins)).item())
    return max(0, min(int(bins) - 1, value))


def make_orthogonal_embedding(observation_dim: int, intrinsic_dim: int, seed: int) -> Tensor:
    generator = make_generator(seed)
    matrix = torch.randn(int(observation_dim), int(intrinsic_dim), generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.float()


def default_latent_dims(dataset: str, args: argparse.Namespace) -> tuple[int, ...]:
    if dataset == "controlled":
        return (int(args.controlled_latent_dim),)
    if dataset == "dsprites":
        return (8, 16, 32)
    return (16, 32, 64)


def model_config(dataset: str) -> tuple[tuple[int, int, int] | None, tuple[int, ...], str]:
    if dataset == "dsprites":
        return (1, 64, 64), (32, 64, 128), "bernoulli"
    if dataset == "shapes3d":
        return (3, 64, 64), (32, 64, 128, 256), "gaussian"
    if dataset == "cifar":
        return (3, 32, 32), (64, 128, 256), "gaussian"
    return None, (), "gaussian"


def cycle(loader: DataLoader) -> Iterable[dict[str, Tensor]]:
    while True:
        yield from loader


def split_dataset(dataset: Dataset[Any], train_count: int, probe_count: int, seed: int) -> tuple[Subset[Any], Subset[Any]]:
    total = len(dataset)
    train_count = min(int(train_count), total)
    probe_count = min(int(probe_count), max(0, total - train_count))
    permutation = torch.randperm(total, generator=make_generator(seed)).tolist()
    train_indices = permutation[:train_count]
    probe_indices = permutation[train_count : train_count + probe_count]
    if not probe_indices:
        raise ValueError(f"Dataset with length {total} is too small for a non-overlapping probe split.")
    return Subset(dataset, train_indices), Subset(dataset, probe_indices)


def build_controlled_bundle(args: argparse.Namespace, seed: int) -> StreamBundle:
    embedding = make_orthogonal_embedding(args.observation_dim, args.intrinsic_dim, seed + 123)
    phases = ("s1", "t2", "local_split")
    active = (("theta",), ("theta", "phi"), ("theta", "phi", "split"))
    num_classes = {"theta": int(args.factor_bins), "phi": int(args.factor_bins) + 1, "split": 5, "stage": 3}
    train_stages: list[StageSpec] = []
    probe_stages: list[StageSpec] = []
    for offset, phase in enumerate(phases):
        train_dataset = ControlledPhaseDataset(
            phase=phase,
            embedding=embedding,
            size=int(args.train_samples_per_stage),
            noise_std=float(args.controlled_noise_std),
            seed=seed + offset * 101,
            factor_bins=int(args.factor_bins),
            split_offset=float(args.split_offset),
            local_probability=float(args.local_probability),
        )
        probe_dataset = ControlledPhaseDataset(
            phase=phase,
            embedding=embedding,
            size=int(args.probe_samples_per_stage),
            noise_std=float(args.controlled_noise_std),
            seed=seed + int(args.probe_seed_offset) + offset * 101,
            factor_bins=int(args.factor_bins),
            split_offset=float(args.split_offset),
            local_probability=float(args.local_probability),
        )
        name = {"s1": "S1_circle", "t2": "S2_torus", "local_split": "S3_torus_local_split"}[phase]
        train_stages.append(StageSpec(name, train_dataset, active[offset], num_classes))
        probe_stages.append(StageSpec(name, probe_dataset, active[offset], num_classes))
    return StreamBundle(train_stages, probe_stages)


def build_public_bundle(args: argparse.Namespace, seed: int) -> StreamBundle:
    if args.dataset == "dsprites":
        train = build_dsprites_stages(args.dsprites_path, args.train_samples_per_stage, args.batch_size, seed)
        probe = build_dsprites_stages(args.dsprites_path, args.probe_samples_per_stage, args.batch_size, seed + int(args.probe_seed_offset))
        return StreamBundle(train, probe)
    if args.dataset == "shapes3d":
        train = build_shapes3d_stages(args.shapes3d_path, args.train_samples_per_stage, args.batch_size, seed)
        probe = build_shapes3d_stages(args.shapes3d_path, args.probe_samples_per_stage, args.batch_size, seed + int(args.probe_seed_offset))
        return StreamBundle(train, probe)
    full = build_cifar_stages(args.cifar10_root, args.cifarc_root, args.download_cifar10)
    train_stages: list[StageSpec] = []
    probe_stages: list[StageSpec] = []
    for offset, stage in enumerate(full):
        train_dataset, probe_dataset = split_dataset(
            stage.dataset,
            int(args.train_samples_per_stage),
            int(args.probe_samples_per_stage),
            seed + int(args.probe_seed_offset) + offset * 101,
        )
        train_stages.append(StageSpec(stage.name, train_dataset, stage.active_tasks, stage.num_classes))
        probe_stages.append(StageSpec(stage.name, probe_dataset, stage.active_tasks, stage.num_classes))
    return StreamBundle(train_stages, probe_stages)


def build_bundle(args: argparse.Namespace, seed: int) -> StreamBundle:
    if args.dataset == "controlled":
        return build_controlled_bundle(args, seed)
    return build_public_bundle(args, seed)


def build_model(args: argparse.Namespace, latent_dim: int, device: torch.device) -> tuple[nn.Module, str]:
    if args.dataset == "controlled":
        model = MLPVAE(args.observation_dim, latent_dim, args.controlled_hidden_dim, args.controlled_hidden_layers)
        return model.to(device), "gaussian"
    input_shape, channels, likelihood = model_config(args.dataset)
    if input_shape is None:
        raise ValueError("Image input shape is required for non-controlled datasets.")
    return ConvVAE(input_shape, channels, latent_dim).to(device), likelihood


def compute_reconstruction_terms(args: argparse.Namespace, logits: Tensor, target: Tensor, likelihood: str) -> tuple[Tensor, Tensor, Tensor]:
    if args.dataset == "controlled":
        reconstruction = logits
        loss = torch.nn.functional.mse_loss(reconstruction, target, reduction="mean")
        mse = loss
        return loss, mse, reconstruction
    rec_loss, rec_mse = reconstruction_loss(logits, target, likelihood)
    reconstruction = torch.sigmoid(logits)
    return rec_loss, rec_mse, reconstruction


def collect_latents(model: nn.Module, dataset: Dataset[Any], batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    latents: list[Tensor] = []
    labels: list[Tensor] = []
    previous_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            mu, _ = model.encode(images)
            latents.append(mu.cpu())
            labels.append(batch["labels"].cpu())
    if previous_training:
        model.train()
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


def probe_tasks(dataset_name: str) -> tuple[tuple[str, int], ...]:
    if dataset_name == "controlled":
        return (("theta", 0), ("phi", 1), ("split", 2), ("stage", 3))
    if dataset_name == "dsprites":
        return (("x", 0), ("y", 1), ("shape", 2), ("orientation", 3))
    if dataset_name == "shapes3d":
        return (("floor_hue", 0), ("wall_hue", 1), ("object_hue", 2), ("scale", 3), ("shape", 4), ("orientation", 5))
    return (("class", 0), ("domain", 1))


def train_linear_probe(
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    seed: int,
    max_steps: int,
    patience: int,
    min_delta: float,
    train_fraction: float,
) -> tuple[float, float, int, int, int]:
    if labels.unique().numel() < 2:
        return float("nan"), float("nan"), 0, 0, int(labels.shape[0])
    permutation = torch.randperm(features.shape[0], generator=make_generator(seed))
    split = max(1, int(float(train_fraction) * features.shape[0]))
    if split >= features.shape[0]:
        split = max(1, features.shape[0] - 1)
    train_idx = permutation[:split]
    val_idx = permutation[split:]
    train_x = features[train_idx].float()
    train_y = labels[train_idx].long()
    val_x = features[val_idx].float()
    val_y = labels[val_idx].long()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 17)
        classifier = nn.Linear(train_x.shape[1], int(num_classes))
    optimizer = AdamW(classifier.parameters(), lr=5e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_acc = 0.0
    stale = 0
    for step in range(1, int(max_steps) + 1):
        logits = classifier(train_x)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_logits = classifier(val_x)
            val_loss = torch.nn.functional.cross_entropy(val_logits, val_y).item()
            val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
        if val_loss < best_loss - float(min_delta):
            best_loss = val_loss
            best_acc = val_acc
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            return best_acc, best_loss, step, int(train_x.shape[0]), int(val_x.shape[0])
    return best_acc, best_loss, int(max_steps), int(train_x.shape[0]), int(val_x.shape[0])


def run_probe(
    args: argparse.Namespace,
    model: nn.Module,
    stage: StageSpec,
    stage_index: int,
    global_step: int,
    device: torch.device,
    run_dir: Path,
    seed: int,
) -> None:
    features, labels = collect_latents(model, stage.dataset, args.batch_size, device)
    for task_offset, (task_name, column) in enumerate(probe_tasks(args.dataset)):
        task_labels = labels[:, column].long()
        num_classes = int(stage.num_classes[task_name])
        acc, loss, steps, train_count, test_count = train_linear_probe(
            features,
            task_labels,
            num_classes,
            seed + 30_000 + global_step + task_offset * 997,
            args.probe_train_steps,
            args.probe_patience,
            args.probe_min_delta,
            args.probe_train_fraction,
        )
        append_csv(
            run_dir / "probe_history.csv",
            {
                "step": global_step,
                "stage_index": stage_index + 1,
                "stage": stage.name,
                "task": task_name,
                "accuracy": acc,
                "loss": loss,
                "active": int(task_name in stage.active_tasks),
                "probe_steps": steps,
                "probe_samples": int(features.shape[0]),
                "probe_train_samples": train_count,
                "probe_test_samples": test_count,
                "probe_train_fraction": float(args.probe_train_fraction),
                "probe_source": "independent_stage_dataset",
                "dataset": args.dataset,
            },
        )


def train_one(args: argparse.Namespace, seed: int, latent_dim: int, device: torch.device) -> dict[str, Any]:
    set_seed(seed)
    run_name = f"{args.dataset}_z{latent_dim}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / args.dataset / f"z{latent_dim}" / run_name
    done_path = run_dir / "DONE"
    if args.resume and done_path.exists():
        return {"status": "skipped", "dataset": args.dataset, "latent_dim": latent_dim, "seed": seed, "run_dir": str(run_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_args.json", vars(args) | {"seed": int(seed), "latent_dim": int(latent_dim)})
    bundle = build_bundle(args, seed)
    train_loaders = [
        make_loader(stage.dataset, args.batch_size, seed + index * 10_007)
        for index, stage in enumerate(bundle.train_stages)
    ]
    model, likelihood = build_model(args, latent_dim, device)
    optimizer = AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
    window_samples = int(args.x_window) * int(args.batch_size)
    gamma = 1.0 / (20.0 * float(args.x_window))
    signal = XiBSignal(
        feature_dim=int(args.residual_feature_dim),
        window_samples=window_samples,
        gamma=gamma,
        alpha=float(args.alpha),
        warmup_batches=int(args.baseline_warmup_batches),
        seed=seed + 777,
    )
    global_step = 0
    last_row: dict[str, Any] = {}
    for stage_index, (train_stage, probe_stage, loader, steps) in enumerate(
        zip(bundle.train_stages, bundle.probe_stages, train_loaders, args.stage_steps, strict=True)
    ):
        iterator = cycle(loader)
        for local_step in range(1, int(steps) + 1):
            global_step += 1
            batch = next(iterator)
            images = batch["image"].to(device)
            logits, mu, logvar, _ = model(images)
            rec_loss, rec_mse, reconstruction_for_residual = compute_reconstruction_terms(args, logits, images, likelihood)
            kl = kl_standard_normal(mu, logvar)
            objective = rec_loss + float(args.beta_kl) * kl
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                residual = reconstruction_for_residual.detach() - images.detach()
                xi_log = signal.update_and_measure(residual, train_stage.name)
            if global_step % int(args.log_every) == 0 or local_step == 1:
                last_row = {
                    "step": global_step,
                    "stage_index": stage_index + 1,
                    "stage": train_stage.name,
                    "local_step": local_step,
                    "objective": float(objective.detach().cpu()),
                    "reconstruction_loss": float(rec_loss.detach().cpu()),
                    "reconstruction_mse": float(rec_mse.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                    "xi_b": xi_log.xi_b,
                    "lambda_edge": xi_log.lambda_edge,
                    "spike_rank": xi_log.spike_rank,
                    "window_count": xi_log.window_count,
                    "baseline_ready": int(xi_log.baseline_ready),
                    "readout_source": xi_log.readout_source,
                    "dataset": args.dataset,
                    "latent_dim": int(latent_dim),
                    "seed": int(seed),
                    "mode": "loss_only",
                    "train_samples_per_stage": int(args.train_samples_per_stage),
                    "probe_samples_per_stage": int(args.probe_samples_per_stage),
                }
                append_csv(run_dir / "train_history.csv", last_row)
            should_probe = local_step >= int(args.probe_start_offset) and (
                (local_step - int(args.probe_start_offset)) % int(args.probe_every) == 0
            )
            if should_probe:
                run_probe(args, model, probe_stage, stage_index, global_step, device, run_dir, seed)
            if not args.quiet_console and global_step % max(1, int(args.log_every) * 10) == 0:
                print(
                    f"[xib] dataset={args.dataset} seed={seed} z={latent_dim} step={global_step} "
                    f"stage={train_stage.name} rec={float(rec_mse):.6f} xi_b={xi_log.xi_b:.4f}",
                    flush=True,
                )
    torch.save(
        {"model": model.state_dict(), "dataset": args.dataset, "latent_dim": int(latent_dim)},
        run_dir / "checkpoint.pt",
    )
    summary = {
        "status": "completed",
        "dataset": args.dataset,
        "latent_dim": int(latent_dim),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "steps": int(sum(args.stage_steps)),
        "parameters": count_parameters(model),
        "window_samples": window_samples,
        "gamma": gamma,
        **last_row,
    }
    write_json(run_dir / "summary.json", summary)
    append_csv(args.output_dir / f"{args.dataset}_summary.csv", summary)
    done_path.write_text("done\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    seeds = tuple(args.seeds) if args.seeds is not None else (int(args.seed),)
    latent_dims = tuple(args.latent_dims) if args.latent_dims is not None else default_latent_dims(args.dataset, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for latent_dim in latent_dims:
            results.append(train_one(args, int(seed), int(latent_dim), device))
    write_json(args.output_dir / f"{args.dataset}_all_results.json", {"results": results})
    print(args.output_dir / f"{args.dataset}_summary.csv", flush=True)


if __name__ == "__main__":
    main()
