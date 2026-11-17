from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from utils import make_torch_generator


CIFAR_C_SEVERITY_SIZE = 10_000


@dataclass(frozen=True)
class StageSpec:
    name: str
    steps: int
    loader: DataLoader


class CifarCSeverityDataset(Dataset[tuple[Tensor, int]]):
    """读取单一 CIFAR-C corruption 和单一 severity。

    CIFAR-C 每个 corruption 存成一个 50k 的 numpy 数组。
    五个 severity 按顺序各占连续 10k 个样本。
    """

    def __init__(self, root: str | Path, corruption: str, severity: int) -> None:
        if severity < 1 or severity > 5:
            raise ValueError("CIFAR-C severity must be in [1, 5].")
        root = Path(root)
        image_path = root / f"{corruption}.npy"
        label_path = root / "labels.npy"
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-C file: {image_path}. "
                "Download CIFAR-C and set data.cifarc_root in config.toml."
            )
        if not label_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-C labels file: {label_path}.")

        images = np.load(image_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        start = (severity - 1) * CIFAR_C_SEVERITY_SIZE
        end = severity * CIFAR_C_SEVERITY_SIZE
        self.images = images[start:end]
        self.labels = labels[start:end]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image = self.images[index]
        label = int(self.labels[index])
        return self.transform(image), label


class MixedCifarCDataset(Dataset[tuple[Tensor, int]]):
    """从 CIFAR-C 的 corruption 和 severity 中做随机访问式混合采样。

    每个 index 会确定性映射到一个 corruption、一个 severity 切片以及该切片
    内的一张图片。这样可以模拟“从完整 CIFAR-10-C 分布采样”的阶段，同时
    保持实验可复现。
    """

    def __init__(
        self,
        root: str | Path,
        corruptions: list[str] | tuple[str, ...] | None = None,
        severities: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
        virtual_size: int = 50_000,
        seed: int = 0,
    ) -> None:
        root = Path(root)
        label_path = root / "labels.npy"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-C labels file: {label_path}.")
        if corruptions is None:
            corruptions = tuple(sorted(path.stem for path in root.glob("*.npy") if path.name != "labels.npy"))
        if not corruptions:
            raise ValueError(f"No CIFAR-C corruption .npy files found under {root}.")
        if not severities:
            raise ValueError("At least one CIFAR-C severity is required.")

        self.root = root
        self.corruptions = tuple(corruptions)
        self.severities = tuple(int(value) for value in severities)
        self.virtual_size = int(virtual_size)
        self.seed = int(seed)
        self.labels = np.load(label_path, mmap_mode="r")
        self.images_by_corruption = {
            corruption: np.load(root / f"{corruption}.npy", mmap_mode="r") for corruption in self.corruptions
        }
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        # 这里的随机数是按 index 固定种子生成的：同一个 index 永远对应同一个
        # corruption/severity/image。因此即使这个阶段表示“完整 CIFAR-C 混合采样”，
        # 每次实验仍然可复现。
        rng = np.random.default_rng(self.seed * 1_000_003 + int(index))
        corruption = self.corruptions[int(rng.integers(0, len(self.corruptions)))]
        severity = self.severities[int(rng.integers(0, len(self.severities)))]
        local_index = int(rng.integers(0, CIFAR_C_SEVERITY_SIZE))
        absolute_index = (severity - 1) * CIFAR_C_SEVERITY_SIZE + local_index
        image = self.images_by_corruption[corruption][absolute_index]
        label = int(self.labels[absolute_index])
        return self.transform(image), label


def build_abrupt_stream(config: dict[str, Any], device: torch.device | str) -> list[StageSpec]:
    data_config = config["data"]
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config.get("num_workers", 0))
    pin_memory = torch.device(device).type == "cuda"
    clean_generator = make_torch_generator(int(data_config.get("seed", 0)) + 11)
    corrupt_generator = make_torch_generator(int(data_config.get("seed", 0)) + 29)

    clean_dataset = build_cifar10_dataset(
        root=data_config["cifar10_root"],
        train=bool(data_config.get("clean_train", True)),
        download=bool(data_config.get("download_cifar10", True)),
    )
    corrupt_dataset = CifarCSeverityDataset(
        root=data_config["cifarc_root"],
        corruption=str(data_config["corruption"]),
        severity=int(data_config["severity"]),
    )
    return [
        StageSpec(
            name="clean_cifar10",
            steps=int(data_config["clean_steps"]),
            loader=DataLoader(
                clean_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                generator=clean_generator,
            ),
        ),
        StageSpec(
            name=f"cifarc_{data_config['corruption']}_s{data_config['severity']}",
            steps=int(data_config["corrupt_steps"]),
            loader=DataLoader(
                corrupt_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                generator=corrupt_generator,
            ),
        ),
    ]


def build_cifar10_dataset(root: str | Path, train: bool, download: bool) -> Dataset[tuple[Tensor, int]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return datasets.CIFAR10(root=str(root), train=train, download=download, transform=transform)


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch
