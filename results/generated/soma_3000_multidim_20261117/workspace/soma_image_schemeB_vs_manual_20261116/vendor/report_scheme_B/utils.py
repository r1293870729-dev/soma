from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def set_seed(seed: int) -> None:
    """Set every stochastic backend we use to a reproducible state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # Older PyTorch does not support warn_only.
        torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int, *, device: str | torch.device = "cpu") -> torch.Generator:
    """Create an isolated generator so one component cannot consume another component's RNG stream."""
    return torch.Generator(device=str(device)).manual_seed(int(seed))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def prepare_run_dir(output_dir: Path, config_path: Path, config: dict[str, Any], mode: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{mode}_seed{config['run']['seed']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.toml")
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
