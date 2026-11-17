"""Log the exact historical XiBSignal; no signal mathematics is copied here."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from .historical_loader import HistoricalSource

class LegacyBMonitor:
    def __init__(self, legacy_signal_file, feature_dim, window_samples, gamma, alpha,
                 warmup_batches, seed, epsilon=1e-5, csv_path=None, expected_hashes=None):
        source = Path(legacy_signal_file).resolve()
        if source.name != "signals_xib.py" or not source.is_file():
            raise FileNotFoundError("Historical signals_xib.py not found: " + str(source))
        self.loader = HistoricalSource(source.parent, expected_hashes)
        module = self.loader.load("signals_xib")
        self.source_sha256 = self.loader.hashes["signals_xib.py"]
        self.config = dict(feature_dim=int(feature_dim), window_samples=int(window_samples),
                           gamma=float(gamma), alpha=float(alpha), warmup_batches=int(warmup_batches),
                           seed=int(seed), epsilon=float(epsilon))
        self.config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.signal = module.XiBSignal(**self.config)
        self.csv_path = Path(csv_path) if csv_path else None
        if self.csv_path and self.csv_path.exists():
            raise FileExistsError("Refuse to append to an existing signal run: " + str(self.csv_path))
        self.run_id = None
        self.last_step = 0
        self.last_row = None

    @torch.no_grad()
    def update(self, residual, stage_name, step, run_id, checkpoint_id, event_label="unlabelled"):
        if not run_id or not checkpoint_id or not stage_name:
            raise ValueError("run_id, checkpoint_id and stage_name are required")
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError("A monitor instance belongs to one run only")
        if isinstance(step, bool) or not isinstance(step, int) or step != self.last_step + 1:
            raise ValueError("Duplicate, skipped or unordered signal step")
        if not bool(torch.isfinite(residual.detach()).all()):
            raise FloatingPointError("Non-finite residual")
        self.loader.verify()
        result = self.signal.update_and_measure(residual.detach(), stage_name)
        if not all(math.isfinite(float(v)) for v in (result.xi_b, result.lambda_edge)):
            raise FloatingPointError("Historical B returned a non-finite value")
        row = {"run_id": str(run_id), "step": step, "checkpoint_id": str(checkpoint_id),
               "stage_name": str(stage_name), "event_label": str(event_label),
               "xi_b": float(result.xi_b), "lambda_edge": float(result.lambda_edge),
               "spike_rank": int(result.spike_rank), "window_count": int(result.window_count),
               "baseline_ready": bool(result.baseline_ready), "readout_source": str(result.readout_source),
               "signal_ready": bool(result.baseline_ready and result.window_count >= self.config["window_samples"]),
               "source_sha256": self.source_sha256, "config_sha256": self.config_sha256,
               "residual_convention": "historical_reconstruction_minus_observation"}
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            first = self.last_row is None
            with self.csv_path.open("x" if first else "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if first:
                    writer.writeheader()
                writer.writerow(row)
        self.run_id, self.last_step, self.last_row = run_id, step, row
        return result
