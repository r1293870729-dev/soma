"""One capped CPU run of the original 03 model; never a paired scientific sweep."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from .historical_loader import HistoricalSource
from .legacy_b_monitor import LegacyBMonitor
from .weighting import FixedWeightController

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class IndexedHistoricalDataset(Dataset):
    def __init__(self, dataset, stage):
        self.dataset, self.stage = dataset, stage
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        result = dict(self.dataset[index])
        result["sample_id"] = "train:" + self.stage + ":seed=" + str(self.dataset.seed) + ":index=" + str(index)
        return result

def parse_historical_args(entry, config):
    previous = sys.argv
    sys.argv = [str(entry.__file__), "--dataset", "controlled", "--device", "cpu",
                "--seed", str(config["seed"]), "--stage-steps", *map(str, config["stage_steps"])]
    try:
        return entry.parse_args()
    finally:
        sys.argv = previous

def run(legacy_root, out, config_path=None):
    config_path = config_path or Path(__file__).parent / "configs/engineering_smoke.json"
    config = json.loads(Path(config_path).read_text())
    if config["status"] != "engineering-only" or config["formal_test_unlocked"] or config["arm"] != "engineering_fixed":
        raise ValueError("Only the capped engineering fixed-weight run is enabled")
    if config["stage_steps"] != [32, 24, 24] or sum(config["stage_steps"]) > 80 or config["max_model_updates"] != 80:
        raise ValueError("This entrypoint cannot launch a formal or extended training run")
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new run ID; prior failures and outputs must be preserved")
    out.mkdir(parents=True)
    started = time.perf_counter()
    status = {"status": "running", "kind": "engineering-only", "model_updates": 0,
              "formal_test_unlocked": False, "pcgrad_enabled": False}
    write_json(out / "status.json", status)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        source = HistoricalSource(legacy_root)
        entry = source.load("train_xib_unified")
        args = parse_historical_args(entry, config)
        assert args.batch_size == config["batch_size"] == 64
        assert float(args.beta_kl) == config["coefficients"]["kl"] == 0.001
        entry.set_seed(args.seed)
        model, likelihood = entry.build_model(args, args.controlled_latent_dim, torch.device("cpu"))
        assert type(model) is entry.MLPVAE
        initial_hash = model_hash(model)
        optimizer = AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
        controller = FixedWeightController(config["coefficients"])
        bundle = entry.build_controlled_bundle(args, args.seed)
        loaders = [entry.make_loader(IndexedHistoricalDataset(stage.dataset, stage.name), args.batch_size,
                                    args.seed + i * 10007) for i, stage in enumerate(bundle.train_stages)]
        signal_config = dict(feature_dim=int(args.residual_feature_dim),
                             window_samples=int(args.x_window) * int(args.batch_size),
                             gamma=1.0 / (20.0 * float(args.x_window)), alpha=float(args.alpha),
                             warmup_batches=int(args.baseline_warmup_batches), seed=int(args.seed) + 777, epsilon=1e-5)
        monitor = LegacyBMonitor(Path(legacy_root) / "signals_xib.py", csv_path=out / "b_trajectory.csv", **signal_config)
        reference = source.load("signals_xib").XiBSignal(**signal_config)
        run_id = out.name
        resolved = dict(config, run_id=run_id, original_args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                        signal=signal_config, source_hashes=source.hashes, initial_model_sha256=initial_hash,
                        baseline_provenance="03 source/README defaults, not an old run resolved_args file",
                        total_training_steps=80, optimizer="AdamW", learning_rate=args.learning_rate, weight_decay=0.0,
                        checkpoint_selection="fixed final step; no validation-based selection")
        write_json(out / "resolved_config.json", resolved)
        (out / "checkpoints").mkdir()
        rows, checkpoints, ids = [], [], []
        global_step = 0
        for stage_index, stage in enumerate(bundle.train_stages):
            iterator = iter(loaders[stage_index])
            for local_step in range(1, config["stage_steps"][stage_index] + 1):
                global_step += 1
                batch = next(iterator)
                images = batch["image"]
                optimizer.zero_grad(set_to_none=True)
                checkpoint_id = run_id + ":pre-update:" + str(global_step)
                checkpoint_file = "checkpoints/pre_" + str(global_step).zfill(4) + ".pt"
                before = model_hash(model)
                saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "torch_rng_before_forward": torch.get_rng_state(), "step": global_step,
                         "stage": stage.name, "stage_index": stage_index, "input_batch": images,
                         "sample_ids": list(batch["sample_id"]), "checkpoint_id": checkpoint_id,
                         "model_sha256": before, "source_hashes": source.hashes}
                torch.save(saved, out / checkpoint_file)
                checkpoints.append({"checkpoint_id": checkpoint_id, "file": checkpoint_file,
                                    "sha256": hashlib.sha256((out / checkpoint_file).read_bytes()).hexdigest(),
                                    "model_sha256": before, "step": global_step, "stage": stage.name})
                logits, mu, logvar, _ = model(images)
                rec, mse, reconstruction = entry.compute_reconstruction_terms(args, logits, images, likelihood)
                kl = entry.kl_standard_normal(mu, logvar)
                losses = {"reconstruction": rec, "kl": kl}
                objective = controller.objective(losses)
                if not torch.equal(objective, rec + args.beta_kl * kl):
                    raise AssertionError("Fixed controller changed the original objective")
                if not bool(torch.isfinite(objective.detach())):
                    raise FloatingPointError("Non-finite model objective")
                objective.backward()
                optimizer.step()
                status["model_updates"] += 1
                # Match the historical order: step first, then use the saved pre-update prediction.
                residual = reconstruction.detach() - images.detach()
                after_update_hash = model_hash(model)
                rng = torch.get_rng_state().clone()
                gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
                result = monitor.update(residual, stage.name, global_step, run_id, checkpoint_id,
                                        "scheduled_transition" if stage_index and local_step == 1 else "within_stage")
                golden = reference.update_and_measure(residual, stage.name)
                if asdict(result) != asdict(golden):
                    raise AssertionError("Adapter differs from the untouched original XiBSignal")
                if not torch.equal(rng, torch.get_rng_state()) or after_update_hash != model_hash(model):
                    raise AssertionError("B logging changed model or global RNG")
                for p, old in zip(model.parameters(), gradients):
                    if (p.grad is None) != (old is None) or (old is not None and not torch.equal(p.grad, old)):
                        raise AssertionError("B logging changed model gradients")
                rows.append({"run_id": run_id, "step": global_step, "checkpoint_id": checkpoint_id,
                             "stage_name": stage.name, "raw_reconstruction": float(rec.detach()), "raw_kl": float(kl.detach()),
                             "weight_reconstruction": 1.0, "weight_kl": float(args.beta_kl),
                             "weighted_reconstruction": float(rec.detach()), "weighted_kl": float(args.beta_kl * kl.detach()),
                             "objective": float(objective.detach()), "reconstruction_mse": float(mse.detach()),
                             "model_updates_for_batch": 1, "batch_sha256": tensor_hash(images),
                             "signal_ready": monitor.last_row["signal_ready"], "status": "complete"})
                ids.append({"step": global_step, "stage_name": stage.name, "sample_ids": list(batch["sample_id"])})
        assert global_step == status["model_updates"] == 80
        validation = []
        with torch.no_grad():
            for stage in bundle.probe_stages:
                # Fixed engineering holdout only; no probe classifier, test evaluation, or model selection.
                x = torch.stack([stage.dataset[i]["image"] for i in range(64)])
                prediction = model.decoder(model.encode(x)[0])
                validation.append({"stage_name": stage.name, "role": "engineering_holdout",
                                   "samples": 64, "metric": "deterministic_mean_reconstruction_mse",
                                   "value": float(torch.mean((prediction-x).square()))})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 80,
                    "torch_rng": torch.get_rng_state(), "source_hashes": source.hashes}, out / "checkpoints/final.pt")
        import csv
        with (out / "training_log.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(out / "checkpoint_manifest.json", checkpoints)
        write_json(out / "batch_manifest.json", ids)
        write_json(out / "engineering_validation.json", validation)
        write_json(out / "role_manifest.json", {
            "training": "Exact source-generated stage datasets and original shuffled loader ordering; IDs in batch_manifest.json",
            "engineering_holdout": [{"stage": s.name, "generator_seed": s.dataset.seed, "indices": list(range(64))} for s in bundle.probe_stages],
            "test_evaluated": False, "formal_calibration_frozen": False,
            "limitations": "Distinct generator identities/seeds; no formal dataset-role approval or historical resolved config recovered."})
        source.verify()
        status.update(status="complete", signal_rows=80, training_rows=80,
                      ready_rows=sum(r["signal_ready"] for r in rows),
                      exact_original_b_matches=80, logger_model_rng_gradient_checks=80,
                      source_hashes_unchanged=True, seed=args.seed, initial_model_sha256=initial_hash,
                      final_model_sha256=model_hash(model), validation_role="engineering_holdout_only",
                      wall_seconds=time.perf_counter()-started, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_rss_unit="bytes" if sys.platform == "darwin" else "KiB", independent_training_seeds=1,
                      scientific_weight_comparison=False, threshold_frozen=False, test_evaluated=False)
        write_json(out / "status.json", status)
        return status
    except BaseException as exc:
        status.update(status="failed", exception_type=type(exc).__name__, error=str(exc),
                      wall_seconds=time.perf_counter()-started)
        write_json(out / "status.json", status)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args()
    print(json.dumps(run(a.legacy_root, a.out), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
