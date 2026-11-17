"""Read-only diagnostics on the single smoke run's saved pre-update checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
from .historical_loader import HistoricalSource
from .engineering_smoke import parse_historical_args, model_hash, write_json
from .gradient_diagnostics import collect_gradient_vectors

def run(legacy_root, smoke, out):
    smoke, out = Path(smoke), Path(out)
    status = json.loads((smoke / "status.json").read_text())
    if status["status"] != "complete" or status["model_updates"] != 80:
        raise ValueError("A complete, capped smoke run is required")
    if out.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = HistoricalSource(legacy_root)
    entry = source.load("train_xib_unified")
    config = json.loads((smoke / "resolved_config.json").read_text())
    args = parse_historical_args(entry, config)
    checkpoints = json.loads((smoke / "checkpoint_manifest.json").read_text())
    with (smoke / "training_log.csv").open() as f:
        original = {int(r["step"]): r for r in csv.DictReader(f)}
    norms, cosines = [], []
    for checkpoint in checkpoints:
        path = smoke / checkpoint["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint["sha256"]:
            raise ValueError("Checkpoint hash changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model, likelihood = entry.build_model(args, args.controlled_latent_dim, torch.device("cpu"))
            model.load_state_dict(saved["model"])
            model.train()
            assert model_hash(model) == checkpoint["model_sha256"]
            torch.set_rng_state(saved["torch_rng_before_forward"])
            x = saved["input_batch"]
            logits, mu, logvar, _ = model(x)
            rec, _, _ = entry.compute_reconstruction_terms(args, logits, x, likelihood)
            kl = entry.kl_standard_normal(mu, logvar)
            for name, value in (("raw_reconstruction", rec), ("raw_kl", kl)):
                if float(value.detach()) != float(original[saved["step"]][name]):
                    raise AssertionError("Pre-update forward replay differs from saved training loss")
            before = model_hash(model)
            rows, pairs = collect_gradient_vectors({"reconstruction": rec, "kl": kl},
                                                   model.encoder.parameters(), config["coefficients"])
            assert before == model_hash(model) and all(p.grad is None for p in model.parameters())
            common = {"run_id": smoke.name, "seed": args.seed, "step": saved["step"],
                      "stage_name": saved["stage"], "checkpoint_id": saved["checkpoint_id"],
                      "parameter_group": "MLPVAE.encoder", "model_updated": False}
            norms.extend([dict(common, **r) for r in rows])
            cosines.extend([dict(common, **r) for r in pairs])
    for name, rows in (("gradient_diagnostics.csv", norms), ("gradient_cosines.csv", cosines)):
        with (out / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for stage in sorted({r["stage_name"] for r in cosines}):
        values = [r["cosine"] for r in cosines if r["stage_name"] == stage and r["kind"] == "raw" and r["cosine"] is not None]
        summary[stage] = {"batches": len(values), "negative_batches": sum(x<0 for x in values),
                          "negative_fraction": sum(x<0 for x in values)/len(values) if values else None,
                          "mean_cosine": sum(values)/len(values) if values else None}
    result = {"status": "complete_read_only", "source_hashes": source.hashes, "checkpoints_replayed": len(checkpoints),
              "norm_rows": len(norms), "cosine_rows": len(cosines), "parameter_group": "MLPVAE.encoder",
              "independent_trained_seeds": 1, "model_optimizer_steps": 0, "b_state_updates": 0,
              "scientific_weight_comparison": False, "pcgrad_enabled": False,
              "pcgrad_stop_reason": "Single short trajectory; no cross-seed evidence or pre-frozen sustained-conflict criterion.",
              "stages": summary}
    source.verify()
    write_json(out / "summary.json", result)
    return result

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root",type=Path,required=True)
    p.add_argument("--smoke",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.legacy_root,a.smoke,a.out),ensure_ascii=False,indent=2))
