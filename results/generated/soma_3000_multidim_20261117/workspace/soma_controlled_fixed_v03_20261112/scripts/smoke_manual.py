"""Short engineering fixture only: the approved fixed controller, 12 updates each."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.b_weight_sensitivity.formal_common import load_config, arm_specs, write_json, code_digest
from experiments.b_weight_sensitivity.formal_runner import args_for, run_unit
from experiments.b_weight_sensitivity.formal_data import build_disjoint_bundle
from experiments.b_weight_sensitivity.historical_loader import HistoricalSource
from experiments.manual_v03.signal import FixedPsi, SignalConfig


def no_monitor_final_model(unit, config, bundle, historical):
    from experiments.b_weight_sensitivity.formal_runner import (initial_references, make_controller,
        losses_for, weights_for, checked_update, tensor_digest)
    from experiments.b_weight_sensitivity.gradient_diagnostics import collect_gradient_vectors
    from torch.optim import AdamW
    entry, models = historical.load('train_xib_unified'), historical.load('models')
    data, utils = historical.load('data'), historical.load('utils')
    args = args_for(entry, config, 'controlled')
    utils.set_seed(unit['seed'])
    device = torch.device('cpu')
    model, likelihood = entry.build_model(args, unit['latent_dim'], device)
    references = initial_references(entry, models, args, model, bundle.train_stages[0], likelihood,
                                    unit['seed'], device, config['calibration_batches'])
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    controller, control_optimizer = make_controller(unit, references, config, device)
    model.train()
    for index, stage in enumerate(bundle.train_stages):
        iterator = entry.cycle(data.make_loader(stage.dataset, args.batch_size, unit['seed'] + index * 10007))
        for _ in range(args.stage_steps[index]):
            losses, _, _ = losses_for(entry, models, args, model, next(iterator)['image'], likelihood)
            norms, _ = collect_gradient_vectors(losses, model.encoder.parameters(), weights_for(controller, losses))
            checked_update(model, optimizer, controller, control_optimizer, losses,
                           torch.tensor([row['raw_norm'] for row in norms]))
    return tensor_digest(model)


def replay_run(path):
    metadata = json.loads((path / 'calibration.json').read_text())
    with np.load(path / 'calibration_residuals.npz', allow_pickle=False) as values:
        signal = FixedPsi.fit(values['white'], values['edge'], SignalConfig(**metadata['config']), metadata['roles'])
    if signal.reference_hash() != metadata['reference_sha256']:
        raise ValueError('Reconstructed reference hash differs')
    residuals = np.load(path / 'preupdate_readout_residuals.npy', mmap_mode='r', allow_pickle=False)
    maximum = 0.
    rows = [json.loads(s) for s in (path / 'psi_signal.jsonl').read_text().splitlines()]
    if len(rows) != len(residuals):
        raise ValueError('Residual/log denominator differs')
    for residual, row in zip(residuals, rows):
        replay = signal.update(residual)
        if replay['status'] != row['status'] or replay['active_modes'] != row['active_modes']:
            raise ValueError('Replay state or active set mismatch')
        if row['psi'] is not None:
            maximum = max(maximum, abs(replay['psi'] - row['psi']))
            if maximum > 1e-10:
                raise ValueError('Signal replay exceeds frozen numerical tolerance')
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', required=True)
    options = parser.parse_args()
    if not options.experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in options.experiment_id):
        raise ValueError('Unsafe experiment ID')
    torch.set_num_threads(1)
    config = load_config()
    config.update(stage_steps=[4, 4, 4], train_samples_per_stage=512,
                  validation_samples_per_stage=64, final_probe=False)
    historical = HistoricalSource(ROOT / 'legacy/03_psi_xib_unified')
    args = args_for(historical.load('train_xib_unified'), config, 'controlled')
    policy = dict(args.manual_signal_policy)
    policy['numeric'] = dict(policy['numeric'], window=16)
    policy.update(white_samples=128, edge_windows=8, edge_cover_windows=8)
    args.manual_signal_policy = policy
    # run_unit reparses args; inject fixture policy through a fixture-local wrapper,
    # never alter configs/signal.json or the live historical project.
    from experiments.b_weight_sensitivity import formal_runner
    original_args_for = formal_runner.args_for
    def fixture_args(entry, configuration, dataset):
        value = original_args_for(entry, configuration, dataset)
        value.manual_signal_policy = policy
        return value
    formal_runner.args_for = fixture_args
    out = ROOT / 'results/engineering' / options.experiment_id
    out.mkdir(parents=True, exist_ok=False)
    bundle, split = build_disjoint_bundle(args, 0, historical, out / 'assets')
    results = []
    for index, (name, method, beta) in enumerate(arm_specs(config)):
        unit = dict(dataset='controlled', latent_dim=4, seed=0, arm=name, method=method, beta=beta)
        result = run_unit(unit, config, historical, bundle, split, torch.device('cpu'), out,
                          'ENGINEERING_ONLY_12_UPDATES', index, len(arm_specs(config)))
        result['replay_maximum_error'] = replay_run(Path(result['run_dir']))
        result['no_monitor_final_model_sha256'] = no_monitor_final_model(unit, config, bundle, historical)
        if result['no_monitor_final_model_sha256'] != result['final_model_sha256']:
            raise ValueError('Read-only signal logging changed actual model updates')
        result['monitor_does_not_change_model_updates'] = True
        results.append(result)
    checks = {tuple(row[k] for k in ('initial_model_sha256', 'split_sha256', 'data_order_sha256')) for row in results}
    if len(checks) != 1:
        raise ValueError('Paired controller initialization/split/order differ')
    write_json(out / 'fixture_summary.json', dict(status='passed_engineering_only', runs=len(results),
               updates_per_run=12, code_sha256=code_digest(), final_test_used=False, independent_generators=1, seed=0,
               maximum_replay_error=max(r['replay_maximum_error'] for r in results), runs_detail=results))
    print('Engineering fixture passed: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
