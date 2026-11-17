"""Paired-seed full-budget training; original 03 model/loss/B functions are called."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .formal_common import ROOT, CONFIG, code_digest, load_config, plan, planned_units, sha_file, unit_id, write_json
from .formal_data import build_disjoint_bundle
from .gradient_diagnostics import collect_gradient_vectors
from .historical_loader import HistoricalSource
from ..manual_v03.monitor import calibrate_monitor, checkpoint_state
from .progress import bar, duration
from .weighting import (FixedWeightController, InitialScaleController, UncertaintyController,
                        GradNormController, assert_disjoint_optimizers)


def tensor_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode()); result.update(str(array.shape).encode())
        result.update(str(array.dtype).encode()); result.update(array.tobytes())
    return result.hexdigest()


def rng_digest(device):
    value = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if device.type == 'cuda':
        value.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return value.hexdigest()


def append(path, row):
    path = Path(path)
    first = not path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)


def args_for(entry, config, dataset):
    previous = sys.argv
    try:
        sys.argv = [str(entry.__file__), '--dataset', dataset]
        args = entry.parse_args()
    finally:
        sys.argv = previous
    args.stage_steps = config['stage_steps']
    args.batch_size = config['batch_size']
    args.learning_rate = config['learning_rate']
    args.train_samples_per_stage = config['train_samples_per_stage']
    args.probe_samples_per_stage = config['validation_samples_per_stage']
    args.dsprites_path = ROOT / 'data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    args.shapes3d_path = ROOT / 'data/shapes3d/3dshapes.h5'
    args.cifar10_root = ROOT / 'data/cifar10'
    args.cifarc_root = ROOT / 'data/CIFAR-C'
    args.download_cifar10 = False
    args.manual_signal_policy = json.loads((ROOT / 'configs/signal.json').read_text())
    return args


def fork_devices(device):
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []


def losses_for(entry, models, args, model, images, likelihood):
    logits, mu, logvar, _ = model(images)
    rec, mse, reconstructed = entry.compute_reconstruction_terms(args, logits, images, likelihood)
    return {'reconstruction': rec, 'kl': models.kl_standard_normal(mu, logvar)}, mse, reconstructed


def initial_references(entry, models, args, model, stage, likelihood, seed, device, batches):
    sums = {'reconstruction': 0.0, 'kl': 0.0}
    count = 0
    # Every arm sees the same stage-1 training prefix, without optimizer/B updates.
    loader = DataLoader(stage.dataset, batch_size=args.batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(seed + 84001))
    with torch.random.fork_rng(devices=fork_devices(device)), torch.no_grad():
        torch.manual_seed(seed + 84002)
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            losses, _, _ = losses_for(entry, models, args, model, batch['image'].to(device), likelihood)
            size = len(batch['image']); count += size
            for key in sums:
                sums[key] += float(losses[key]) * size
    references = {key: value / count for key, value in sums.items()}
    if count != batches * args.batch_size or any(not math.isfinite(v) or v <= 0 for v in references.values()):
        raise ValueError('Nonpositive/nonfinite or incomplete fixed training calibration reference')
    return references


def make_controller(unit, references, config, device):
    if unit['method'] == 'fixed':
        return FixedWeightController({'reconstruction': 1.0, 'kl': unit['beta']}), None
    if unit['method'] == 'initial_scale':
        return InitialScaleController(references), None
    if unit['method'] == 'uncertainty':
        controller = UncertaintyController(['reconstruction', 'kl']).to(device)
    else:
        controller = GradNormController(['reconstruction', 'kl'], references, config['gradnorm_alpha']).to(device)
    return controller, AdamW(controller.parameters(), lr=config['controller_learning_rate'], weight_decay=0.0)


def weights_for(controller, losses):
    if isinstance(controller, GradNormController):
        return controller.weights_dict()
    if isinstance(controller, UncertaintyController):
        return controller.current_weights()
    return {key: float(value) for key, value in controller.weights(losses).items()}


def checked_update(model, optimizer, controller, controller_optimizer, losses, raw_norms):
    optimizer.zero_grad(set_to_none=True)
    if controller_optimizer:
        controller_optimizer.zero_grad(set_to_none=True)
    balancing = None
    if isinstance(controller, GradNormController):
        balancing = controller.gradnorm_loss(losses, raw_norms)
        objective = controller.model_objective(losses)
    elif isinstance(controller, UncertaintyController):
        objective = controller.objective(losses)
    else:
        w = controller.weights(losses)
        objective = sum(w[k] * losses[k] for k in losses)
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError('Nonfinite model objective')
    objective.backward()
    if balancing is not None:
        balancing.backward()  # Detached norms/target; controller-only graph.
    for group in [model.parameters(), controller.parameters() if controller_optimizer else []]:
        for parameter in group:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError('Nonfinite gradient before optimizer update')
    optimizer.step()  # Exactly one model update per training batch.
    if controller_optimizer:
        controller_optimizer.step()
    return float(objective.detach()), None if balancing is None else float(balancing.detach())


def validation(entry, models, args, model, stages, likelihood, seed, device, stage_offset=0):
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=fork_devices(device)), torch.no_grad():
            for stage_index, stage in enumerate(stages):
                torch.manual_seed(seed + 92000 + stage_index + stage_offset)
                loader = DataLoader(stage.dataset, batch_size=args.batch_size, shuffle=False,
                                    generator=torch.Generator().manual_seed(seed + 92100 + stage_index + stage_offset))
                totals = {'reconstruction': 0.0, 'kl': 0.0, 'mse': 0.0}; count = 0
                for batch in loader:
                    images = batch['image'].to(device)
                    losses, mse, _ = losses_for(entry, models, args, model, images, likelihood)
                    n = len(images); count += n
                    for name, value in dict(losses, mse=mse).items():
                        totals[name] += float(value) * n
                means = {k: v / count for k, v in totals.items()}
                if any(not math.isfinite(v) for v in means.values()):
                    raise FloatingPointError('Nonfinite validation loss')
                rows.append(dict(stage=stage.name, samples=count, **means))
    finally:
        model.train(was_training)
    return rows


def signal_state(signal):
    return checkpoint_state(signal)


def save_checkpoint(path, model, optimizer, controller, controller_optimizer, monitor, step, unit, cursor):
    if Path(path).exists():
        raise FileExistsError(str(path))
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'controller': controller.state_dict() if controller_optimizer else {},
               'controller_optimizer': controller_optimizer.state_dict() if controller_optimizer else {},
               'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'python_rng': random.getstate(), 'signal': signal_state(monitor.signal),
               'step': step, 'unit': unit, 'data_cursor': cursor,
               'interrupted_resume_supported': False}
    temporary = Path(str(path) + '.partial')
    torch.save(payload, temporary); temporary.replace(path)
    return sha_file(path)


def run_unit(unit, config, historical, bundle, split_sha, device, root, fingerprint, planned_index=0, planned_total=240):
    name = unit_id(unit); out = root / name
    if out.exists():
        summary_file = out / 'run_summary.json'
        if summary_file.exists():
            old = json.loads(summary_file.read_text())
            if old['status'] == 'complete' and old['fingerprint'] == fingerprint and old['split_sha256'] == split_sha:
                if sha_file(out / 'checkpoint_final.pt') != old['checkpoint_sha256']:
                    raise ValueError('Existing final checkpoint hash changed')
                print('复用已完成 run: ' + name, flush=True)
                return old
        raise FileExistsError('保留已有失败/未完成 run；使用新的 BWS_EXPERIMENT_ID: ' + str(out))
    out.mkdir()
    started = time.monotonic(); step = 0; updates = 0; cursor = {}; model = None; monitor = None
    write_json(out / 'status.json', {'status': 'running', 'unit': unit, 'updates': 0})
    write_json(root / 'progress.json', {'phase': 'initializing_run', 'unit': unit, 'completed_runs': planned_index,
               'total_runs': planned_total, 'step': 0, 'total_steps': sum(config['stage_steps']), 'run_started_at': time.time()})
    try:
        entry = historical.load('train_xib_unified'); models = historical.load('models')
        utils = historical.load('utils'); data = historical.load('data')
        args = args_for(entry, config, unit['dataset'])
        utils.set_seed(unit['seed'])
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model, likelihood = entry.build_model(args, unit['latent_dim'], device)
        initial_sha = tensor_digest(model)
        references = initial_references(entry, models, args, model, bundle.train_stages[0], likelihood,
                                        unit['seed'], device, config['calibration_batches'])
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        controller, controller_optimizer = make_controller(unit, references, config, device)
        if controller_optimizer:
            assert_disjoint_optimizers(optimizer, controller_optimizer)
        monitor = calibrate_monitor(args, model, entry, models, likelihood, bundle, unit,
                                    args.manual_signal_policy, out, device)
        resolved = {'unit': unit, 'config': config, 'source_defaults': vars(args),
                    'initial_model_sha256': initial_sha, 'split_sha256': split_sha,
                    'initial_training_references': references, 'fingerprint': fingerprint,
                    'Psi_config': monitor.config, 'Psi_config_sha256': monitor.config_sha256,
                    'Psi_source_sha256': monitor.source_sha256, 'device': str(device),
                    'shared_parameters': [n for n, _ in model.encoder.named_parameters()],
                    'step_state_id': 'SHA256 of actual pre-update model tensors; not a materialized checkpoint file',
                    'checkpoint_policy': 'initial and final full checkpoints; every step has state hash and batch IDs',
                    'scientific_scope': 'manual v0.3 signal migration development; not full Q1-Q3 confirmation'}
        write_json(out / 'resolved_config.json', json.loads(json.dumps(resolved, default=str)))
        save_checkpoint(out / 'checkpoint_initial.pt', model, optimizer, controller, controller_optimizer, monitor, 0, unit, {})
        order = hashlib.sha256(); model.train()
        for stage_index, (stage, steps) in enumerate(zip(bundle.train_stages, args.stage_steps)):
            iterator = entry.cycle(data.make_loader(stage.dataset, args.batch_size, unit['seed'] + stage_index * 10007))
            for local_step in range(1, steps + 1):
                step += 1
                batch = next(iterator); images = batch['image'].to(device)
                ids = list(batch['sample_id'])
                order.update(json.dumps(ids, separators=(',', ':')).encode())
                pre_sha = tensor_digest(model)
                pre_rng = rng_digest(device)
                state_id = name + ':pre_' + str(step) + ':' + hashlib.sha256((pre_sha + pre_rng).encode()).hexdigest()
                losses, mse, reconstructed = losses_for(entry, models, args, model, images, likelihood)
                if any(float(v.detach()) < 0 or not bool(torch.isfinite(v.detach())) for v in losses.values()):
                    raise FloatingPointError('Invalid raw loss')
                weights = weights_for(controller, losses)
                if any(v < config['weight_limits'][0] or v > config['weight_limits'][1] for v in weights.values()):
                    raise FloatingPointError('Weight outside declared numerical bounds')
                norms, cosines = collect_gradient_vectors(losses, model.encoder.parameters(), weights)
                norm_tensor = torch.tensor([r['raw_norm'] for r in norms], device=device)
                # Observe then record the detached signal before any model update.
                monitor.update(images.detach(), reconstructed.detach(), step, name, state_id)
                objective, balancing = checked_update(model, optimizer, controller, controller_optimizer, losses, norm_tensor)
                updates += 1
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite updated model parameters')
                current_weights = weights_for(controller, losses)
                if any(not math.isfinite(v) or v < config['weight_limits'][0] or v > config['weight_limits'][1]
                       for v in current_weights.values()):
                    raise FloatingPointError('Updated weight outside declared numerical bounds')
                base = {'run_id': name, 'step': step, 'stage_index': stage_index + 1,
                        'stage': stage.name, 'checkpoint_id': state_id}
                append(out / 'metrics.csv', dict(base, objective=objective,
                       reconstruction=float(losses['reconstruction'].detach()), kl=float(losses['kl'].detach()),
                       reconstruction_mse=float(mse.detach()), gradnorm_objective=balancing))
                append(out / 'weights.csv', dict(base, **weights))
                for row in norms:
                    append(out / 'gradients.csv', dict(base, **row))
                for row in cosines:
                    append(out / 'gradient_cosines.csv', dict(base, **row))
                with (out / 'batches.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(base, sample_ids=ids, pre_model_sha256=pre_sha, pre_rng_sha256=pre_rng)) + '\n')
                cursor = {'stage_index': stage_index, 'local_step': local_step}
                if step % 10 == 0 or local_step == 1:
                    elapsed = time.monotonic() - started
                    speed = step / elapsed if elapsed > 0 else 0
                    write_json(root / 'progress.json', {'phase': 'training', 'unit': unit, 'completed_runs': planned_index,
                               'total_runs': planned_total, 'step': step, 'total_steps': sum(args.stage_steps),
                               'stage': stage.name, 'steps_per_second': speed, 'elapsed_run_seconds': elapsed,
                               'remaining_training_step_seconds': (sum(args.stage_steps) - step) / speed if speed else None,
                               'updated_at': time.time()})
                if local_step % 200 == 0:
                    values = validation(entry, models, args, model, [bundle.probe_stages[stage_index]],
                                        likelihood, unit['seed'], device, stage_offset=stage_index)
                    for row in values:
                        append(out / 'validation.csv', dict(run_id=name, step=step, evaluation='online_current_stage',
                               checkpoint_id=name + ':post_' + str(step) + ':' + tensor_digest(model), **row))
                if step % 100 == 0:
                    write_json(out / 'status.json', {'status': 'running', 'unit': unit, 'updates': step})
                    print(name + f' step={step}/3000 rec={float(losses["reconstruction"].detach()):.6g}', flush=True)
                    print('总体 ' + bar(planned_index, planned_total), flush=True)
                    print('当前 ' + bar(step, sum(args.stage_steps)) + ' | 本次训练步剩余 ' + duration((sum(args.stage_steps)-step) / speed if speed else None), flush=True)
        write_json(root / 'progress.json', {'phase': 'validation_and_probe', 'unit': unit,
                   'completed_runs': planned_index, 'total_runs': planned_total, 'step': step,
                   'total_steps': sum(args.stage_steps), 'updated_at': time.time()})
        monitor.close()
        val_rows = validation(entry, models, args, model, bundle.probe_stages, likelihood, unit['seed'], device)
        for row in val_rows:
            append(out / 'validation.csv', dict(run_id=name, step=step, evaluation='final_all_stages',
                   checkpoint_id=name + ':post_' + str(step) + ':' + tensor_digest(model), **row))
        primary = sum(row['reconstruction'] for row in val_rows) / len(val_rows)
        probe_started = time.monotonic()
        if config['final_probe']:
            with torch.random.fork_rng(devices=fork_devices(device)):
                for index, stage in enumerate(bundle.probe_stages):
                    entry.run_probe(args, model, stage, index, step, device, out, unit['seed'])
        checkpoint_sha = save_checkpoint(out / 'checkpoint_final.pt', model, optimizer, controller,
                                          controller_optimizer, monitor, step, unit, cursor)
        result = dict(unit, status='complete', fingerprint=fingerprint, updates=step,
                      initial_model_sha256=initial_sha, split_sha256=split_sha,
                      data_order_sha256=order.hexdigest(), checkpoint_sha256=checkpoint_sha,
                      final_model_sha256=tensor_digest(model), validation_reconstruction=primary,
                      validation_stages=val_rows, wall_seconds=time.monotonic() - started,
                      final_probe_seconds=time.monotonic() - probe_started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == 'darwin' else 1),
                      B_detection_metrics_evaluated=False, final_test_used=False, run_dir=str(out))
        write_json(out / 'run_summary.json', result)
        write_json(out / 'status.json', {'status': 'complete', 'updates': step})
        write_json(root / 'progress.json', {'phase': 'run_complete', 'unit': unit,
                   'completed_runs': planned_index + 1, 'total_runs': planned_total, 'step': step,
                   'total_steps': sum(args.stage_steps), 'updated_at': time.time()})
        return result
    except BaseException as exc:
        if monitor is not None:
            monitor.close()
        write_json(root / 'progress.json', {'phase': 'failed', 'unit': unit, 'completed_runs': planned_index,
                   'total_runs': planned_total, 'step': step, 'total_steps': sum(config['stage_steps']),
                   'error': repr(exc), 'updated_at': time.time()})
        write_json(out / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                   'updates_attempted': step, 'model_updates_completed': updates, 'cursor': cursor, 'error': repr(exc), 'traceback': traceback.format_exc()})
        if model is not None:
            torch.save({'model': model.state_dict(), 'updates_attempted': step, 'unit': unit}, out / 'failed_model.pt')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['plan', 'train'], default='plan')
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    options = parser.parse_args()
    if not options.experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in options.experiment_id):
        raise ValueError('Unsafe experiment ID')
    config = load_config()
    if options.action == 'plan':
        print(json.dumps(plan(config), ensure_ascii=False, indent=2)); return
    fingerprint = code_digest() + ':' + sha_file(CONFIG)
    from ..manual_v03.contract import require_author_decisions
    require_author_decisions(ROOT)
    gate = json.loads((ROOT / 'results/preflight.json').read_text())
    if gate.get('status') != 'passed' or gate.get('fingerprint') != fingerprint:
        raise ValueError('Run preflight for this exact code/config before training')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from check_data import hash_inputs
    actual_data = hash_inputs()
    if actual_data['data_fingerprint'] != gate['data']['data_fingerprint']:
        raise ValueError('Dataset changed since preflight')
    fingerprint += ':' + actual_data['data_fingerprint']
    torch.set_num_threads(int(os.environ.get('BWS_CPU_THREADS', '4')))
    historical = HistoricalSource(ROOT / 'legacy/03_psi_xib_unified')
    device = historical.load('utils').resolve_device(options.device)
    root = ROOT / 'results/generated' / options.experiment_id
    root.mkdir(parents=True, exist_ok=True)
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                   'cuda_runtime': torch.version.cuda, 'device': str(device),
                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'torch_threads': torch.get_num_threads(), 'fingerprint': fingerprint,
                   'data_fingerprint': actual_data['data_fingerprint']}
    environment_path = root / 'environment.json'
    if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
        raise ValueError('Existing experiment environment changed')
    if not environment_path.exists():
        write_json(environment_path, environment)
    units = planned_units(config); summaries = []; pair_checks = {}
    write_json(root / 'plan.json', plan(config))
    current_key = None; bundle = None
    try:
        for planned_index, unit in enumerate(units):
            key = (unit['dataset'], unit['seed'])
            if key != current_key:
                write_json(root / 'progress.json', {'phase': 'preparing_data', 'unit': unit,
                           'completed_runs': len(summaries), 'total_runs': len(units),
                           'step': 0, 'total_steps': sum(config['stage_steps']), 'updated_at': time.time()})
                # Release image caches from the previous pair before loading the next.
                bundle = None
                args = args_for(historical.load('train_xib_unified'), config, unit['dataset'])
                bundle, split_sha = build_disjoint_bundle(args, unit['seed'], historical, ROOT / 'results/assets')
                current_key = key
            result = run_unit(unit, config, historical, bundle, split_sha, device, root, fingerprint, planned_index, len(units))
            pair_key = (unit['dataset'], unit['latent_dim'], unit['seed'])
            check = tuple(result[k] for k in ['initial_model_sha256', 'split_sha256', 'data_order_sha256'])
            if pair_key in pair_checks and pair_checks[pair_key] != check:
                raise ValueError('Paired-arm initialization, split or order mismatch')
            pair_checks[pair_key] = check
            summaries.append(result)
            write_json(root / 'ledger.json', {'status': 'running', 'planned': len(units),
                       'completed': len(summaries), 'failed': 0, 'incomplete': len(units) - len(summaries),
                       'remaining': len(units) - len(summaries), 'runs': summaries})
        write_json(root / 'ledger.json', {'status': 'complete', 'planned': len(units),
                   'completed': len(summaries), 'failed': 0, 'incomplete': 0, 'remaining': 0, 'runs': summaries})
    except BaseException as exc:
        failure_count = 0 if isinstance(exc, KeyboardInterrupt) else 1
        write_json(root / 'ledger.json', {'status': 'incomplete', 'planned': len(units),
                   'completed': len(summaries), 'failed': failure_count,
                   'incomplete': len(units) - len(summaries) - failure_count,
                   'remaining': len(units) - len(summaries), 'runs': summaries,
                   'failures': [{'unit': unit, 'status': 'incomplete' if not failure_count else 'failed',
                                 'error': repr(exc), 'raw_status_path': str(root / unit_id(unit) / 'status.json')}]})
        raise
    print('训练矩阵完成：' + str(root), flush=True)


if __name__ == '__main__':
    main()
