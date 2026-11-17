"""Read-only monitor plus replay assets; no model optimizer references."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .readout import FrozenReadout
from .signal import FixedPsi, SignalConfig, validate_roles


class ManualMonitor:
    def __init__(self, signal, readout, out, total_steps, batch_size):
        self.signal, self.readout, self.out = signal, readout, Path(out)
        self.config = asdict(signal.config)
        self.config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.source_sha256 = hashlib.sha256(Path(__file__).with_name('signal.py').read_bytes()).hexdigest()
        self.last_step = 0
        self.residual_path = self.out / 'preupdate_readout_residuals.npy'
        self.residuals = np.lib.format.open_memmap(self.residual_path, mode='w+', dtype=np.float32,
                                                  shape=(total_steps, batch_size, len(signal.mean)))
        self.log = (self.out / 'psi_signal.jsonl').open('x')

    @torch.no_grad()
    def update(self, observations, reconstruction, step, run_id, checkpoint_id):
        if step != self.last_step + 1:
            raise ValueError('Signal must advance once in arrival order')
        residual = self.readout.residual(observations, reconstruction)
        if residual.dtype != np.float32:
            raise ValueError('Replay storage currently requires float32 readout output')
        self.residuals[step - 1] = residual
        row = self.signal.update(residual)
        row.update(run_id=run_id, step=step, checkpoint_id=checkpoint_id,
                   residual_asset=self.residual_path.name, residual_row=step - 1,
                   config_sha256=self.config_sha256,
                   readout=self.readout.specification, residual_convention='h(observation)-h(reconstructed_mean)',
                   display_midrank=None, display_status='not-measured-no-independent-stable-calibration-stream')
        self.log.write(json.dumps(row, allow_nan=False) + '\n')
        self.log.flush()
        if step % 100 == 0:
            self.residuals.flush()
        self.last_step = step
        return row

    def close(self):
        self.residuals.flush()
        if not self.log.closed:
            self.log.close()


def calibrate_monitor(args, model, entry, models, likelihood, bundle, unit, signal_policy, out, device):
    from ..b_weight_sensitivity.formal_runner import losses_for, fork_devices, tensor_digest
    readout_config = signal_policy['readouts'][unit['dataset']]
    readout = FrozenReadout(unit['dataset'], readout_config, device)
    cfg = SignalConfig(**signal_policy['numeric'])
    model_sha = tensor_digest(model)
    saved = {}
    ids = {}
    training_mode = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=fork_devices(device)), torch.no_grad():
            for index, role in enumerate(('white', 'edge', 'edge-cover')):
                torch.manual_seed(unit['seed'] + 771001 + index)
                pieces, role_ids = [], []
                loader = DataLoader(bundle.calibration_roles[role], batch_size=args.batch_size,
                                    shuffle=False, generator=torch.Generator().manual_seed(772001 + index))
                for batch in loader:
                    images = batch['image'].to(device)
                    _, _, reconstructed = losses_for(entry, models, args, model, images, likelihood)
                    pieces.append(readout.residual(images, reconstructed))
                    role_ids += list(batch['sample_id'])
                saved[role] = np.concatenate(pieces)
                ids[role] = role_ids
    finally:
        model.train(training_mode)
    if tensor_digest(model) != model_sha:
        raise ValueError('Reference calibration changed model state')
    validate_roles(dict(ids, train=bundle.role_source_ids['train'], validation=bundle.role_source_ids['validation']))
    signal = FixedPsi.fit(saved['white'], saved['edge'], cfg, ids)
    out = Path(out)
    np.savez_compressed(out / 'calibration_residuals.npz', white=saved['white'], edge=saved['edge'],
                        edge_cover=saved['edge-cover'])
    np.savez_compressed(out / 'fixed_reference.npz', mean=signal.mean, inverse_sqrt=signal.inverse_sqrt,
                        edge=np.array(signal.edge))
    metadata = dict(signal.metadata, reference_sha256=signal.reference_hash(), initial_model_sha256=model_sha,
                    readout=readout_config, edge_cover=signal.coverage(saved['edge-cover']),
                    no_alarm_threshold_fitted=True, final_test_used=False)
    (out / 'calibration.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return ManualMonitor(signal, readout, out, sum(args.stage_steps), args.batch_size)


def checkpoint_state(signal):
    return dict(mean=torch.from_numpy(signal.mean.copy()), inverse_sqrt=torch.from_numpy(signal.inverse_sqrt.copy()),
                edge=signal.edge, ring=torch.tensor(np.asarray(signal.ring)), samples_seen=signal.samples_seen,
                metadata=signal.metadata, reference_sha256=signal.reference_hash())
