"""Standard-library planning and immutable experiment configuration."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/formal.json'


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def code_digest():
    digest = hashlib.sha256()
    files = list((ROOT / 'experiments').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    files += list((ROOT / 'legacy').rglob('*.py')) + [ROOT / 'configs/signal.json']
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text())
    if config['seeds'] != [0, 1, 2] or config['stage_steps'] != [1000, 1000, 1000]:
        raise ValueError('Approved seed/step plan changed; create a new reviewed protocol')
    if config['split_policy'] != 'controlled_independent_calibration_streams_v03' or config['pcgrad']:
        raise ValueError('Unapproved split or PCGrad change')
    if config['metric'] != 'final_validation_reconstruction_loss_mean_over_3_stages':
        raise ValueError('Primary metric changed')
    if config['batch_size'] != 64 or config['learning_rate'] != 0.001:
        raise ValueError('Historical optimizer/batch default changed')
    if config['weight_limits'] != [1e-8, 1e8]:
        raise ValueError('Numerical safety bounds changed')
    return config


def arm_specs(config):
    if config.get('weight_policy') != 'historical_fixed_only' or config.get('fixed_coefficients') != {'reconstruction': 1.0, 'kl': 0.001}:
        raise ValueError('This approved experiment uses rec=1, KL=0.001 only')
    return [('historical_fixed', 'fixed', 0.001)]


def planned_units(config):
    return [dict(dataset=dataset, latent_dim=z, seed=seed, arm=name, method=method, beta=beta)
            for dataset, dimensions in config['datasets'].items()
            for z in dimensions for seed in config['seeds']
            for name, method, beta in arm_specs(config)]


def unit_id(unit):
    return f"{unit['dataset']}_z{unit['latent_dim']}_seed{unit['seed']}_{unit['arm']}"


def plan(config):
    units = planned_units(config)
    return {'status': 'planned_only', 'training_started': False,
            'runs': len(units), 'steps_per_run': sum(config['stage_steps']),
            'total_model_updates': len(units) * sum(config['stage_steps']),
            'arms': [x[0] for x in arm_specs(config)], 'seeds': config['seeds'],
            'datasets': config['datasets'], 'test_unlocked': False, 'pcgrad_enabled': False,
            'alarm_metrics': 'not_evaluated_without_frozen_detection_contract',
            'checkpoint_selection': 'fixed_final_step_3000',
            'primary_metric': None, 'diagnostic_metric': config['metric'],
            'weight_policy': 'historical_fixed_only', 'independent_controlled_generators': 1, 'config_sha256': sha_file(CONFIG)}
