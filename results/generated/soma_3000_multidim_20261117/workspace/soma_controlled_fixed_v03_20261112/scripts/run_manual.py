"""One entry point. Drafts can test; a full rerun requires explicit recorded choices."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.b_weight_sensitivity.formal_common import load_config, plan, code_digest, sha_file, CONFIG, write_json
from experiments.manual_v03.contract import require_author_decisions


def execute(arguments):
    subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, check=True,
                   env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                            MPLCONFIGDIR=str(ROOT / '.cache/matplotlib')))


def preflight():
    require_author_decisions(ROOT)
    execute(['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'])
    from check_data import hash_inputs
    from experiments.b_weight_sensitivity.historical_loader import HistoricalSource
    from experiments.manual_v03.readout import FrozenReadout
    import torch
    HistoricalSource(ROOT / 'legacy/03_psi_xib_unified').verify()
    policy = json.loads((ROOT / 'configs/signal.json').read_text())
    training = load_config()
    for dataset in training['datasets']:
        specification = policy['readouts'][dataset]
        FrozenReadout(dataset, specification, torch.device('cpu'))
    fixtures = sorted((ROOT / 'results/engineering').glob('*/fixture_summary.json'))
    suitable = [json.loads(path.read_text()) for path in fixtures]
    if not any(f.get('code_sha256') == code_digest() and all(r.get('monitor_does_not_change_model_updates') for r in f['runs_detail']) for f in suitable):
        raise ValueError('A passing read-only model-update fixture is required')
    gate = dict(status='passed', fingerprint=code_digest() + ':' + sha_file(CONFIG),
                data=hash_inputs(), timestamp=time.time(), scope='approved development matrix; no Q1-Q3 final test')
    write_json(ROOT / 'results/preflight.json', gate)
    print('Preflight passed for exact current code/config/data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'test', 'smoke', 'roles', 'preflight', 'train', 'analyze', 'all'])
    parser.add_argument('--experiment-id', default='manual03_' + time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    options = parser.parse_args()
    if options.action == 'plan':
        p = plan(load_config())
        try:
            require_author_decisions(ROOT); p['migration_decisions'] = 'recorded'
        except ValueError as exc:
            p['migration_decisions'] = str(exc)
        p['signal_config_sha256'] = sha_file(ROOT / 'configs/signal.json')
        p['scope'] = 'controlled signal development with historical fixed weights; no weight search, no final Q1-Q3 test'
        print(json.dumps(p, ensure_ascii=False, indent=2)); return
    if options.action == 'test':
        execute(['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']); return
    if options.action == 'smoke':
        execute(['scripts/smoke_manual.py', '--experiment-id', options.experiment_id]); return
    if options.action == 'roles':
        execute(['scripts/audit_roles.py']); return
    if options.action in ('preflight', 'all'):
        preflight()
    if options.action in ('train', 'all'):
        require_author_decisions(ROOT)
        execute(['-m', 'experiments.b_weight_sensitivity.formal_runner', '--action', 'train',
                 '--experiment-id', options.experiment_id, '--device', options.device])
    if options.action in ('analyze', 'all'):
        execute(['scripts/analyze_manual.py', '--experiment-id', options.experiment_id])


if __name__ == '__main__':
    main()
