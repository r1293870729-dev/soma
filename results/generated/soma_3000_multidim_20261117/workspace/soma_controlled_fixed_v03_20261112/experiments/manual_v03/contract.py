"""Fail closed on unresolved choices. This is a protocol check, not a security boundary."""
import hashlib
import json


def content_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_author_decisions(root):
    path = root / 'configs/author_decisions.json'
    if not path.is_file():
        raise ValueError('author-input-needed: signal parameters, reserved roles and image readout remain unapproved; see docs/DECISIONS_NEEDED.md')
    d = json.loads(path.read_text())
    for field in ('author', 'timestamp', 'decision_evidence'):
        if not d.get(field):
            raise ValueError('Missing author decision evidence: ' + field)
    for field in ('scope_development_runs', 'signal_numeric_parameters', 'controlled_independent_role_namespaces',
                  'resource_and_failure_policy'):
        if d.get(field) is not True:
            raise ValueError('Unresolved author decision: ' + field)
    if d.get('signal_config_sha256') != content_hash(root / 'configs/signal.json'):
        raise ValueError('Signal config differs from author decision record')
    if d.get('training_config_sha256') != content_hash(root / 'configs/formal.json'):
        raise ValueError('Training config differs from author decision record')
    training = json.loads((root / 'configs/formal.json').read_text())
    from ..b_weight_sensitivity.formal_common import planned_units
    count = len(planned_units(training))
    if d.get('approved_planned_run_count') != count:
        raise ValueError('Planned run count differs from author decision')
    policy = json.loads((root / 'configs/signal.json').read_text())
    if policy['alarms_enabled'] or policy['final_test_unlocked']:
        raise ValueError('This migration does not unlock sustained alarms or Q1-Q3 final tests')
    if set(training['datasets']) != {'controlled'}:
        raise ValueError('Image experiments deferred by user; no approved stable image data')
    for dataset in training['datasets']:
        config = policy['readouts'][dataset]
        if dataset != 'controlled' and config.get('kind') != 'resnet18':
            raise ValueError('Frozen image readout metadata is incomplete: ' + dataset)
    return d
