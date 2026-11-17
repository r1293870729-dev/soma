"""Independent linear probes. No main-model optimizer or online signal access."""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F


def array_hash(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def assert_disjoint(roles):
    seen = set()
    for name, ids in roles.items():
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping IDs: ' + name)
        seen.update(ids)


def metrics(logits, labels, classes):
    prediction = logits.argmax(1)
    recalls, f1s = [], []
    for c in range(classes):
        true = labels == c
        pred = prediction == c
        tp = int((true & pred).sum())
        support = int(true.sum())
        if not support:
            raise ValueError('Declared class absent from probe role')
        recalls.append(tp / support)
        denominator = int(true.sum()) + int(pred.sum())
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return dict(accuracy=float((prediction == labels).double().mean()),
                balanced_accuracy=float(np.mean(recalls)), macro_f1=float(np.mean(f1s)),
                cross_entropy=float(F.cross_entropy(logits, labels)))


def fit_head(features, labels, classes, policy):
    """Fit only train, choose lambda only val, then evaluate test exactly once."""
    mean = features['probe-train'].mean(0)
    std = features['probe-train'].std(0, unbiased=False).clamp_min(1e-6)
    x = {role: (v.double() - mean) / std for role, v in features.items()}
    y = {}
    for role, values in labels.items():
        mapping = {old: new for new, old in enumerate(classes)}
        if set(values.tolist()) != set(classes):
            raise ValueError('Probe class support differs from frozen protocol')
        y[role] = torch.tensor([mapping[int(v)] for v in values], dtype=torch.long)
    candidates = []
    for penalty in policy['l2_grid']:
        weight = torch.zeros((x['probe-train'].shape[1], len(classes)), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros(len(classes), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=policy['max_iter'],
            tolerance_grad=policy['tolerance_grad'], tolerance_change=policy['tolerance_change'],
            line_search_fn='strong_wolfe')
        evaluations = 0
        def closure():
            nonlocal evaluations
            evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            value = F.cross_entropy(x['probe-train'] @ weight + bias, y['probe-train']) + float(penalty) * weight.square().sum() / 2
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite probe objective')
            value.backward()
            return value
        optimizer.step(closure)
        closure()
        grad_max = max(float(weight.grad.abs().max()), float(bias.grad.abs().max()))
        with torch.no_grad():
            val = metrics(x['probe-val'] @ weight + bias, y['probe-val'], len(classes))
        candidates.append(dict(l2=penalty, val=val, weight=weight.detach().clone(), bias=bias.detach().clone(),
                               gradient_max=grad_max, evaluations=evaluations,
                               iterations=optimizer.state[weight]['n_iter']))
    selected = min(range(len(candidates)), key=lambda i: candidates[i]['val']['cross_entropy'])
    chosen = candidates[selected]
    with torch.no_grad():
        test = metrics(x['probe-test'] @ chosen['weight'] + chosen['bias'], y['probe-test'], len(classes))
    public = [{k: v for k, v in c.items() if k not in ('weight', 'bias')} for c in candidates]
    state = {k: chosen[k] for k in ('weight', 'bias')}
    state.update(mean=mean, std=std, classes=classes, l2=chosen['l2'])
    return dict(**test, selected_l2=chosen['l2'], validation_cross_entropy=chosen['val']['cross_entropy'],
                gradient_max=chosen['gradient_max'], candidates=public), state


def build_probe_data(entry, bundle, args, seed, protocol):
    roles, metadata, ids = {}, [], {}
    for stage, source in enumerate(bundle.train_stages):
        original = source.dataset.original
        roles[stage] = {}
        for offset, (role, count) in enumerate(protocol['probe_roles'].items()):
            ds = copy.copy(original)
            ds.seed = protocol['probe_namespace_base'] + seed * 1000000 + stage * 100003 + offset * 10007
            ds.size = count
            samples = [ds[i] for i in range(count)]
            x = torch.stack([r['image'] for r in samples])
            y = torch.stack([r['labels'] for r in samples])
            key = str(stage) + ':' + role
            role_ids = ['controlled:' + str(ds.seed) + ':' + str(i) for i in range(count)]
            ids[key] = role_ids
            roles[stage][role] = (x, y)
            metadata.append(dict(stage=stage, role=role, generator_seed=ds.seed, rows=count,
                observation_sha256=array_hash(x.numpy()), labels_sha256=array_hash(y.numpy()),
                sample_ids_sha256=hashlib.sha256(json.dumps(role_ids).encode()).hexdigest(),
                class_counts={str(c): torch.bincount(y[:, c]).tolist() for c in range(3)}))
    all_ids = dict(bundle.role_manifest['roles'])
    all_ids.update(ids)
    assert_disjoint(all_ids)
    return roles, dict(metadata=metadata, roles=all_ids, disjoint=True, development_probe_test=True,
                       final_test_stream_used=False)


def evaluate_snapshot(model, data, protocol, step):
    encoder = copy.deepcopy(model).eval()
    rows, head_states = [], {}
    with torch.no_grad():
        features = {stage: {role: encoder.encode(values[0])[0].double().detach()
                    for role, values in role_data.items()}
                    for stage, role_data in data.items()
                    if any(t['stage'] == stage and step >= t['first_report_step'] for t in protocol['tasks'])}
    for task in protocol['tasks']:
        if step < task['first_report_step']:
            continue
        stage, name = task['stage'], task['name']
        labels = {role: values[1][:, task['column']] for role, values in data[stage].items()}
        result, state = fit_head(features[stage], labels, task['classes'], protocol['head'])
        candidates = result.pop('candidates')
        rows.append(dict(step=step, updates_completed=step-1, snapshot_phase='pre-update',
                         probe_stage=stage+1, task=name, classes=len(task['classes']),
                         **result, samples_train=protocol['probe_roles']['probe-train'],
                         samples_val=protocol['probe_roles']['probe-val'],
                         samples_test=protocol['probe_roles']['probe-test']))
        head_states[str(stage+1) + ':' + name] = dict(**state, candidates=candidates)
    return rows, head_states


def first_sustained_drop(rows, specification):
    lo, hi = specification['baseline_window']
    baseline_values = [r['balanced_accuracy'] for r in rows if lo <= r['step'] <= hi]
    expected = (hi-lo)//50 + 1
    if len(baseline_values) != expected:
        raise ValueError('Incomplete baseline grid')
    baseline = float(np.median(baseline_values))
    threshold = baseline - specification['drop_absolute']
    count, onset = 0, None
    for r in rows:
        if r['step'] <= hi:
            continue
        if r['balanced_accuracy'] <= threshold:
            count += 1
            if count == 1:
                onset = r['step']
            if count == specification['persistence_checks']:
                return dict(status='sustained_drop', baseline=baseline, threshold=threshold,
                            onset_step=onset, confirmation_step=r['step'])
        else:
            count, onset = 0, None
    return dict(status='no_sustained_drop', baseline=baseline, threshold=threshold,
                onset_step=None, confirmation_step=None)


def auc(labels, values):
    labels, values = np.asarray(labels, bool), np.asarray(values, float)
    if not labels.any() or labels.all():
        return None
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    positive = int(labels.sum()); negative = len(labels)-positive
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))
