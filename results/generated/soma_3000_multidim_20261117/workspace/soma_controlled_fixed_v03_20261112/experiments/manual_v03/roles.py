"""Proposed roles: reserve white/edge/edge-cover, then split remaining IDs 70/30.

This changes the earlier split and requires a recorded author decision. Sampling
with replacement within a role never increases the count of unique source IDs.
"""
from types import SimpleNamespace
import copy
import hashlib
import json
import numpy as np
from .signal import validate_roles

ROLES = ('white', 'edge', 'edge-cover')


def partition_source(total, stable_ids, seed, fractions):
    stable_ids = np.unique(np.asarray(stable_ids, dtype=np.int64))
    rng = np.random.default_rng(seed + 630001)
    shuffled = rng.permutation(stable_ids)
    roles = np.full(total, 'unassigned', dtype='<U12')
    offset = 0
    for name in ROLES:
        count = int(len(stable_ids) * fractions[name])
        if count < 2:
            raise ValueError('Insufficient distinct stable IDs for ' + name)
        roles[shuffled[offset:offset + count]] = name
        offset += count
    remaining = rng.permutation(np.flatnonzero(roles == 'unassigned'))
    boundary = int(.7 * len(remaining))
    roles[remaining[:boundary]] = 'train'
    roles[remaining[boundary:]] = 'validation'
    return roles


def draws(pool, count, seed, allow_replacement):
    if not len(pool):
        raise ValueError('Empty role pool')
    if count > len(pool) and not allow_replacement:
        raise ValueError(f'Need {count} distinct calibration IDs, only {len(pool)} available')
    return np.random.default_rng(seed).choice(pool, count, replace=count > len(pool)).astype(np.int64)


def attach_calibration(bundle, args, entry, seed, policy, StageSpec, ControlledRows,
                       ImageRows, pools=None, masks=None, readers=None, labelers=None, families=None):
    counts = {'white': policy['white_samples'],
              'edge': policy['numeric']['window'] * policy['edge_windows'],
              'edge-cover': policy['numeric']['window'] * policy['edge_cover_windows']}
    calibration = {}
    metadata = {}
    if args.dataset == 'controlled':
        original = bundle.train_stages[0].dataset.original
        for index, role in enumerate(ROLES):
            source = copy.copy(original)
            source.seed = int(seed) + 16000001 + index * 100003
            source.size = counts[role]
            calibration[role] = ControlledRows(source, 'controlled:' + str(source.seed))
            metadata[role] = dict(rows=counts[role], unique_source_ids=counts[role],
                                  generator_seed=source.seed, replacement=False)
    else:
        for index, role in enumerate(ROLES):
            pool = np.intersect1d(pools[0], np.flatnonzero(masks[0] == role))
            ids = draws(pool, counts[role], seed + 640001 + index, policy['allow_within_role_replacement'])
            calibration[role] = ImageRows(readers[0], labelers[0], ids, families[0])
            metadata[role] = dict(rows=counts[role], unique_source_ids=int(len(np.unique(ids))),
                                  pool_unique_source_ids=len(pool), replacement=counts[role] > len(pool))
    role_source_ids = {}
    for name, stages in [('train', bundle.train_stages), ('validation', bundle.probe_stages)]:
        ids = []
        for stage in stages:
            ds = stage.dataset
            if hasattr(ds, 'ids'):
                ids += [ds.family + ':' + str(int(i)) for i in ds.ids]
            else:
                ids += [ds.family + ':' + str(i) for i in range(len(ds))]
        role_source_ids[name] = ids
    all_ids = dict(role_source_ids)
    for name, dataset in calibration.items():
        all_ids[name] = [dataset.family + ':' + str(int(i)) for i in dataset.ids] if hasattr(dataset, 'ids') else [dataset.family + ':' + str(i) for i in range(len(dataset))]
    validate_roles(all_ids)
    manifest = dict(policy='controlled_independent_calibration_streams_v03' if args.dataset == 'controlled' else 'reserved_calibration_then_70_30_v03_draft', roles=all_ids,
                    calibration=metadata, intersections='empty', final_test_used=False,
                    within_role_replacement_does_not_add_independent_source_images=True)
    return SimpleNamespace(train_stages=bundle.train_stages, probe_stages=bundle.probe_stages,
                           calibration_roles=calibration, role_source_ids=role_source_ids,
                           role_manifest=manifest)
