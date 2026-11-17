"""Approved disjoint data adapter; historical models, losses and B stay unchanged."""
from __future__ import annotations
import hashlib
import json
import shutil
import zipfile
import zlib
from ..manual_v03.roles import partition_source, attach_calibration
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def role_mask(total, seed):
    """One global partition per source, shared by all stages and corruptions."""
    order = np.random.default_rng(int(seed) + 910_003).permutation(int(total))
    mask = np.zeros(int(total), dtype=bool)
    mask[order[:int(0.7 * total)]] = True
    return mask


def select_ids(valid, mask, train, count, seed):
    pool = np.asarray(valid, dtype=np.int64)
    pool = pool[mask[pool] == ('train' if train else 'validation')]
    if not len(pool):
        raise ValueError('Empty disjoint data role')
    return np.random.default_rng(int(seed)).choice(pool, int(count), replace=count > len(pool)).astype(np.int64)


class ImageRows(Dataset):
    def __init__(self, images, labels, ids, family):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.family = str(family)
        unique, inverse = np.unique(self.ids, return_inverse=True)
        self.images = torch.from_numpy(np.asarray(images(unique)).copy())
        self.lookup = torch.from_numpy(inverse.astype(np.int64))
        self.labels = torch.from_numpy(np.asarray(labels(self.ids), dtype=np.int64).copy())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = self.images[self.lookup[index]].float()
        if image.ndim == 2:
            image = image.unsqueeze(0)  # dSprites binary uint8 is already 0/1.
        else:
            image = image.permute(2, 0, 1).contiguous() / 255.0
        return {'image': image, 'labels': self.labels[index],
                'sample_id': self.family + ':' + str(int(self.ids[index]))}


class ControlledRows(Dataset):
    def __init__(self, original, family):
        self.original, self.family = original, family

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        return dict(self.original[index], sample_id=self.family + ':' + str(index))


def dsprites_arrays(path, cache):
    """Extract only numeric imgs.npy; never access the NPZ pickle metadata."""
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / 'dsprites_imgs.npy'
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('imgs.npy')
        reused = destination.exists()
        if not destination.exists():
            if shutil.disk_usage(cache).free < info.file_size + 512 * 1024**2:
                raise OSError('Insufficient space for numeric dSprites mmap cache')
            partial = destination.with_suffix('.npy.partial')
            with archive.open(info) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            if partial.stat().st_size != info.file_size:
                raise OSError('Incomplete dSprites extraction')
            partial.replace(destination)
        if destination.stat().st_size != info.file_size:
            raise ValueError('dSprites numeric cache size mismatch')
        if reused:
            crc = 0
            with destination.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    crc = zlib.crc32(block, crc)
            if crc != info.CRC:
                raise ValueError('dSprites numeric cache CRC changed')
    with np.load(path, allow_pickle=False) as data:
        classes = np.asarray(data['latents_classes'], dtype=np.int64)
    images = np.load(destination, allow_pickle=False, mmap_mode='r')
    if images.shape != (737280, 64, 64) or classes.shape != (737280, 6):
        raise ValueError('Unexpected dSprites shape')
    return images, classes


def h5_rows(path, indices):
    import h5py
    indices = np.asarray(indices, dtype=np.int64)
    pieces = []
    with h5py.File(path, 'r') as handle:
        for block in np.unique(indices // 4096):
            start = int(block) * 4096
            wanted = indices[(indices // 4096) == block]
            data = np.asarray(handle['images'][start:start + 4096], dtype=np.uint8)
            pieces.append(data[wanted - start])
    return np.concatenate(pieces, axis=0)


def build_disjoint_bundle(args, seed, historical, asset_root):
    """Reuse historical stage/factor definitions with approved global ID roles."""
    policy = args.manual_signal_policy
    asset_root = Path(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    StageSpec = historical.load('data').StageSpec
    entry = historical.load('train_xib_unified')
    if args.dataset == 'controlled':
        bundle = entry.build_controlled_bundle(args, seed)
        train, val, records = [], [], []
        for index, (a, b) in enumerate(zip(bundle.train_stages, bundle.probe_stages)):
            train.append(StageSpec(a.name, ControlledRows(a.dataset, f'controlled:{a.dataset.seed}'), a.active_tasks, a.num_classes))
            val.append(StageSpec(b.name, ControlledRows(b.dataset, f'controlled:{b.dataset.seed}'), b.active_tasks, b.num_classes))
            records.append({'stage': a.name, 'train_generator_seed': a.dataset.seed,
                            'validation_generator_seed': b.dataset.seed,
                            'train_ids': list(range(len(a.dataset))), 'validation_ids': list(range(len(b.dataset))),
                            'role_definition': 'independent seeded historical generator streams'})
    else:
        if args.dataset == 'dsprites':
            images, classes = dsprites_arrays(args.dsprites_path, asset_root / 'numeric_cache')
            names = ['S1_position', 'S2_position_shape', 'S3_position_shape_orientation']
            active = [('x', 'y'), ('x', 'y', 'shape'), ('x', 'y', 'shape', 'orientation')]
            num_classes = {'x': 32, 'y': 32, 'shape': 3, 'orientation': 40}
            common = classes[:, 2] == 3
            pools = [np.flatnonzero(common & (classes[:, 1] == 0) & (classes[:, 3] == 0)),
                     np.flatnonzero(common & (classes[:, 3] == 0)), np.flatnonzero(common)]
            masks = [partition_source(len(classes), pools[0], seed, policy['calibration_role_fractions'])] * 3
            readers = [lambda idx: images[idx]] * 3
            labelers = [lambda idx: classes[idx][:, [4, 5, 1, 3]]] * 3
            families = ['dsprites'] * 3
        elif args.dataset == 'shapes3d':
            import h5py
            with h5py.File(args.shapes3d_path, 'r') as handle:
                raw = np.asarray(handle['labels'])
            classes = np.stack([np.searchsorted(np.unique(raw[:, c]), raw[:, c]) for c in range(6)], axis=1)
            names = ['S1_background', 'S2_background_object', 'S3_background_object_shape_pose']
            factors = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
            num_classes = dict(zip(factors, (10, 10, 10, 8, 4, 15)))
            active = [factors[:2], factors[:4], factors]
            pools = [np.flatnonzero((classes[:, 2] == 0) & (classes[:, 3] == 4) & (classes[:, 4] == 0) & (classes[:, 5] == 0)),
                     np.flatnonzero((classes[:, 4] == 0) & (classes[:, 5] == 0)), np.arange(len(classes))]
            masks = [partition_source(len(classes), pools[0], seed, policy['calibration_role_fractions'])] * 3
            readers = [lambda idx: h5_rows(args.shapes3d_path, idx)] * 3
            labelers = [lambda idx: classes[idx]] * 3
            families = ['shapes3d'] * 3
        else:
            from torchvision.datasets import CIFAR10
            source = CIFAR10(str(args.cifar10_root), train=True, download=False)
            clean_images, clean_labels = source.data, np.asarray(source.targets)
            corruptions = [np.load(args.cifarc_root / (s + '.npy'), allow_pickle=False, mmap_mode='r') for s in ['motion_blur', 'frost']]
            labels = np.load(args.cifarc_root / 'labels.npy', allow_pickle=False, mmap_mode='r')
            if any(x.shape != (50000, 32, 32, 3) for x in corruptions) or labels.shape != (50000,):
                raise ValueError('Unexpected CIFAR-C arrays')
            names = ['S1_cifar10_clean', 'S2_motion_blur_s3', 'S3_frost_s3']
            active = [('class',), ('class', 'domain'), ('class', 'domain')]
            num_classes = {'class': 10, 'domain': 3}
            pools = [np.arange(50000), np.arange(10000), np.arange(10000)]
            masks = [partition_source(50000, pools[0], seed, policy['calibration_role_fractions']),
                     np.where(role_mask(10000, seed), 'train', 'validation'),
                     np.where(role_mask(10000, seed), 'train', 'validation')]
            readers = [lambda idx: clean_images[idx], lambda idx: corruptions[0][20000 + idx], lambda idx: corruptions[1][20000 + idx]]
            labelers = [lambda idx: np.column_stack([clean_labels[idx], np.zeros(len(idx), dtype=np.int64)]),
                        lambda idx: np.column_stack([labels[20000 + idx], np.ones(len(idx), dtype=np.int64)]),
                        lambda idx: np.column_stack([labels[20000 + idx], np.full(len(idx), 2, dtype=np.int64)])]
            families = ['cifar10_train', 'cifar10_test_origin', 'cifar10_test_origin']
        train, val, records = [], [], []
        all_train, all_val = set(), set()
        for index, name in enumerate(names):
            ids_t = select_ids(pools[index], masks[index], True, args.train_samples_per_stage, seed + index * 101)
            ids_v = select_ids(pools[index], masks[index], False, args.probe_samples_per_stage, seed + 50000 + index * 101)
            all_train.update((families[index], int(i)) for i in ids_t)
            all_val.update((families[index], int(i)) for i in ids_v)
            train.append(StageSpec(name, ImageRows(readers[index], labelers[index], ids_t, families[index]), tuple(active[index]), num_classes))
            val.append(StageSpec(name, ImageRows(readers[index], labelers[index], ids_v, families[index]), tuple(active[index]), num_classes))
            records.append({'stage': name, 'family': families[index], 'train_ids': ids_t.tolist(),
                            'validation_ids': ids_v.tolist(), 'train_unique': int(len(np.unique(ids_t))),
                            'validation_unique': int(len(np.unique(ids_v))),
                            'replacement_is_allowed_within_each_role_only': True})
        if all_train & all_val:
            raise ValueError('Cross-stage source-image leakage')
    base = entry.StreamBundle(train, val)
    kwargs = {} if args.dataset == 'controlled' else dict(pools=pools, masks=masks, readers=readers, labelers=labelers, families=families)
    prepared = attach_calibration(base, args, entry, seed, policy, StageSpec, ControlledRows, ImageRows, **kwargs)
    payload = {'role_manifest': prepared.role_manifest, 'dataset': args.dataset, 'seed': seed, 'policy': 'controlled_independent_calibration_streams_v03',
               'final_test_used': False, 'stages': records}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    manifest = asset_root / f'{args.dataset}_seed{seed}_split.json'
    if manifest.exists() and manifest.read_bytes() != encoded:
        raise ValueError('Prepared split manifest changed')
    if not manifest.exists():
        manifest.write_bytes(encoded)
    return prepared, digest
