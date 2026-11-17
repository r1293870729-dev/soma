"""Read-only dataset format/integrity checks; no model training or pickle loads here."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def training_files():
    batch = ROOT / 'data/cifar10/cifar-10-batches-py'
    return [batch / ('data_batch_' + str(i)) for i in range(1, 6)] + [batch / 'test_batch', batch / 'batches.meta'] + [
        ROOT / 'data/CIFAR-C' / name for name in ['motion_blur.npy', 'frost.npy', 'labels.npy']] + [
        ROOT / 'data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
        ROOT / 'data/shapes3d/3dshapes.h5']


def hash_inputs(verbose=True):
    selected = json.loads((ROOT / 'configs/formal.json').read_text())['datasets']
    if set(selected) == {'controlled'}:
        generator = ROOT / 'legacy/03_psi_xib_unified/train_xib_unified.py'
        descriptor = {'generator_source_sha256': hashlib.sha256(generator.read_bytes()).hexdigest(),
                      'external_datasets_required': False,
                      'role_policy': 'controlled_independent_calibration_streams_v03'}
        return {'status': 'built_in_controlled_generator', 'files': [], 'generator': descriptor,
                'data_fingerprint': hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()}
    records = []
    for path in training_files():
        if not path.is_file():
            raise FileNotFoundError('缺少必需数据：' + str(path))
        if verbose:
            print('核验输入：' + path.relative_to(ROOT).as_posix(), flush=True)
        value = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(block)
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': value.hexdigest()})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pinned = json.loads((ROOT / 'configs/data_hashes.json').read_text())['files']
    if records != pinned:
        raise ValueError('Dataset bytes differ from the frozen delivery inputs; restore from the package, do not silently change splits/data.')
    return {'status': 'all_required_inputs_present', 'files': records, 'data_fingerprint': digest}


def check():
    import numpy as np
    import h5py
    from torchvision.datasets import CIFAR10
    result = hash_inputs()
    # torchvision checks official per-batch MD5s before its legacy CIFAR unpickler.
    dataset = CIFAR10(str(ROOT / 'data/cifar10'), train=True, download=False)
    if len(dataset) != 50000:
        raise ValueError('CIFAR-10 training size mismatch')
    for name in ['motion_blur', 'frost']:
        x = np.load(ROOT / 'data/CIFAR-C' / (name + '.npy'), allow_pickle=False, mmap_mode='r')
        if x.shape != (50000, 32, 32, 3) or x.dtype != np.uint8:
            raise ValueError('Unexpected corruption array: ' + name)
    labels = np.load(ROOT / 'data/CIFAR-C/labels.npy', allow_pickle=False, mmap_mode='r')
    if labels.shape != (50000,) or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 9:
        raise ValueError('Invalid CIFAR-C labels')
    if not all(np.array_equal(labels[:10000], labels[i * 10000:(i + 1) * 10000]) for i in range(1, 5)):
        raise ValueError('CIFAR-C severities do not share source-image labels')
    path = ROOT / 'data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    with zipfile.ZipFile(path) as archive:
        if 'imgs.npy' not in archive.namelist() or 'latents_classes.npy' not in archive.namelist():
            raise ValueError('Missing numeric dSprites members')
        with archive.open('imgs.npy') as handle:
            version = np.lib.format.read_magic(handle)
            shape, _, dtype = (np.lib.format.read_array_header_1_0(handle) if version == (1, 0)
                               else np.lib.format.read_array_header_2_0(handle))
            if shape != (737280, 64, 64) or dtype.hasobject:
                raise ValueError('Invalid dSprites image array')
    with np.load(path, allow_pickle=False) as data:
        if data['latents_classes'].shape != (737280, 6):
            raise ValueError('Invalid dSprites factors')
    with h5py.File(ROOT / 'data/shapes3d/3dshapes.h5', 'r') as handle:
        if handle['images'].shape != (480000, 64, 64, 3) or handle['labels'].shape != (480000, 6):
            raise ValueError('Invalid Shapes3D arrays')
    return result


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
