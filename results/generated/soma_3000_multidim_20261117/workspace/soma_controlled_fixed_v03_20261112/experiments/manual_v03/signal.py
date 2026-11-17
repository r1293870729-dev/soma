"""Manual v0.3 equations 3.1--3.11; float64, fixed reference, no stage input."""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np

SIGNAL_ID = 'Psi_mass_manual_v03_fixed_white_empirical'


class SignalError(ValueError):
    pass


@dataclass(frozen=True)
class SignalConfig:
    window: int
    alpha: float
    beta: float
    energy_floor: float
    negative_tolerance: float
    backend: str = 'auto'

    def __post_init__(self):
        if type(self.window) is not int or self.window < 2:
            raise SignalError('window must be an integer >= 2')
        for key in ('alpha', 'beta', 'energy_floor', 'negative_tolerance'):
            if not math.isfinite(getattr(self, key)):
                raise SignalError('Nonfinite config: ' + key)
        if self.alpha <= 0 or not 0 < self.beta < 1 or min(self.energy_floor, self.negative_tolerance) < 0:
            raise SignalError('Invalid numeric configuration')
        if self.backend not in ('auto', 'covariance', 'gram'):
            raise SignalError('Unknown exact decomposition backend')


def matrix(value, minimum=2, dimension=None):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < minimum or x.shape[1] < 1 or not np.isfinite(x).all():
        raise SignalError('Expected finite samples x features')
    if dimension is not None and x.shape[1] != dimension:
        raise SignalError('Readout dimension changed')
    return x


def shape_spectrum(x, config):
    x = matrix(x)
    n, d = x.shape
    centered = x - x.mean(axis=0)
    trace = float(np.sum(centered * centered) / (n - 1))
    if not math.isfinite(trace):
        raise SignalError('Nonfinite covariance trace')
    if trace <= config.energy_floor:
        return np.zeros(min(d, n - 1)), trace, 'low-energy', 'none'
    backend = config.backend
    if backend == 'auto':
        backend = 'gram' if n < d else 'covariance'
    gram = (np.einsum('ik,jk->ij', centered, centered, optimize=False) if backend == 'gram'
            else np.einsum('ki,kj->ij', centered, centered, optimize=False))
    covariance = gram * (d / ((n - 1) * trace))
    if not np.isfinite(covariance).all():
        raise SignalError('Nonfinite normalized covariance')
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -config.negative_tolerance:
        raise SignalError('Spectrum violates frozen negative tolerance')
    # All rmax modes are retained, including zeros. No top-k approximation.
    return np.maximum(eigenvalues, 0)[::-1][:min(d, n - 1)], trace, 'valid', backend


def mass(eigenvalues, edge):
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.ndim != 1 or not len(eigenvalues) or not np.isfinite(eigenvalues).all() or np.any(eigenvalues < 0):
        raise SignalError('Invalid ordered spectrum')
    if not math.isfinite(edge) or edge <= 0:
        raise SignalError('Nonpositive/nonfinite empirical edge; no MP fallback')
    psi = float(np.maximum(eigenvalues / edge - 1, 0).sum())
    peak = float(np.maximum(eigenvalues.max() / edge - 1, 0))
    return dict(psi=psi, active_modes=int((eigenvalues > edge).sum()),
                peak_excess=peak, mass_minus_peak=psi - peak)


def validate_roles(roles):
    seen = set()
    for role, ids in roles.items():
        if not ids or any(not isinstance(i, str) or not i for i in ids):
            raise SignalError('Missing canonical source IDs for ' + role)
        current = set(ids)
        if current & seen:
            raise SignalError('Source-ID overlap at role ' + role)
        seen.update(current)


def midrank_position(calibration, value):
    x = np.asarray(calibration, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not math.isfinite(value):
        raise SignalError('Missing/nonfinite independent display calibration')
    return float((1 + (x < value).sum() + .5 * (x == value).sum()) / (len(x) + 1))


class FixedPsi:
    def __init__(self, mean, inverse_sqrt, edge, config, metadata):
        self.config = config
        self.mean = np.array(mean, dtype=np.float64, copy=True)
        self.inverse_sqrt = np.array(inverse_sqrt, dtype=np.float64, copy=True)
        self.edge = float(edge)
        mass(np.zeros(1), self.edge)
        self.mean.setflags(write=False)
        self.inverse_sqrt.setflags(write=False)
        self.metadata = json.loads(json.dumps(metadata))
        self.ring = deque(maxlen=config.window)
        self.samples_seen = 0

    @classmethod
    def fit(cls, white, edge, config, roles):
        white = matrix(white)
        edge = matrix(edge, dimension=white.shape[1])
        validate_roles(roles)
        if len(roles['white']) != len(white) or len(roles['edge']) != len(edge):
            raise SignalError('Calibration rows and role draw IDs differ')
        if len(edge) % config.window:
            raise SignalError('Edge block has an incomplete window')
        mean = white.mean(axis=0)
        x = white - mean
        covariance = np.einsum('ki,kj->ij', x, x, optimize=False) / (len(x) - 1)
        trace = float(np.trace(covariance))
        if not math.isfinite(trace) or trace <= 0:
            raise SignalError('Nonpositive white covariance trace')
        regularized = covariance + config.alpha * trace / white.shape[1] * np.eye(white.shape[1])
        values, vectors = np.linalg.eigh(regularized)
        if not np.isfinite(values).all() or values.min() <= 0:
            raise SignalError('White reference is not positive definite')
        inverse_sqrt = np.einsum('ik,jk->ij', vectors * (1 / np.sqrt(values)), vectors, optimize=False)
        transformed = np.einsum('ij,kj->ik', edge - mean, inverse_sqrt, optimize=False)
        maxima = []
        for i in range(0, len(edge), config.window):
            spectrum, _, _, _ = shape_spectrum(transformed[i:i + config.window], config)
            maxima.append(float(spectrum[0]))
        k = math.ceil((1 - config.beta) * len(maxima))
        threshold = float(np.sort(maxima)[k - 1])
        metadata = dict(signal_id=SIGNAL_ID, config=asdict(config),
                        roles=roles, edge_maxima=maxima, order_statistic_k=k,
                        edge_windows=len(maxima), low_energy_edge_windows=maxima.count(0.))
        return cls(mean, inverse_sqrt, threshold, config, metadata)

    def reference_hash(self):
        h = hashlib.sha256(json.dumps(dict(self.metadata, edge=self.edge), sort_keys=True, allow_nan=False).encode())
        for x in (self.mean, self.inverse_sqrt):
            h.update(x.astype('<f8').tobytes())
        return h.hexdigest()

    def measure_window(self, residuals):
        x = np.einsum('ij,kj->ik', matrix(residuals, dimension=len(self.mean)) - self.mean,
                      self.inverse_sqrt, optimize=False)
        spectrum, trace, status, backend = shape_spectrum(x, self.config)
        return dict(status=status, trace=trace, mean_total_variance=trace / len(self.mean),
                    spectrum=spectrum.tolist(), backend=backend, edge=self.edge,
                    **mass(spectrum, self.edge))

    def update(self, residuals):
        x = matrix(residuals, minimum=1, dimension=len(self.mean))
        for row in x:
            self.ring.append(row.copy())
            self.samples_seen += 1
        base = dict(signal_id=SIGNAL_ID, samples_seen=self.samples_seen,
                    window_count=len(self.ring), reference_sha256=self.reference_hash())
        if len(self.ring) < self.config.window:
            return dict(base, status='warming-window', eligible=False, psi=None,
                        trace=None, spectrum=[], active_modes=None, edge=self.edge)
        # Low-energy windows remain eligible. Stage identity is never an input.
        return dict(base, eligible=True, **self.measure_window(np.stack(self.ring)))

    def coverage(self, residuals):
        x = matrix(residuals, dimension=len(self.mean))
        if len(x) % self.config.window:
            raise SignalError('Incomplete edge-cover window')
        rows = [self.measure_window(x[i:i + self.config.window]) for i in range(0, len(x), self.config.window)]
        count = sum(r['spectrum'][0] > self.edge for r in rows)
        return dict(windows=len(rows), exceedances=int(count), exceedance_fraction=count / len(rows),
                    low_energy_windows=sum(r['status'] == 'low-energy' for r in rows),
                    interpretation='descriptive fixed-snapshot coverage; not sustained online false-alarm coverage',
                    rows=rows)
