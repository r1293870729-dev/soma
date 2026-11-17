"""Final B = sum_i max(lambda_i / empirical_edge - 1, 0).

CPU float64 reference for manual equations 3.4--3.10, not a complete SOMA trainer.
The online monitor has no stage-label argument and never updates its reference.
A/C are offline diagnostics only. No MP fallback, extra covariance jitter, signal
normalization by mode count, differentiable signal loss, or test-time re-selection.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from soma_protocol.signal import (CalibrationError, FixedReference, calibrate_edge,
                                  ordered_shape_spectrum)
from soma_protocol.validation import assert_disjoint_roles

SIGNAL_ID = "B_mass_fixed_white_empirical_v1"

def _finite_array(x, *, min_rows=1, dim=None):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < min_rows or a.shape[1] < 1:
        raise CalibrationError("Expected nonempty samples x features matrix")
    if dim is not None and a.shape[1] != dim:
        raise CalibrationError("Frozen readout dimension changed")
    if not np.isfinite(a).all():
        raise CalibrationError("Non-finite residual input")
    return a

@dataclass(frozen=True)
class BConfig:
    window: int
    alpha: float
    beta: float
    energy_floor: float
    negative_tolerance: float
    backend: str = "covariance"

    def __post_init__(self):
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 2:
            raise CalibrationError("window must be an integer >= 2")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise CalibrationError("alpha must be finite and positive")
        if not math.isfinite(self.beta) or not 0 < self.beta < 1:
            raise CalibrationError("beta must lie strictly in (0,1)")
        if not math.isfinite(self.energy_floor) or self.energy_floor < 0:
            raise CalibrationError("energy_floor must be finite and nonnegative")
        if not math.isfinite(self.negative_tolerance) or self.negative_tolerance < 0:
            raise CalibrationError("negative_tolerance must be finite and nonnegative")
        if self.backend not in {"covariance", "gram"}:
            raise CalibrationError("Only exact covariance or Gram backends are allowed")


def validate_main_signal(policy: dict) -> None:
    """Reject A/C/entropy/MP or loss-coupled policies, even in development config."""
    expected = {"signal_id": SIGNAL_ID, "family": "B", "readout": "mass",
                "reference_policy": "fixed", "edge_policy": "empirical_order_statistic",
                "normalize_by_mode_count": False, "backpropagate_signal": False}
    for key, val in expected.items():
        if key not in policy or policy[key] != val:
            raise CalibrationError(f"Final B policy violation: {key} must equal {val!r}")


def spectrum_diagnostics(eigenvalues, edge: float) -> dict:
    """A/B/C in the original ABC readout sense. Only B is the main signal.

    This is NOT a p-value; an empirical max-eigenvalue edge does not calibrate
    the sustained online alarm or establish structural necessity.
    """
    eig = np.asarray(eigenvalues, dtype=np.float64)
    if eig.ndim != 1 or eig.size == 0 or not np.isfinite(eig).all() or np.min(eig) < 0:
        raise CalibrationError("Expected finite nonnegative 1-D spectrum")
    if not math.isfinite(edge) or edge <= 0:
        raise CalibrationError("Empirical edge must be positive and finite")
    eig = np.sort(eig)[::-1]
    excess = np.maximum(eig / edge - 1.0, 0.0)
    a = float(excess[0])
    b = float(np.sum(excess))
    second = float(eig[1]) if len(eig) > 1 else 0.0
    c = max(0.0, (float(eig[0]) - max(second, edge)) / edge)
    active = int(np.sum(eig > edge))
    return {"psi_b": b, "a_edge_diagnostic": a, "c_gap_diagnostic": c,
            "active_modes": active, "b_minus_a": b-a,
            "multi_mode_active": active >= 2}

class BMonitor:
    """Frozen B reference plus causal residual ring. Diagnostic state is mutable.

    update() accepts pre-update residuals in arrival order. It cannot by itself
    prove their provenance: model snapshot, role and arrival controls belong in
    the caller. No oracle/stage parameter is accepted.
    """
    def __init__(self, config: BConfig, mean, inverse_sqrt, edge: float,
                 role_manifest: dict, edge_report: dict):
        self.config = config
        self.mean = np.array(mean, dtype=np.float64, copy=True)
        self.inverse_sqrt = np.array(inverse_sqrt, dtype=np.float64, copy=True)
        if (self.mean.ndim != 1 or self.mean.size < 1
            or self.inverse_sqrt.shape != (self.mean.size, self.mean.size)
            or not np.isfinite(self.mean).all() or not np.isfinite(self.inverse_sqrt).all()):
            raise CalibrationError("Invalid reference arrays")
        if not np.allclose(self.inverse_sqrt, self.inverse_sqrt.T, rtol=1e-12, atol=1e-12):
            raise CalibrationError("Inverse square root is not symmetric")
        if np.min(np.linalg.eigvalsh(self.inverse_sqrt)) <= 0:
            raise CalibrationError("Inverse square root is not positive definite")
        if not math.isfinite(edge) or edge <= 0:
            raise CalibrationError("Nonpositive empirical edge is a calibration failure")
        self.mean.setflags(write=False)
        self.inverse_sqrt.setflags(write=False)
        self.edge = float(edge)
        self.role_manifest = json.loads(json.dumps(role_manifest))
        self.edge_report = json.loads(json.dumps(edge_report))
        assert_disjoint_roles(self.role_manifest)
        self._ring = deque(maxlen=config.window)
        self.samples_seen = 0

    @classmethod
    def fit(cls, white, edge_residuals, config: BConfig, role_ids: dict) -> "BMonitor":
        w = _finite_array(white, min_rows=2)
        e = _finite_array(edge_residuals, min_rows=config.window, dim=w.shape[1])
        if set(role_ids) != {"white", "edge"}:
            raise CalibrationError("Calibration requires explicit white and edge ID lists")
        if len(role_ids['white']) != len(w) or len(role_ids['edge']) != len(e):
            raise CalibrationError("Calibration ID lengths do not match residuals")
        if any(not isinstance(x, str) or not x for ids in role_ids.values() for x in ids):
            raise CalibrationError("Sample IDs must be nonempty canonical strings")
        assert_disjoint_roles(role_ids)
        ref = FixedReference.fit(w, config.alpha)
        report = calibrate_edge(ref.transform(e), config.window, config.beta,
                                config.energy_floor, config.negative_tolerance)
        return cls(config, ref.mean, ref.inverse_sqrt, report['edge'], role_ids, report)

    def _metadata(self) -> dict:
        return {"schema": 1, "signal_id": SIGNAL_ID, "config": asdict(self.config),
                "edge": self.edge, "roles": self.role_manifest, "edge_report": self.edge_report}

    def reference_id(self) -> str:
        h = hashlib.sha256(json.dumps(self._metadata(), sort_keys=True, allow_nan=False).encode())
        for a in (self.mean, self.inverse_sqrt):
            h.update(np.asarray(a, dtype='<f8').tobytes())
        return h.hexdigest()

    def reset_window(self) -> None:
        self._ring.clear()
        self.samples_seen = 0

    def update(self, residuals) -> dict:
        x = _finite_array(residuals, dim=len(self.mean))
        white = [(row - self.mean) @ self.inverse_sqrt.T for row in x]
        if any(not np.isfinite(row).all() for row in white):
            raise CalibrationError("Whitening overflow; no silent sample drop")
        for row in white:
            self._ring.append(row.copy())
            self.samples_seen += 1
        return self.measure()

    def measure(self) -> dict:
        meta = {"signal_id": SIGNAL_ID, "window_count": len(self._ring),
                "samples_seen": self.samples_seen, "lambda_plus_0": self.edge}
        if len(self._ring) < self.config.window:
            return {**meta, "status": "warming-window", "psi_b": None,
                    "trace": None, "spectrum": [], "active_modes": None,
                    "a_edge_diagnostic": None, "c_gap_diagnostic": None,
                    "b_minus_a": None, "multi_mode_active": None}
        eig, trace, state = ordered_shape_spectrum(np.stack(self._ring),
                      self.config.energy_floor, self.config.negative_tolerance, self.config.backend)
        return {**meta, "status": state, "trace": trace, "spectrum": eig.tolist(),
                **spectrum_diagnostics(eig, self.edge)}

    def save(self, path: str | Path, *, include_window: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self._metadata(), "reference_id": self.reference_id(),
                "has_window": include_window,
                "samples_seen": self.samples_seen if include_window else 0}
        window = np.stack(self._ring) if include_window and self._ring else np.empty((0,len(self.mean)))
        with path.open('xb') as handle:
            np.savez_compressed(handle, mean=self.mean, inverse_sqrt=self.inverse_sqrt,
                ring=window, metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)))

    @classmethod
    def load(cls, path: str | Path) -> "BMonitor":
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a['metadata'].item()))
            if meta.get('schema') != 1 or meta.get('signal_id') != SIGNAL_ID:
                raise CalibrationError("Reference version is not final B")
            obj = cls(BConfig(**meta['config']), a['mean'], a['inverse_sqrt'],
                      meta['edge'], meta['roles'], meta['edge_report'])
            if obj.reference_id() != meta.get('reference_id'):
                raise CalibrationError("Reference metadata/array mismatch")
            ring = np.asarray(a['ring'], dtype=np.float64)
            if (ring.ndim != 2 or ring.shape[1] != len(obj.mean)
                or len(ring) > obj.config.window or not np.isfinite(ring).all()):
                raise CalibrationError("Invalid saved residual ring")
            seen = meta.get('samples_seen')
            if isinstance(seen, bool) or not isinstance(seen, int) or seen < len(ring):
                raise CalibrationError("Invalid saved sample count")
            if not meta['has_window'] and (len(ring) or seen):
                raise CalibrationError("Reference-only asset contains online state")
            if meta['has_window'] and len(ring) != min(seen, obj.config.window):
                raise CalibrationError("Incomplete saved online ring")
            obj._ring.extend(row.copy() for row in ring)
            obj.samples_seen = seen
            return obj
