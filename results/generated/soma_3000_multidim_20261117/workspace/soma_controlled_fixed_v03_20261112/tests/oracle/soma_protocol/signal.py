"""Float64 offline reference for experiment_manual.pdf equations 3.4--3.10.

Inputs are residuals, NOT arbitrary features. The caller is responsible for
pre-update predictions, a frozen readout, disjoint role IDs and frozen parameters.
This module deliberately does not import, alter, or certify the legacy trainers.
"""
from dataclasses import dataclass
from math import ceil
from typing import Literal
import numpy as np

class CalibrationError(ValueError):
    """Invalid calibration or a numerical protocol violation."""

def matrix(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 1:
        raise CalibrationError("Expected at least two samples and one feature")
    if not np.isfinite(a).all():
        raise CalibrationError("Non-finite residuals")
    return a

@dataclass(frozen=True)
class FixedReference:
    mean: np.ndarray
    inverse_sqrt: np.ndarray

    @classmethod
    def fit(cls, white: np.ndarray, alpha: float) -> "FixedReference":
        x = matrix(white)
        if not np.isfinite(alpha) or alpha <= 0:
            raise CalibrationError("alpha must be finite and positive")
        mu = x.mean(axis=0)
        c = x - mu
        cov = c.T @ c / (len(c) - 1)
        trace = float(np.trace(cov))
        if not np.isfinite(trace) or trace <= 0:
            raise CalibrationError("Non-positive reference trace")
        reg = cov + alpha * trace / x.shape[1] * np.eye(x.shape[1])
        eig, vec = np.linalg.eigh(reg)
        if not np.isfinite(eig).all() or np.min(eig) <= 0:
            raise CalibrationError("Reference is not positive definite")
        inv = (vec * (1 / np.sqrt(eig))) @ vec.T
        mu.setflags(write=False)
        inv.setflags(write=False)
        return cls(mu, inv)

    def transform(self, residuals: np.ndarray) -> np.ndarray:
        x = matrix(residuals)
        if x.shape[1] != len(self.mean):
            raise CalibrationError("Readout dimension changed")
        return (x - self.mean) @ self.inverse_sqrt.T

def ordered_shape_spectrum(white_window: np.ndarray, floor: float,
                           negative_tolerance: float,
                           backend: Literal["covariance", "gram"] = "covariance"):
    """Return rmax eigenvalues, energy and state.

    negative_tolerance is absolute in the trace-d normalized spectrum. It is a
    mandatory caller-supplied parameter, not a test-time adjustable fallback.
    """
    x = matrix(white_window)
    if not np.isfinite(floor) or floor < 0:
        raise CalibrationError("Invalid low-energy floor")
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0:
        raise CalibrationError("Invalid eigenvalue tolerance")
    if backend not in {"covariance", "gram"}:
        raise CalibrationError("Unknown backend")
    n, d = x.shape
    rmax = min(d, n - 1)
    c = x - x.mean(axis=0)
    trace = float(np.sum(c * c) / (n - 1))
    if not np.isfinite(trace):
        raise CalibrationError("Non-finite window energy")
    if trace <= floor:
        return np.zeros(rmax), trace, "low-energy"
    scale = d / ((n - 1) * trace)
    cov = (c.T @ c if backend == "covariance" else c @ c.T) * scale
    eig = np.linalg.eigvalsh(cov)
    if not np.isfinite(eig).all() or np.min(eig) < -negative_tolerance:
        raise CalibrationError("Invalid spectrum; no silent window exclusion")
    eig = np.maximum(eig, 0)[::-1][:rmax]
    return eig, trace, "valid"

def psi_window(white_window: np.ndarray, edge: float, floor: float,
               negative_tolerance: float, backend: str = "covariance") -> dict:
    """Compute Psi after application of a single frozen whitening reference."""
    if not np.isfinite(edge) or edge <= 0:
        raise CalibrationError("Empirical edge must be positive and finite")
    eig, trace, state = ordered_shape_spectrum(
        white_window, floor, negative_tolerance, backend)
    active = eig > edge
    return {"psi": float(np.sum(eig[active] / edge - 1)),
            "spectrum": eig, "trace": trace, "active_modes": int(active.sum()),
            "status": state}

def calibrate_edge(whitened_edge: np.ndarray, window: int, beta: float,
                   floor: float, negative_tolerance: float) -> dict:
    """Non-overlapping windows, kth order statistic, NO interpolation.

    Require an exact number of windows to avoid silent leftover sample exclusion.
    Edge coverage and sustained online false alarms require separate assets.
    """
    x = matrix(whitened_edge)
    if not isinstance(window, int) or window < 2 or len(x) % window:
        raise CalibrationError("Edge block must contain complete disjoint windows")
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise CalibrationError("beta must be in (0,1)")
    maxima = []
    for start in range(0, len(x), window):
        eig, _, _ = ordered_shape_spectrum(x[start:start+window], floor,
                                           negative_tolerance)
        maxima.append(float(eig[0]))
    k = ceil((1 - beta) * len(maxima))
    edge = float(np.sort(maxima)[k - 1])
    if not np.isfinite(edge) or edge <= 0:
        raise CalibrationError("Uncalibratable non-positive edge")
    return {"edge": edge, "k": k, "n_windows": len(maxima),
            "low_energy_windows": sum(v == 0 for v in maxima), "maxima": maxima}
