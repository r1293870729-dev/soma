"""Audited controllers; optimizers and model-step ownership remain explicit."""
from __future__ import annotations
import math
import torch
from torch import nn

def _names(losses, expected):
    if set(losses) != set(expected):
        raise ValueError("Loss names differ from the declared active loss set")
    for name, loss in losses.items():
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
            raise ValueError("Non-finite or nonscalar loss: " + name)

def _positive(values, what):
    result = {str(k): float(v) for k, v in values.items()}
    if not result or any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError(what + " must be finite and strictly positive; no abs/clipping fallback")
    return result

def assert_disjoint_optimizers(model_optimizer, controller_optimizer):
    model_ids = {id(p) for g in model_optimizer.param_groups for p in g["params"]}
    control_ids = {id(p) for g in controller_optimizer.param_groups for p in g["params"]}
    if model_ids & control_ids:
        raise ValueError("Model and controller optimizers share parameters")

class FixedWeightController:
    def __init__(self, coefficients):
        self.coefficients = _positive(coefficients, "Active fixed coefficients")

    def weights(self, losses):
        _names(losses, self.coefficients)
        return {k: losses[k].new_tensor(self.coefficients[k]) for k in losses}

    def objective(self, losses):
        return torch.stack([self.weights(losses)[k] * losses[k] for k in losses]).sum()

class InitialScaleController:
    """References are estimated on declared calibration batches and frozen by caller."""
    def __init__(self, reference_losses, base_coefficients=None, eps=1e-8):
        self.reference = _positive(reference_losses, "Calibration reference losses")
        self.base = _positive(base_coefficients or {k: 1.0 for k in self.reference}, "Base coefficients")
        if set(self.base) != set(self.reference):
            raise ValueError("Coefficient and calibration loss sets differ")
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

    def weights(self, losses):
        _names(losses, self.reference)
        return {k: losses[k].new_tensor(self.base[k] / (self.reference[k] + self.eps)) for k in losses}

class UncertaintyController(nn.Module):
    """Empirical learnable weighting unless the loss likelihood justifies more."""
    def __init__(self, loss_names):
        super().__init__()
        names = list(loss_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Loss names must be nonempty and unique")
        self.log_vars = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in names})

    def objective(self, losses):
        _names(losses, self.log_vars)
        if any(float(v.detach()) < 0 for v in losses.values()):
            raise ValueError("Negative losses require an approved weighting definition")
        self.current_weights()
        result = torch.stack([torch.exp(-self.log_vars[k]) * losses[k] + self.log_vars[k] for k in losses]).sum()
        if not bool(torch.isfinite(result.detach())):
            raise FloatingPointError("Non-finite uncertainty objective")
        return result

    def current_weights(self):
        if not all(bool(torch.isfinite(v.detach())) for v in self.log_vars.values()):
            raise FloatingPointError("Non-finite uncertainty controller state")
        result = {k: float(torch.exp(-v.detach()).cpu()) for k, v in self.log_vars.items()}
        if any(not math.isfinite(v) or v <= 0 for v in result.values()):
            raise FloatingPointError("Uncertainty weights overflowed or collapsed")
        return result

class GradNormController(nn.Module):
    """Only controller parameters receive gradients from gradnorm_loss."""
    def __init__(self, loss_names, initial_losses, alpha=1.5):
        super().__init__()
        self.names = list(loss_names)
        if not self.names or len(set(self.names)) != len(self.names) or set(self.names) != set(initial_losses):
            raise ValueError("Initial loss names do not match declared losses")
        self.initial = _positive(initial_losses, "GradNorm initial losses")
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.raw = nn.Parameter(torch.zeros(len(self.names)))

    def weights_tensor(self):
        if not bool(torch.isfinite(self.raw.detach()).all()):
            raise FloatingPointError("Non-finite GradNorm controller state")
        w = torch.nn.functional.softplus(self.raw) + 1e-8
        result = len(self.names) * w / w.sum()
        if not bool(torch.isfinite(result.detach()).all()) or bool((result.detach() <= 0).any()):
            raise FloatingPointError("GradNorm weights overflowed or collapsed")
        return result

    def gradnorm_loss(self, current_losses, unweighted_norms):
        _names(current_losses, self.names)
        if any(float(v.detach()) < 0 for v in current_losses.values()):
            raise ValueError("Negative losses are unsupported by this relative-rate definition")
        norms = unweighted_norms.detach().to(device=self.raw.device, dtype=self.raw.dtype)
        if tuple(norms.shape) != (len(self.names),) or not bool(torch.isfinite(norms).all()) or bool((norms < 0).any()):
            raise ValueError("Invalid unweighted gradient norms")
        ratios = torch.stack([current_losses[k].detach().to(self.raw) / self.initial[k] for k in self.names])
        if not bool(torch.isfinite(ratios).all()) or float(ratios.mean()) <= 0:
            raise ValueError("Undefined relative training rates")
        rates = ratios / ratios.mean()
        weighted_norms = self.weights_tensor() * norms
        # GradNorm Eq. 2: the complete target is constant during weight optimization.
        target = (weighted_norms.mean() * rates.pow(self.alpha)).detach()
        result = torch.abs(weighted_norms - target).sum()
        if not bool(torch.isfinite(result.detach())):
            raise FloatingPointError("Non-finite GradNorm objective")
        return result

    def model_objective(self, losses):
        _names(losses, self.names)
        # The model loss must not train GradNorm weights through the task objective.
        weights = self.weights_tensor().detach()
        return torch.stack([weights[i] * losses[k] for i, k in enumerate(self.names)]).sum()

    def weights_dict(self):
        return dict(zip(self.names, self.weights_tensor().detach().cpu().tolist()))
