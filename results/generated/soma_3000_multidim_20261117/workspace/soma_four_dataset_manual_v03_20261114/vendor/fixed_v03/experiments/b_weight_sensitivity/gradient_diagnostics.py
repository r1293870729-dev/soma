"""Read-only per-loss gradients on one named, fixed shared parameter group."""
from __future__ import annotations
import math
import torch

def collect_gradient_vectors(losses, parameters, coefficients=None):
    params = [p for p in parameters if p.requires_grad]
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("Shared parameters must be nonempty and unique")
    coefficients = coefficients or {k: 1.0 for k in losses}
    if set(coefficients) != set(losses):
        raise ValueError("Coefficient keys differ from loss keys")
    vectors, rows = {}, []
    for name, loss in losses.items():
        c = float(coefficients[name])
        if not math.isfinite(c) or c <= 0:
            raise ValueError("Active coefficients must be finite and positive")
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError("Invalid loss " + name)
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if loss.requires_grad else (None,) * len(params)
        parts, unused = [], 0
        for p, grad in zip(params, grads):
            if grad is None:
                unused += p.numel()
                parts.append(torch.zeros_like(p).reshape(-1))
            else:
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError("Non-finite gradient " + name)
                parts.append(grad.detach().reshape(-1))
        raw = torch.cat(parts)
        weighted = c * raw
        if not bool(torch.isfinite(weighted).all()):
            raise FloatingPointError("Weighted gradient overflow " + name)
        raw_norm = torch.linalg.vector_norm(raw)
        weighted_norm = torch.linalg.vector_norm(weighted)
        if not bool(torch.isfinite(raw_norm)) or not bool(torch.isfinite(weighted_norm)):
            raise FloatingPointError("Gradient norm overflow " + name)
        vectors[name] = (raw, weighted)
        total = raw.numel()
        rows.append({
            "loss_name": name, "coefficient": c,
            "raw_loss": float(loss.detach()), "weighted_loss": c * float(loss.detach()),
            "raw_norm": float(raw_norm.cpu()),
            "weighted_norm": float(weighted_norm.cpu()),
            "raw_nonzero_fraction": int(torch.count_nonzero(raw)) / total,
            "weighted_nonzero_fraction": int(torch.count_nonzero(weighted)) / total,
            "unused_parameter_elements": unused, "parameter_elements": total,
            "gradient_status": "constant_loss" if not loss.requires_grad else ("disconnected" if unused == total else "measured"),
        })
    cosines = []
    names = list(vectors)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for kind, idx in (("raw", 0), ("weighted", 1)):
                va, vb = vectors[a][idx], vectors[b][idx]
                norm_a, norm_b = torch.linalg.vector_norm(va), torch.linalg.vector_norm(vb)
                defined = float(norm_a) > 0 and float(norm_b) > 0
                cosine = float(torch.dot(va/norm_a, vb/norm_b).clamp(-1, 1).cpu()) if defined else None
                if cosine is not None and not math.isfinite(cosine):
                    raise FloatingPointError("Non-finite gradient cosine")
                cosines.append({"loss_a": a, "loss_b": b, "kind": kind, "cosine": cosine,
                                "status": "measured" if defined else "undefined_zero_norm"})
    return rows, cosines
