"""Separate manual-aligned primitives; keep imported legacy metrics intact."""
import math
from numbers import Real

def exact_alignment(edits: list[float], boundaries: list[float], tolerance: float) -> dict:
    """Max-cardinality 1-D matching, then minimum total absolute delay.

    Sorted absolute-distance matching has an optimal non-crossing solution;
    dynamic programming returns it rather than greedy nearest-edge matching.
    Small-event reference retaining paths, not intended for all signal windows.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid alignment tolerance")
    for xs in (edits, boundaries):
        if any(not isinstance(v, Real) or not math.isfinite(v) for v in xs):
            raise ValueError("Invalid event time")
        if len(xs) != len(set(xs)):
            raise ValueError("Duplicate event timestamps require explicit IDs/policy")
    a, b = sorted(edits), sorted(boundaries)
    n, m = len(a), len(b)
    dp = [[(0, 0.0, ()) for _ in range(m+1)] for _ in range(n+1)]
    def key(v): return (-v[0], v[1], v[2])
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [dp[i-1][j], dp[i][j-1]]
            distance = abs(a[i-1] - b[j-1])
            if distance <= tolerance:
                c, cost, pairs = dp[i-1][j-1]
                options.append((c+1, cost+distance, pairs+((i-1,j-1),)))
            dp[i][j] = min(options, key=key)
    count, cost, pairs = dp[n][m]
    return {"matches": [(a[i],b[j]) for i,j in pairs], "n_matched": count,
            "total_absolute_delay": cost, "taef": count/n if n else 0.0,
            "boundary_hit_rate": count/m if m else None, "no_edit": n == 0}

def restricted_recovery_burden(drop: bool, recovery_delay: float | None,
                               horizon: float) -> float:
    """I_drop * min(delay,horizon); None denotes unrecovered/censored.

    Caller must pre-freeze common events, pre-event baseline, thresholds,
    persistence and horizon. This is NOT excess-loss area.
    """
    if not isinstance(drop, bool):
        raise ValueError("drop must be a boolean, not an inferred area")
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Invalid common follow-up horizon")
    if recovery_delay is not None and (math.isnan(recovery_delay) or recovery_delay < 0):
        raise ValueError("Invalid recovery delay")
    if not drop:
        return 0.0
    return horizon if recovery_delay is None else min(float(recovery_delay), horizon)
