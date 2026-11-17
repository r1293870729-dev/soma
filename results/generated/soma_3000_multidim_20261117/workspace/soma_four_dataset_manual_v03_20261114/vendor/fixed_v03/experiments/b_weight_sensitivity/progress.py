"""Read-only training progress; no random draws or tensor operations."""
import math


def bar(value, total, width=28):
    fraction = min(1., max(0., value / total)) if total else 0.
    filled = int(width * fraction)
    return '[' + '=' * filled + '-' * (width - filled) + f'] {value}/{total} ({100 * fraction:.1f}%)'


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return '待估计'
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600); minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
