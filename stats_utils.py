import math

import numpy as np

# Two-sided 95% normal quantile, Phi^{-1}(0.975).
Z_95 = 1.959963984540054


def estimate_probability(successes: int, n: int) -> tuple[float, float]:
    """Point estimate p_hat = successes / n and its Wald standard error.

    The standard error sqrt(p_hat (1 - p_hat) / n) is fine for z-scores when
    there are many successes, but it should NOT be used to build confidence
    intervals for small p (it gives zero width at 0 successes and negative
    lower bounds for a few successes). Use `wilson_interval` for that.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")

    p_hat = successes / n
    standard_error = math.sqrt(p_hat * (1.0 - p_hat) / n)
    return p_hat, standard_error


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    With p_hat = successes / n:

        center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
        half   = z / (1 + z^2 / n) * sqrt(p_hat (1 - p_hat) / n + z^2 / (4 n^2))

    The interval is [center - half, center + half]. Unlike the Wald interval,
    the center is pulled slightly toward 0.5, which keeps both bounds inside
    [0, 1] and gives a non-zero upper bound when there are no successes.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")

    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n

    center = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )

    # Exact edges: with 0 successes the lower bound is mathematically 0, and
    # with n successes the upper bound is 1. Rounding would otherwise give
    # values like 4e-22 or 0.9999999999999999, so we pin them explicitly.
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == n else min(1.0, center + half_width)
    return lower, upper


def wilson_interval_array(successes, n, z: float = Z_95):
    """Vectorized `wilson_interval` for numpy arrays or pandas columns.

    Returns two numpy arrays (lower, upper) with the same shape as the input.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= successes <= n and n > 0")

    p_hat = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * np.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )

    lower = np.where(k == 0, 0.0, np.maximum(0.0, center - half_width))
    upper = np.where(k == n, 1.0, np.minimum(1.0, center + half_width))
    return lower, upper


def trials_for_relative_error(p: float, relative_error: float = 0.10) -> float:
    """Trials needed so that SE / p is about `relative_error`.

    From SE = sqrt(p (1 - p) / N), setting SE / p = r gives
    N = (1 - p) / (p r^2), which is about 100 / p for r = 10%.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    return (1.0 - p) / (p * relative_error ** 2)