import math


def catch_up_prob(q: float, deficit: float) -> float:
    if deficit <= 0:
        return 1.0
    if q >= 0.5:
        return 1.0

    p = 1.0 - q
    a = (q / p) ** deficit
    return a


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0

    # This scalar implementation is adequate for the z range used here, but
    # math.exp(-lam) underflows for lam > ~745, so scipy.stats.poisson.pmf is
    # the numerically stable choice when looking at much larger lambda values.
    term = math.exp(-lam)
    for i in range(1, k + 1):
        term *= lam / i
    return term


def poisson_tail_from(z: int, lam: float) -> float:
    if z < 0:
        return 1.0

    total = 0.0
    k = z + 1
    term = poisson_pmf(k, lam)
    while term > 0.0:
        total += term
        k += 1
        term *= lam / k
        if term < 1e-300:
            break
    return total


def attacker_success_probability(q: float, z: int) -> float:
    if q >= 0.5:
        return 1.0
    p = 1.0 - q
    lam = z * (q / p)

    # P = sum_{k>z} Pois(k; lam) + sum_{k=0}^z Pois(k; lam) * (q/p)^(z-k)
    total = poisson_tail_from(z, lam)
    for k in range(z + 1):
        pmf = poisson_pmf(k, lam)
        total += pmf * catch_up_prob(q, z - k)
    return total


def min_confirmations(prob_fn, q: float, threshold: float = 0.001, z_max: int = 2000) -> int:
    for z in range(z_max + 1):
        if prob_fn(q, z) < threshold:
            return z
    raise ValueError(f"no z <= {z_max} gives P < {threshold} for q={q}")


def _logaddexp(log_a: float, log_b: float) -> float:
    if math.isinf(log_a):
        return log_b
    if math.isinf(log_b):
        return log_a
    if log_a < log_b:
        log_a, log_b = log_b, log_a
    return log_a + math.log1p(math.exp(log_b - log_a))


def rosenfeld(q: float, n: int) -> float:
    if q == 0.0:
        return 0.0
    if q >= 0.5:
        return 1.0
    if n == 0:
        return 1.0

    p = 1.0 - q
    log_total = -math.inf

    # Rosenfeld can be written as a sum of positive terms, but the direct
    # combinatorial terms overflow for moderate n. We therefore sum in log-space.
    for m in range(n + 1):
        log_term = (
            math.lgamma(n + m)
            - math.lgamma(m + 1)
            - math.lgamma(n)
            + m * math.log(p)
            + n * math.log(q)
        )
        log_total = _logaddexp(log_total, log_term)

    m = n + 1
    log_term = (
        math.lgamma(2 * n + 1)
        - math.lgamma(n + 2)
        - math.lgamma(n)
        + n * math.log(p)
        + (n + 1) * math.log(q)
    )
    while log_term > math.log(1e-300):
        log_total = _logaddexp(log_total, log_term)
        m += 1
        log_term += math.log((n + m - 1) / m) + math.log(q)

    return math.exp(log_total)
