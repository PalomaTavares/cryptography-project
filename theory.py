import math
from math import comb

def catch_up_prob(q: float, deficit: float) -> float:
    if deficit <= 0:
        return 1.0
    if q >= 0.5:
        return 1.0

    p = 1.0 - q
    a = (q / p) ** deficit
    return a


def attacker_success_probability(q: float, z: int) -> float:
    if q >= 0.5:
        return 1.0
    p = 1.0 - q
    lam = z * (q / p)
    total = 1.0

    for k in range(z + 1):
        poisson = math.exp(-lam)
        for i in range(1, k + 1):
            poisson *= lam / i
        total -= poisson * (1.0 - catch_up_prob(q, z - k))

    return total


def rosenfeld(q: float, n: int) -> float:
    if q >= 0.5:
        return 1.0
    if n == 0:
        return 1.0
    p = 1.0 - q
    total = 0.0
    for m in range(n + 1):
        total += comb(m + n - 1, m) * (p**n * q**m - p**m * q**n)
    return 1.0 - total
