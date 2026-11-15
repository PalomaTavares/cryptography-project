import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    njit = None


def simulate_attack(q: float, z: int, n_trials: int,
                    rng: np.random.Generator,
                    overtake: bool = False,
                    max_deficit: int = 60) -> int:
    # Monte Carlo estimate of double-spend success.
    # Returns the number of successful attacks out of n_trials.
    successes = 0
    for _ in range(n_trials):
        honest, attacker = 0, 0

        # Phase 1: confirmation phase
        while honest < z:
            if rng.random() < q:
                attacker += 1
            else:
                honest += 1

        # Phase 2: race
        target = 1 if overtake else 0
        while True:
            if attacker - honest >= target:
                successes += 1
                break
            if honest - attacker > max_deficit:
                break
            if rng.random() < q:
                attacker += 1
            else:
                honest += 1

    return successes


def simulate_attack_exp(q: float, z: int, n_trials: int,
                        rng: np.random.Generator,
                        overtake: bool = False,
                        max_deficit: int = 60,
                        return_durations: bool = False):
    successes = 0
    durations = []

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    for _ in range(n_trials):
        honest, attacker = 0, 0
        elapsed = 0.0

        # Phase 1: confirmation phase
        while honest < z:
            if q == 0.0:
                t_honest = rng.exponential(1.0)
                elapsed += t_honest
                honest += 1
            elif q == 1.0:
                t_attack = rng.exponential(1.0)
                elapsed += t_attack
                attacker += 1
            else:
                t_honest = rng.exponential(1.0 / (1.0 - q))
                t_attack = rng.exponential(1.0 / q)
                elapsed += min(t_honest, t_attack)
                if t_attack < t_honest:
                    attacker += 1
                else:
                    honest += 1

        # Phase 2: race
        target = 1 if overtake else 0
        while True:
            if attacker - honest >= target:
                successes += 1
                durations.append(elapsed)
                break
            if honest - attacker > max_deficit:
                break

            if q == 0.0:
                t_honest = rng.exponential(1.0)
                elapsed += t_honest
                honest += 1
            elif q == 1.0:
                t_attack = rng.exponential(1.0)
                elapsed += t_attack
                attacker += 1
            else:
                t_honest = rng.exponential(1.0 / (1.0 - q))
                t_attack = rng.exponential(1.0 / q)
                elapsed += min(t_honest, t_attack)
                if t_attack < t_honest:
                    attacker += 1
                else:
                    honest += 1

    if return_durations:
        return successes, np.asarray(durations)
    return successes


def simulate_attack_fast(q: float, z: int, n_trials: int, seed: int,
                        overtake: bool = False,
                        max_deficit: int = 60) -> int:
    if njit is None:
        raise ImportError("Numba is required for simulate_attack_fast. Install it with 'pip install numba'.")

    @njit
    def _simulate_attack_fast(q, z, n_trials, seed, overtake, max_deficit):
        np.random.seed(seed)
        successes = 0
        target = 1 if overtake else 0

        for _ in range(n_trials):
            honest = 0
            attacker = 0

            while honest < z:
                if np.random.random() < q:
                    attacker += 1
                else:
                    honest += 1

            while True:
                if attacker - honest >= target:
                    successes += 1
                    break
                if honest - attacker > max_deficit:
                    break
                if np.random.random() < q:
                    attacker += 1
                else:
                    honest += 1

        return successes

    return _simulate_attack_fast(q, z, n_trials, seed, overtake, max_deficit)