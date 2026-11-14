import numpy as np

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