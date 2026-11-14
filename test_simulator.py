import numpy as np

from simulator import simulate_attack


def test_z_zero_tie_rule():
    rng = np.random.default_rng(123)
    n_trials = 1000
    successes = simulate_attack(0.30, 0, n_trials, rng, overtake=False)
    assert successes == n_trials


def test_q_zero_never_mines():
    rng = np.random.default_rng(123)
    successes = simulate_attack(0.0, 5, 2000, rng, overtake=False)
    assert successes == 0


def test_q_near_half_has_high_success():
    rng = np.random.default_rng(123)
    successes = simulate_attack(0.49, 1, 5000, rng, overtake=False)
    assert successes > 0.9 * 5000


def test_overtake_is_harder_than_tie():
    n_trials = 5000
    tie_rng = np.random.default_rng(123)
    overtake_rng = np.random.default_rng(123)

    tie_successes = simulate_attack(0.30, 3, n_trials, tie_rng, overtake=False)
    overtake_successes = simulate_attack(0.30, 3, n_trials, overtake_rng, overtake=True)

    assert overtake_successes <= tie_successes


def test_larger_z_reduces_success():
    n_trials = 5000
    small_z_rng = np.random.default_rng(123)
    large_z_rng = np.random.default_rng(123)

    small_z_successes = simulate_attack(0.30, 2, n_trials, small_z_rng, overtake=False)
    large_z_successes = simulate_attack(0.30, 5, n_trials, large_z_rng, overtake=False)

    assert large_z_successes < small_z_successes


def test_cap_is_negligible_for_large_n():
    n_trials = 500_000
    caps = [30, 60, 120]
    rates = []

    for cap in caps:
        rng = np.random.default_rng(123)
        successes = simulate_attack(0.45, 5, n_trials, rng, overtake=False, max_deficit=cap)
        rates.append(successes / n_trials)

    assert max(abs(rate - rates[1]) for rate in rates) < 0.0025
