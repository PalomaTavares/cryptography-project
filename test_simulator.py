import numpy as np
import pytest

from simulator import simulate_attack, simulate_attack_exp, simulate_attack_fast
from theory import rosenfeld


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


def test_same_seed_same_result():
    a = simulate_attack(0.3, 5, 10_000, np.random.default_rng(42))
    b = simulate_attack(0.3, 5, 10_000, np.random.default_rng(42))
    assert a == b


def test_matches_rosenfeld():
    n = 20_000
    for q, z in [(0.1, 2), (0.3, 3)]:
        p_hat = simulate_attack(q, z, n, np.random.default_rng(7)) / n
        p = rosenfeld(q, z)
        z_score = (p_hat - p) / ((p * (1 - p) / n) ** 0.5)
        assert abs(z_score) < 4


def test_exponential_simulator_zero_cases_and_rosenfeld_match():
    rng = np.random.default_rng(123)
    n = 20_000

    assert simulate_attack_exp(0.0, 5, n, rng) == 0
    assert simulate_attack_exp(0.3, 0, n, rng) == n

    q, z = 0.3, 3
    p_hat = simulate_attack_exp(q, z, n, np.random.default_rng(7)) / n
    p = rosenfeld(q, z)
    z_score = (p_hat - p) / ((p * (1 - p) / n) ** 0.5)
    assert abs(z_score) < 4


def test_numba_simulator_zero_cases_and_rosenfeld_match():
    pytest.importorskip("numba")

    assert simulate_attack_fast(0.0, 5, 2000, 123) == 0
    assert simulate_attack_fast(0.3, 0, 2000, 123) == 2000

    q, z, n = 0.3, 3, 20_000
    p_hat = simulate_attack_fast(q, z, n, 7) / n
    p = rosenfeld(q, z)
    z_score = (p_hat - p) / ((p * (1 - p) / n) ** 0.5)
    assert abs(z_score) < 4