import pytest

from theory import attacker_success_probability, min_confirmations, rosenfeld


def test_attacker_success_probability_large_z_is_positive_and_monotone():
    prob_50 = attacker_success_probability(0.1, 50)
    prob_100 = attacker_success_probability(0.1, 100)
    assert 0.0 < prob_100 < prob_50

    prob_50_q2 = attacker_success_probability(0.2, 50)
    prob_100_q2 = attacker_success_probability(0.2, 100)
    assert 0.0 < prob_100_q2 < prob_50_q2


def test_rosenfeld_large_n_is_positive_and_monotone():
    prob_50 = rosenfeld(0.1, 50)
    prob_100 = rosenfeld(0.1, 100)
    assert 0.0 < prob_100 < prob_50

    prob_50_q2 = rosenfeld(0.2, 50)
    prob_100_q2 = rosenfeld(0.2, 100)
    assert 0.0 < prob_100_q2 < prob_50_q2


def test_nakamoto_reference_values():
    assert abs(attacker_success_probability(0.1, 5) - 0.0009137) < 1e-7
    assert abs(attacker_success_probability(0.3, 10) - 0.0416605) < 1e-7


def test_rosenfeld_zero_q_is_zero():
    assert rosenfeld(0.0, 5) == 0.0


def test_min_confirmations_raises_when_threshold_never_reached():
    with pytest.raises(ValueError):
        min_confirmations(rosenfeld, 0.5)


def test_nakamoto_min_confirmations_table():
    q_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    expected = [5, 8, 11, 15, 24, 41, 89, 340]

    for q, want in zip(q_values, expected):
        assert min_confirmations(attacker_success_probability, q) == want


def test_rosenfeld_min_confirmations_table():
    q_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    expected = [6, 9, 13, 20, 32, 58, 133, 539]

    for q, want in zip(q_values, expected):
        assert min_confirmations(rosenfeld, q) == want


def test_rosenfeld_reference_cases():
    cases = [
        (0.10, 1, 0.20),
        (0.10, 2, 0.05600),
        (0.10, 5, 0.00178),
        (0.20, 3, 0.11584),
        (0.20, 6, 0.02331),
        (0.30, 4, 0.25207),
        (0.30, 10, 0.06511),
        (0.40, 5, 0.53314),
        (0.40, 10, 0.37218),
        (0.50, 3, 1.0),
    ]

    for q, n, expected in cases:
        got = rosenfeld(q, n)
        assert abs(got - expected) < 1e-5, f"q={q}, n={n}: got {got}, expected {expected}"
