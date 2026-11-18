import math

import numpy as np
import pytest

from stats_utils import (
    Z_95,
    estimate_probability,
    trials_for_relative_error,
    wilson_interval,
    wilson_interval_array,
)


def test_estimate_probability_basic():
    p_hat, se = estimate_probability(10, 100)
    assert p_hat == pytest.approx(0.1)
    assert se == pytest.approx(math.sqrt(0.1 * 0.9 / 100))


def test_wilson_interval_handles_zero_and_one():
    lower, upper = wilson_interval(0, 100)
    assert lower == pytest.approx(0.0)
    assert 0.0 < upper < 1.0

    lower, upper = wilson_interval(100, 100)
    assert 0.0 < lower < 1.0
    assert upper == pytest.approx(1.0)


def test_wilson_interval_array_matches_scalar_behavior():
    lower, upper = wilson_interval_array([0, 10, 100], [100, 100, 100])
    assert np.all(lower >= 0.0)
    assert np.all(upper <= 1.0)
    scalar_lower, scalar_upper = wilson_interval(10, 100)
    assert lower[1] == pytest.approx(scalar_lower)
    assert upper[1] == pytest.approx(scalar_upper)


def test_trials_for_relative_error_matches_formula():
    n = trials_for_relative_error(0.1, relative_error=0.1)
    assert n == pytest.approx(900.0)


def test_z95_constant_is_expected_quantile():
    assert Z_95 == pytest.approx(1.959963984540054)
