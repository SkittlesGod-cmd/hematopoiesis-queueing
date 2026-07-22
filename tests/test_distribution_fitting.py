"""Tests for distribution_fitting module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from queuediff.distribution_fitting import (
    FitResult,
    fit_exponential,
    fit_gamma,
    likelihood_ratio_test,
)


@pytest.fixture
def gamma_data():
    """Gamma-distributed data with known parameters."""
    rng = np.random.default_rng(42)
    # shape=10, scale=1.5 => mean=15, clearly non-exponential
    return rng.gamma(shape=10, scale=1.5, size=500)


@pytest.fixture
def exponential_data():
    """Exponential data (gamma with shape=1)."""
    rng = np.random.default_rng(42)
    return rng.exponential(scale=10.0, size=500)


class TestFitGamma:
    def test_returns_fit_result(self, gamma_data):
        result = fit_gamma(gamma_data)
        assert isinstance(result, FitResult)
        assert result.distribution == "gamma"

    def test_recovers_shape(self, gamma_data):
        """Fitted shape should be close to true shape=10."""
        result = fit_gamma(gamma_data)
        assert abs(result.params["shape"] - 10.0) < 2.0

    def test_recovers_scale(self, gamma_data):
        """Fitted scale should be close to true scale=1.5."""
        result = fit_gamma(gamma_data)
        assert abs(result.params["scale"] - 1.5) < 0.5

    def test_mean_close_to_true(self, gamma_data):
        result = fit_gamma(gamma_data)
        assert abs(result.mean - 15.0) < 2.0

    def test_n_params_is_two(self, gamma_data):
        result = fit_gamma(gamma_data)
        assert result.n_params == 2

    def test_rejects_negative_data(self):
        with pytest.raises(ValueError, match="positive"):
            fit_gamma(np.array([-1.0, 2.0, 3.0]))

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_gamma(np.array([1.0]))


class TestFitExponential:
    def test_returns_fit_result(self, exponential_data):
        result = fit_exponential(exponential_data)
        assert isinstance(result, FitResult)
        assert result.distribution == "exponential"

    def test_recovers_scale(self, exponential_data):
        """Fitted scale should be close to true scale=10."""
        result = fit_exponential(exponential_data)
        assert abs(result.params["scale"] - 10.0) < 2.0

    def test_n_params_is_one(self, exponential_data):
        result = fit_exponential(exponential_data)
        assert result.n_params == 1


class TestLikelihoodRatioTest:
    def test_gamma_vs_exponential_on_gamma_data(self, gamma_data):
        """LR test should reject exponential in favor of gamma for gamma data."""
        gamma_fit = fit_gamma(gamma_data)
        exp_fit = fit_exponential(gamma_data)
        lr_stat, p_value = likelihood_ratio_test(exp_fit, gamma_fit)
        assert lr_stat > 0
        assert p_value < 0.001  # Strong rejection of exponential

    def test_gamma_vs_exponential_on_exp_data(self, exponential_data):
        """LR test should NOT strongly reject exponential for exponential data."""
        gamma_fit = fit_gamma(exponential_data)
        exp_fit = fit_exponential(exponential_data)
        lr_stat, p_value = likelihood_ratio_test(exp_fit, gamma_fit)
        # p-value should be relatively high (not reject H0)
        assert p_value > 0.01

    def test_same_n_samples_required(self, gamma_data, exponential_data):
        gamma_fit = fit_gamma(gamma_data)
        exp_fit = fit_exponential(exponential_data[:100])
        with pytest.raises(AssertionError):
            likelihood_ratio_test(exp_fit, gamma_fit)

    def test_aic_favors_gamma_for_gamma_data(self, gamma_data):
        """AIC should be lower for gamma when data is truly gamma."""
        gamma_fit = fit_gamma(gamma_data)
        exp_fit = fit_exponential(gamma_data)
        # delta_aic = exp_aic - gamma_aic should be > 0
        assert exp_fit.aic - gamma_fit.aic > 2.0
