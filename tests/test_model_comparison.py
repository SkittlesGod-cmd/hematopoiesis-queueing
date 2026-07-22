"""Tests for model_comparison module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queuediff.model_comparison import (
    apply_fdr_correction,
    compare_models_per_state,
    summarize_model_comparison,
)


@pytest.fixture
def residence_times_gamma():
    """Residence times from gamma distributions (not exponential)."""
    rng = np.random.default_rng(42)
    return {
        "StateA": rng.gamma(shape=15, scale=1.2, size=300),
        "StateB": rng.gamma(shape=10, scale=1.5, size=300),
        "StateC": rng.gamma(shape=20, scale=0.8, size=300),
    }


@pytest.fixture
def residence_times_mixed():
    """Mix of gamma and exponential residence times."""
    rng = np.random.default_rng(42)
    return {
        "StateA": rng.gamma(shape=15, scale=1.2, size=300),
        "StateB": rng.exponential(scale=10.0, size=300),
    }


class TestCompareModelsPerState:
    def test_returns_dataframe(self, residence_times_gamma):
        result = compare_models_per_state(residence_times_gamma)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, residence_times_gamma):
        result = compare_models_per_state(residence_times_gamma)
        expected = {
            "state", "n_samples", "gamma_shape", "gamma_scale", "gamma_mean",
            "gamma_variance", "exp_scale", "exp_mean", "gamma_aic", "exp_aic",
            "delta_aic", "gamma_bic", "exp_bic", "delta_bic",
            "lr_statistic", "lr_pvalue", "gamma_loglik", "exp_loglik",
        }
        assert set(result.columns) == expected

    def test_one_row_per_state(self, residence_times_gamma):
        result = compare_models_per_state(residence_times_gamma)
        assert len(result) == 3

    def test_delta_aic_positive_for_gamma_data(self, residence_times_gamma):
        """ΔAIC should be positive (favoring gamma) when data is gamma."""
        result = compare_models_per_state(residence_times_gamma)
        assert (result["delta_aic"] > 2.0).all()

    def test_skips_small_samples(self):
        times = {"StateA": np.array([1.0, 2.0, 3.0])}
        result = compare_models_per_state(times, min_samples=10)
        assert len(result) == 0


class TestApplyFdrCorrection:
    def test_adds_fdr_columns(self, residence_times_gamma):
        comparison = compare_models_per_state(residence_times_gamma)
        result = apply_fdr_correction(comparison)
        assert "fdr_pvalue" in result.columns
        assert "gamma_preferred" in result.columns

    def test_gamma_preferred_for_gamma_data(self, residence_times_gamma):
        comparison = compare_models_per_state(residence_times_gamma)
        result = apply_fdr_correction(comparison)
        assert result["gamma_preferred"].all()

    def test_handles_empty_dataframe(self):
        empty = pd.DataFrame()
        result = apply_fdr_correction(empty)
        assert len(result) == 0

    def test_mixed_data_classification(self, residence_times_mixed):
        """Gamma data should be classified as gamma-preferred."""
        comparison = compare_models_per_state(residence_times_mixed)
        result = apply_fdr_correction(comparison)
        # StateA (gamma with shape=15) should prefer gamma
        state_a = result[result["state"] == "StateA"]
        assert state_a["gamma_preferred"].values[0]
        # StateA should have high gamma shape (k >> 1)
        assert state_a["gamma_shape"].values[0] > 5.0


class TestSummarizeModelComparison:
    def test_returns_string(self, residence_times_gamma):
        comparison = compare_models_per_state(residence_times_gamma)
        corrected = apply_fdr_correction(comparison)
        summary = summarize_model_comparison(corrected)
        assert isinstance(summary, str)
        assert "GAMMA" in summary or "gamma" in summary.lower()

    def test_contains_state_names(self, residence_times_gamma):
        comparison = compare_models_per_state(residence_times_gamma)
        corrected = apply_fdr_correction(comparison)
        summary = summarize_model_comparison(corrected)
        for state in ["StateA", "StateB", "StateC"]:
            assert state in summary
