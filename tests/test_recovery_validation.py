"""Tests for recovery_validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queuediff.recovery_validation import (
    recovery_summary,
    validate_bottleneck_recovery,
    validate_parameter_recovery,
)
from queuediff.synthetic_generator import (
    SyntheticParameters,
    default_hematopoiesis_params,
)


class TestValidateParameterRecovery:
    def test_returns_dataframe(self):
        params = default_hematopoiesis_params()
        rng = np.random.default_rng(42)
        result = validate_parameter_recovery(params, n_samples=1000, rng=rng)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self):
        params = default_hematopoiesis_params()
        rng = np.random.default_rng(42)
        result = validate_parameter_recovery(params, n_samples=1000, rng=rng)
        expected = {"state", "true_shape", "fitted_shape", "shape_error",
                    "true_mean", "fitted_mean", "mean_error",
                    "shape_valid", "mean_valid"}
        assert set(result.columns) == expected

    def test_one_row_per_state(self):
        params = default_hematopoiesis_params()
        rng = np.random.default_rng(42)
        result = validate_parameter_recovery(params, n_samples=1000, rng=rng)
        assert len(result) == len(params.states)

    def test_shape_recovery_near_exact_at_large_n(self):
        """At n=5000, shape should be recovered within tolerance."""
        params = default_hematopoiesis_params()
        rng = np.random.default_rng(42)
        result = validate_parameter_recovery(params, n_samples=5000, rng=rng)
        assert result["shape_valid"].all()

    def test_mean_recovery_near_exact_at_large_n(self):
        """At n=5000, mean should be recovered within tolerance."""
        params = default_hematopoiesis_params()
        rng = np.random.default_rng(42)
        result = validate_parameter_recovery(params, n_samples=5000, rng=rng)
        assert result["mean_valid"].all()

    def test_errors_decrease_with_n(self):
        """Shape errors should decrease as n increases."""
        params = default_hematopoiesis_params()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result_small = validate_parameter_recovery(params, n_samples=100, rng=rng1)
        result_large = validate_parameter_recovery(params, n_samples=5000, rng=rng2)
        # Median error should be smaller at larger n
        assert result_large["shape_error"].median() <= result_small["shape_error"].median()

    def test_deterministic_with_same_seed(self):
        """Same random seed should produce same results."""
        params = default_hematopoiesis_params()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = validate_parameter_recovery(params, n_samples=500, rng=rng1)
        r2 = validate_parameter_recovery(params, n_samples=500, rng=rng2)
        pd.testing.assert_frame_equal(r1, r2)

    def test_custom_tolerance(self):
        """Tight tolerance should be stricter than loose tolerance."""
        params = default_hematopoiesis_params()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        tight = validate_parameter_recovery(params, n_samples=100, shape_tolerance=0.01, rng=rng1)
        loose = validate_parameter_recovery(params, n_samples=100, shape_tolerance=0.5, rng=rng2)
        assert tight["shape_valid"].sum() <= loose["shape_valid"].sum()


class TestValidateBottleneckRecovery:
    def test_returns_dict(self):
        params = default_hematopoiesis_params()
        result = validate_bottleneck_recovery(params, detected_bottleneck="GMP")
        assert isinstance(result, dict)

    def test_match_true_bottleneck(self):
        """HSC should be the true bottleneck for default params (source state with highest ρ)."""
        params = default_hematopoiesis_params()
        # HSC gets external_arrival_rate=1.0 -> ρ = 1.0 * 16.5 = 16.5, highest of all states
        result = validate_bottleneck_recovery(params, detected_bottleneck="HSC")
        assert result["match"]

    def test_mismatch_detected(self):
        """A wrong detection should not match."""
        params = default_hematopoiesis_params()
        result = validate_bottleneck_recovery(params, detected_bottleneck="GMP")
        assert not result["match"]

    def test_all_intensities_present(self):
        params = default_hematopoiesis_params()
        result = validate_bottleneck_recovery(params, detected_bottleneck="GMP")
        assert all(s in result["all_intensities"] for s in params.states)

    def test_custom_true_bottleneck(self):
        """When user provides true_bottleneck, it should be used."""
        params = default_hematopoiesis_params()
        result = validate_bottleneck_recovery(params, detected_bottleneck="HSC",
                                               true_bottleneck="HSC")
        assert result["match"]
        assert result["true_bottleneck"] == "HSC"


class TestRecoverySummary:
    @pytest.fixture
    def param_recovery(self):
        return pd.DataFrame({
            "state": ["HSC", "MPP", "GMP"],
            "true_shape": [15.0, 12.0, 20.0],
            "fitted_shape": [14.8, 11.9, 20.2],
            "shape_error": [0.013, 0.008, 0.010],
            "true_mean": [16.5, 13.2, 19.4],
            "fitted_mean": [16.3, 13.0, 19.6],
            "mean_error": [0.012, 0.015, 0.010],
            "shape_valid": [True, True, True],
            "mean_valid": [True, True, True],
        })

    @pytest.fixture
    def bottleneck_recovery(self):
        return {
            "true_bottleneck": "GMP",
            "detected_bottleneck": "GMP",
            "match": True,
            "all_intensities": {"HSC": 0.15, "MPP": 0.25, "GMP": 0.45},
        }

    def test_returns_string(self, param_recovery, bottleneck_recovery):
        summary = recovery_summary(param_recovery, bottleneck_recovery)
        assert isinstance(summary, str)

    def test_contains_state_names(self, param_recovery, bottleneck_recovery):
        summary = recovery_summary(param_recovery, bottleneck_recovery)
        for state in ["HSC", "MPP", "GMP"]:
            assert state in summary

    def test_match_in_summary(self, param_recovery, bottleneck_recovery):
        summary = recovery_summary(param_recovery, bottleneck_recovery)
        assert "MATCH" in summary

    def test_mismatch_in_summary(self, param_recovery):
        br = {
            "true_bottleneck": "GMP",
            "detected_bottleneck": "HSC",
            "match": False,
            "all_intensities": {"HSC": 0.15, "MPP": 0.25, "GMP": 0.45},
        }
        summary = recovery_summary(param_recovery, br)
        assert "MISMATCH" in summary
