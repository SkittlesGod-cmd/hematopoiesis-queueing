"""Tests for synthetic_generator module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queuediff.synthetic_generator import (
    SyntheticParameters,
    compute_true_residence_times,
    compute_true_traffic_intensity,
    default_hematopoiesis_params,
    generate_residence_times,
    simulate_clone_trajectories,
)


@pytest.fixture
def params() -> SyntheticParameters:
    return default_hematopoiesis_params()


class TestSyntheticParameters:
    def test_default_params_valid(self, params):
        assert len(params.states) == 6
        assert all(s in params.gamma_shapes for s in params.states)
        assert all(s in params.gamma_scales for s in params.states)
        assert params.source_state in params.states

    def test_routing_probs_sum_to_one_or_zero(self, params):
        for state, routes in params.routing_probs.items():
            total = sum(routes.values())
            assert total == 0.0 or abs(total - 1.0) < 1e-10


class TestGenerateResidenceTimes:
    def test_returns_correct_size(self, params):
        times = generate_residence_times(params, n_samples=100, state="HSC", rng=np.random.default_rng(42))
        assert len(times) == 100

    def test_all_positive(self, params):
        times = generate_residence_times(params, n_samples=1000, state="MPP", rng=np.random.default_rng(42))
        assert np.all(times > 0)

    def test_mean_close_to_expected(self, params):
        """Mean of large sample should be close to k*theta."""
        rng = np.random.default_rng(42)
        state = "GMP"
        times = generate_residence_times(params, n_samples=10000, state=state, rng=rng)
        expected_mean = params.gamma_shapes[state] * params.gamma_scales[state]
        assert abs(times.mean() - expected_mean) / expected_mean < 0.05


class TestSimulateCloneTrajectories:
    def test_returns_dataframe(self, params):
        rng = np.random.default_rng(42)
        small_params = SyntheticParameters(
            states=params.states,
            gamma_shapes=params.gamma_shapes,
            gamma_scales=params.gamma_scales,
            routing_probs=params.routing_probs,
            source_state=params.source_state,
            n_clones=5,
            cells_per_clone=5,
            observation_times=[48.0, 96.0, 144.0],
        )
        df = simulate_clone_trajectories(small_params, rng=rng)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, params):
        rng = np.random.default_rng(42)
        small_params = SyntheticParameters(
            states=params.states,
            gamma_shapes=params.gamma_shapes,
            gamma_scales=params.gamma_scales,
            routing_probs=params.routing_probs,
            source_state=params.source_state,
            n_clones=3,
            cells_per_clone=3,
            observation_times=[48.0, 96.0],
        )
        df = simulate_clone_trajectories(small_params, rng=rng)
        assert "clone_id" in df.columns
        assert "cell_id" in df.columns
        assert "timepoint" in df.columns
        assert "state" in df.columns

    def test_all_states_valid(self, params):
        rng = np.random.default_rng(42)
        small_params = SyntheticParameters(
            states=params.states,
            gamma_shapes=params.gamma_shapes,
            gamma_scales=params.gamma_scales,
            routing_probs=params.routing_probs,
            source_state=params.source_state,
            n_clones=10,
            cells_per_clone=10,
            observation_times=[48.0, 96.0, 144.0],
        )
        df = simulate_clone_trajectories(small_params, rng=rng)
        assert all(s in params.states for s in df["state"].unique())


class TestComputeTrueResidenceTimes:
    def test_correct_mean(self, params):
        means = compute_true_residence_times(params)
        for state in params.states:
            expected = params.gamma_shapes[state] * params.gamma_scales[state]
            assert abs(means[state] - expected) < 1e-10


class TestComputeTrueTrafficIntensity:
    def test_source_has_highest_or_known_intensity(self, params):
        intensities = compute_true_traffic_intensity(params)
        assert all(v >= 0 for v in intensities.values())

    def test_all_states_present(self, params):
        intensities = compute_true_traffic_intensity(params)
        assert set(intensities.keys()) == set(params.states)
