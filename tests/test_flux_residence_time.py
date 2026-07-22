"""Tests for flux_residence_time module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from queuediff.flux_residence_time import (
    compute_state_occupancy,
    fit_transition_rates,
    flux_residence_time_summary,
    identify_degenerate_states,
)


@pytest.fixture
def occupancy_data():
    """Synthetic state occupancy fractions over time."""
    # 3 timepoints (days), 3 states
    data = {
        "HSC": [0.4, 0.3, 0.2],
        "MPP": [0.3, 0.35, 0.3],
        "GMP": [0.3, 0.35, 0.5],
    }
    return pd.DataFrame(data, index=[2.0, 4.0, 6.0])


@pytest.fixture
def routing():
    """Simple linear routing structure."""
    return {
        "HSC": ["MPP"],
        "MPP": ["GMP"],
        "GMP": [],  # terminal
    }


class TestComputeStateOccupancy:
    def test_returns_dataframe(self):
        obs = pd.DataFrame(
            {"Time_point": [2.0, 2.0, 4.0, 4.0]},
            index=["c1", "c2", "c3", "c4"],
        )
        adata = ad.AnnData(
            X=np.zeros((4, 5), dtype=np.float32),
            obs=obs,
        )
        states = pd.Series(["HSC", "MPP", "HSC", "MPP"], index=adata.obs_names)
        result = compute_state_occupancy(adata, states)
        assert isinstance(result, pd.DataFrame)

    def test_fractions_sum_to_one(self):
        obs = pd.DataFrame(
            {"Time_point": [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]},
            index=[f"c{i}" for i in range(6)],
        )
        adata = ad.AnnData(
            X=np.zeros((6, 5), dtype=np.float32),
            obs=obs,
        )
        states = pd.Series(["HSC", "HSC", "MPP", "MPP", "MPP", "GMP"],
                          index=adata.obs_names)
        result = compute_state_occupancy(adata, states)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)


class TestFitTransitionRates:
    def test_returns_dataframe(self, occupancy_data, routing):
        result = fit_transition_rates(occupancy_data, routing)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, occupancy_data, routing):
        result = fit_transition_rates(occupancy_data, routing)
        expected = {"state", "exit_rate_per_hour", "residence_time_hours", "is_degenerate"}
        assert set(result.columns) == expected

    def test_one_row_per_state(self, occupancy_data, routing):
        result = fit_transition_rates(occupancy_data, routing)
        assert len(result) == 3

    def test_rates_positive(self, occupancy_data, routing):
        result = fit_transition_rates(occupancy_data, routing)
        assert (result["exit_rate_per_hour"] > 0).all()

    def test_residence_times_positive(self, occupancy_data, routing):
        result = fit_transition_rates(occupancy_data, routing)
        assert (result["residence_time_hours"] > 0).all()


class TestIdentifyDegenerateStates:
    def test_returns_list(self, occupancy_data, routing):
        fit = fit_transition_rates(occupancy_data, routing)
        degenerate = identify_degenerate_states(fit)
        assert isinstance(degenerate, list)

    def test_empty_when_none_degenerate(self):
        """Non-degenerate fit should return empty list."""
        df = pd.DataFrame({
            "state": ["A", "B"],
            "exit_rate_per_hour": [0.05, 0.03],
            "residence_time_hours": [20.0, 33.3],
            "is_degenerate": [False, False],
        })
        assert identify_degenerate_states(df) == []


class TestFluxResidenceTimeSummary:
    def test_returns_merged_dataframe(self):
        clonal = pd.DataFrame({
            "state": ["HSC", "MPP"],
            "mean_hours": [16.0, 13.0],
        })
        flux = pd.DataFrame({
            "state": ["HSC", "MPP"],
            "residence_time_hours": [18.0, 15.0],
            "is_degenerate": [False, False],
        })
        result = flux_residence_time_summary(clonal, flux)
        assert "clonal_mean_hours" in result.columns
        assert "flux_mean_hours" in result.columns
        assert "ratio_clonal_flux" in result.columns
        assert len(result) == 2
