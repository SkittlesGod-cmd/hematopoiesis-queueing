"""Tests for clonal_residence_time module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from queuediff.clonal_residence_time import (
    compute_normalized_arrival_rates,
    compute_residence_time_summary,
    estimate_residence_times_clonal,
    extract_clone_trajectories,
)


@pytest.fixture
def clonal_adata():
    """AnnData with clone matrix and timepoints for trajectory extraction."""
    n_cells = 60
    n_clones = 5

    # 3 timepoints, 20 cells each
    timepoints = np.repeat([2.0, 4.0, 6.0], 20)
    obs = pd.DataFrame(
        {"Time_point": timepoints},
        index=[f"Cell{i}" for i in range(n_cells)],
    )

    # Clone matrix: 5 clones, each with cells across timepoints
    clone_data = np.zeros((n_cells, n_clones), dtype=np.float32)
    # Clone 0: cells 0,1 (tp=2), 20,21 (tp=4), 40,41 (tp=6)
    clone_data[[0, 1, 20, 21, 40, 41], 0] = 1.0
    # Clone 1: cells 5,6 (tp=2), 25,26 (tp=4), 45,46 (tp=6)
    clone_data[[5, 6, 25, 26, 45, 46], 1] = 1.0
    # Clone 2: cells 10 (tp=2), 30 (tp=4)
    clone_data[[10, 30], 2] = 1.0
    # Clone 3: single cell (should be filtered)
    clone_data[15, 3] = 1.0
    # Clone 4: cells 3,4 (tp=2), 23,24 (tp=4), 43,44 (tp=6)
    clone_data[[3, 4, 23, 24, 43, 44], 4] = 1.0

    adata = ad.AnnData(
        X=np.zeros((n_cells, 10), dtype=np.float32),
        obs=obs,
    )
    adata.obsm["clone_matrix"] = csr_matrix(clone_data)

    return adata


@pytest.fixture
def state_assignments(clonal_adata):
    """State assignments where some clones transition between states."""
    n_cells = clonal_adata.n_obs
    states = []
    for i in range(n_cells):
        if i < 20:  # tp=2
            states.append("HSC" if i < 10 else "MPP")
        elif i < 40:  # tp=4
            states.append("MPP" if i < 30 else "CMP")
        else:  # tp=6
            states.append("CMP" if i < 50 else "GMP")

    return pd.Series(states, index=clonal_adata.obs_names, name="state")


class TestExtractCloneTrajectories:
    def test_returns_dataframe(self, clonal_adata, state_assignments):
        df = extract_clone_trajectories(clonal_adata, state_assignments)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, clonal_adata, state_assignments):
        df = extract_clone_trajectories(clonal_adata, state_assignments)
        expected = {"clone_id", "cell_id", "timepoint", "state"}
        assert set(df.columns) == expected

    def test_filters_small_clones(self, clonal_adata, state_assignments):
        """Clone 3 has only 1 cell, should be excluded with min_clone_size=2."""
        df = extract_clone_trajectories(clonal_adata, state_assignments, min_clone_size=2)
        assert 3 not in df["clone_id"].values

    def test_includes_multi_cell_clones(self, clonal_adata, state_assignments):
        df = extract_clone_trajectories(clonal_adata, state_assignments, min_clone_size=2)
        assert 0 in df["clone_id"].values
        assert 1 in df["clone_id"].values

    def test_missing_clone_matrix_raises(self, state_assignments):
        adata = ad.AnnData(X=np.zeros((60, 10), dtype=np.float32))
        adata.obs["Time_point"] = np.repeat([2.0, 4.0, 6.0], 20)
        adata.obs_names = [f"Cell{i}" for i in range(60)]
        with pytest.raises(ValueError, match="clone_matrix"):
            extract_clone_trajectories(adata, state_assignments)


class TestEstimateResidenceTimesClonal:
    def test_returns_dict(self, clonal_adata, state_assignments):
        traj = extract_clone_trajectories(clonal_adata, state_assignments)
        times = estimate_residence_times_clonal(traj)
        assert isinstance(times, dict)

    def test_values_are_positive(self, clonal_adata, state_assignments):
        traj = extract_clone_trajectories(clonal_adata, state_assignments)
        times = estimate_residence_times_clonal(traj)
        for state, values in times.items():
            assert np.all(values > 0), f"State {state} has non-positive times"

    def test_values_in_hours(self, clonal_adata, state_assignments):
        """With time_unit_hours=24, values should be multiples of 24h intervals."""
        traj = extract_clone_trajectories(clonal_adata, state_assignments)
        times = estimate_residence_times_clonal(traj, time_unit_hours=24.0)
        for state, values in times.items():
            # Minimum should be at least one interval (2 days = 48h)
            assert np.all(values >= 24.0), f"State {state}: min={values.min()}"

    def test_empty_trajectories(self):
        empty = pd.DataFrame(columns=["clone_id", "cell_id", "timepoint", "state"])
        times = estimate_residence_times_clonal(empty)
        assert times == {}


class TestComputeResidenceTimeSummary:
    def test_returns_dataframe(self):
        times = {
            "HSC": np.array([48.0, 96.0, 72.0]),
            "MPP": np.array([24.0, 48.0]),
        }
        summary = compute_residence_time_summary(times)
        assert isinstance(summary, pd.DataFrame)

    def test_correct_columns(self):
        times = {"HSC": np.array([48.0, 96.0])}
        summary = compute_residence_time_summary(times)
        expected = {"state", "n_observations", "mean_hours", "std_hours",
                    "median_hours", "min_hours", "max_hours"}
        assert set(summary.columns) == expected

    def test_correct_mean(self):
        times = {"HSC": np.array([48.0, 96.0])}
        summary = compute_residence_time_summary(times)
        assert summary.iloc[0]["mean_hours"] == 72.0


class TestComputeNormalizedArrivalRates:
    def test_returns_dict(self, clonal_adata, state_assignments):
        traj = extract_clone_trajectories(clonal_adata, state_assignments)
        rates = compute_normalized_arrival_rates(traj, state_assignments)
        assert isinstance(rates, dict)

    def test_rates_non_negative(self, clonal_adata, state_assignments):
        traj = extract_clone_trajectories(clonal_adata, state_assignments)
        rates = compute_normalized_arrival_rates(traj, state_assignments)
        assert all(r >= 0 for r in rates.values())

    def test_empty_trajectories(self):
        empty = pd.DataFrame(columns=["clone_id", "cell_id", "timepoint", "state"])
        assignments = pd.Series(dtype=str)
        rates = compute_normalized_arrival_rates(empty, assignments)
        assert rates == {}
