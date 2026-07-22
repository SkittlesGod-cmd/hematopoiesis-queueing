"""Tests for state_discretization module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from queuediff.state_discretization import (
    MARKER_PANELS,
    assign_states,
    calibrate_division_death_rates,
    score_apoptosis,
    score_cell_cycle,
    score_marker_states,
)


@pytest.fixture
def adata_with_markers():
    """AnnData with known marker genes in lognorm layer and lognorm_full in obsm."""
    rng = np.random.default_rng(42)
    n_cells = 100

    # Create gene set that includes some marker genes
    hvg_genes = ["Cd34", "Kit", "Flt3", "Gata1", "Klf1", "Mpo", "Elane",
                 "Gene7", "Gene8", "Gene9"]
    n_hvgs = len(hvg_genes)

    # Full gene set includes cell-cycle and apoptosis genes
    full_genes = hvg_genes + ["Pcna", "Mcm2", "Top2a", "Ccnb1",
                              "Casp3", "Bax", "Bcl2", "Bcl2l1",
                              "OtherGene1", "OtherGene2"]
    n_full = len(full_genes)

    # Build lognorm expression (HVG subset)
    lognorm = rng.exponential(0.5, size=(n_cells, n_hvgs)).astype(np.float32)
    # Make MPP markers high for first 30 cells
    lognorm[:30, 0:3] = rng.exponential(2.0, size=(30, 3)).astype(np.float32)  # Cd34,Kit,Flt3
    # Make MEP markers high for next 30
    lognorm[30:60, 3:5] = rng.exponential(2.0, size=(30, 2)).astype(np.float32)  # Gata1,Klf1
    # Make GMP markers high for last 40
    lognorm[60:, 5:7] = rng.exponential(2.0, size=(40, 2)).astype(np.float32)  # Mpo,Elane

    # Build lognorm_full (full gene set)
    lognorm_full = rng.exponential(0.3, size=(n_cells, n_full)).astype(np.float32)
    # Cell-cycle genes (indices 10-13) should have signal
    lognorm_full[:, 10:14] = rng.exponential(1.0, size=(n_cells, 4)).astype(np.float32)
    # Apoptosis genes (indices 14-17)
    lognorm_full[:, 14:16] = rng.exponential(0.8, size=(n_cells, 2)).astype(np.float32)
    lognorm_full[:, 16:18] = rng.exponential(0.4, size=(n_cells, 2)).astype(np.float32)

    # Timepoints
    timepoints = np.repeat([2.0, 4.0, 6.0], [40, 30, 30])

    obs = pd.DataFrame(
        {"Time_point": timepoints},
        index=[f"Cell{i}" for i in range(n_cells)],
    )

    adata = ad.AnnData(
        X=np.zeros((n_cells, n_hvgs), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=hvg_genes),
    )
    adata.layers["lognorm"] = lognorm
    adata.obsm["lognorm_full"] = lognorm_full
    adata.uns["lognorm_full_genes"] = full_genes

    return adata


class TestScoreMarkerStates:
    def test_returns_dataframe(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assert isinstance(scores, pd.DataFrame)

    def test_correct_columns(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assert set(scores.columns) == set(MARKER_PANELS.keys())

    def test_correct_index(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assert list(scores.index) == list(adata_with_markers.obs_names)

    def test_scores_non_negative(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assert (scores >= 0).all().all()

    def test_custom_panels(self, adata_with_markers):
        custom = {"StateA": ["Cd34", "Kit"], "StateB": ["Gata1"]}
        scores = score_marker_states(adata_with_markers, panels=custom)
        assert list(scores.columns) == ["StateA", "StateB"]

    def test_missing_lognorm_raises(self, adata_with_markers):
        del adata_with_markers.layers["lognorm"]
        with pytest.raises(ValueError, match="lognorm"):
            score_marker_states(adata_with_markers)


class TestAssignStates:
    def test_returns_series(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        assert isinstance(assignments, pd.Series)

    def test_all_cells_assigned(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        assert len(assignments) == adata_with_markers.n_obs
        assert assignments.notna().all()

    def test_assignments_are_valid_states(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        assert all(s in MARKER_PANELS for s in assignments.unique())


class TestScoreCellCycle:
    def test_returns_dataframe(self, adata_with_markers):
        result = score_cell_cycle(adata_with_markers)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, adata_with_markers):
        result = score_cell_cycle(adata_with_markers)
        assert set(result.columns) == {"s_score", "g2m_score", "cycling_score"}

    def test_cycling_is_sum(self, adata_with_markers):
        result = score_cell_cycle(adata_with_markers)
        expected = result["s_score"] + result["g2m_score"]
        np.testing.assert_allclose(result["cycling_score"], expected, rtol=1e-5)

    def test_scores_positive(self, adata_with_markers):
        """Cell-cycle genes have positive expression, so scores should be > 0."""
        result = score_cell_cycle(adata_with_markers)
        assert (result["cycling_score"] > 0).all()

    def test_missing_lognorm_full_raises(self, adata_with_markers):
        del adata_with_markers.obsm["lognorm_full"]
        with pytest.raises(ValueError, match="lognorm_full"):
            score_cell_cycle(adata_with_markers)


class TestScoreApoptosis:
    def test_returns_dataframe(self, adata_with_markers):
        result = score_apoptosis(adata_with_markers)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, adata_with_markers):
        result = score_apoptosis(adata_with_markers)
        assert set(result.columns) == {"pro_score", "anti_score", "net_apoptotic_score"}

    def test_net_is_pro_minus_anti(self, adata_with_markers):
        result = score_apoptosis(adata_with_markers)
        expected = result["pro_score"] - result["anti_score"]
        np.testing.assert_allclose(result["net_apoptotic_score"], expected, rtol=1e-5)


class TestCalibrateDivisionDeathRates:
    def test_returns_dataframe(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        cc = score_cell_cycle(adata_with_markers)
        apop = score_apoptosis(adata_with_markers)

        rates = calibrate_division_death_rates(
            adata_with_markers, assignments,
            cc["cycling_score"], apop["net_apoptotic_score"],
        )
        assert isinstance(rates, pd.DataFrame)

    def test_correct_columns(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        cc = score_cell_cycle(adata_with_markers)
        apop = score_apoptosis(adata_with_markers)

        rates = calibrate_division_death_rates(
            adata_with_markers, assignments,
            cc["cycling_score"], apop["net_apoptotic_score"],
        )
        expected_cols = {"state", "net_growth_rate", "signature_ratio",
                        "division_rate", "death_rate", "net_shrinking"}
        assert set(rates.columns) == expected_cols

    def test_death_rate_non_negative(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        cc = score_cell_cycle(adata_with_markers)
        apop = score_apoptosis(adata_with_markers)

        rates = calibrate_division_death_rates(
            adata_with_markers, assignments,
            cc["cycling_score"], apop["net_apoptotic_score"],
        )
        assert (rates["death_rate"] >= 0).all()

    def test_division_rate_non_negative(self, adata_with_markers):
        scores = score_marker_states(adata_with_markers)
        assignments = assign_states(scores)
        cc = score_cell_cycle(adata_with_markers)
        apop = score_apoptosis(adata_with_markers)

        rates = calibrate_division_death_rates(
            adata_with_markers, assignments,
            cc["cycling_score"], apop["net_apoptotic_score"],
        )
        assert (rates["division_rate"] >= 0).all()
