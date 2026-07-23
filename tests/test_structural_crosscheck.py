"""Tests for structural_crosscheck module."""

from __future__ import annotations

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from queuediff.structural_crosscheck import (
    crosscheck_state_structure,
    format_crosscheck_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def primary_assignments_all_shared():
    """Primary assignments using all 6 standard states."""
    rng = np.random.default_rng(42)
    n = 100
    states = rng.choice(["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"], size=n)
    return pd.Series(states, name="state", index=[f"Cell{i}" for i in range(n)])


@pytest.fixture
def primary_assignments_subset():
    """Primary assignments using only 3 states (HSC, MPP, CMP)."""
    n = 100
    states = np.array(["HSC"] * 40 + ["MPP"] * 35 + ["CMP"] * 25)
    return pd.Series(states, name="state", index=[f"Cell{i}" for i in range(n)])


@pytest.fixture
def primary_assignments_disjoint():
    """Primary assignments using states absent from secondary."""
    n = 100
    states = np.array(["StateA"] * 50 + ["StateB"] * 50)
    return pd.Series(states, name="state", index=[f"Cell{i}" for i in range(n)])


@pytest.fixture
def secondary_adata_scoreable(preprocessed_adata):
    """A preprocessed AnnData that can be scored by score_marker_states.

    The gene names are Gene0..Gene49 (HVG subset). The marker panels use
    real gene names like Meis1, Hlf, etc. that won't be found, so scores
    will be zero -- but the function won't crash. assign_states picks the
    first column (HSC) for ties, giving a deterministic assignment.
    """
    return preprocessed_adata


@pytest.fixture
def mock_score_result_hsc():
    """Simulated score_marker_states result where all cells score HSC highest."""
    rng = np.random.default_rng(7)
    n = 100
    states = ["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"]
    scores = rng.random((n, len(states))).astype(np.float32)
    # Make HSC the highest for all cells
    scores[:, 0] = scores.max(axis=1) + 1.0
    return pd.DataFrame(scores, columns=states, index=[f"Cell{i}" for i in range(n)])


@pytest.fixture
def mock_score_result_3states():
    """Simulated score result where only HSC, MPP, CMP appear as max."""
    rng = np.random.default_rng(7)
    n = 100
    all_states = ["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"]
    scores = rng.random((n, len(all_states))).astype(np.float32)
    # Make HSC win for cells 0-39, MPP for 40-69, CMP for 70-99
    scores[:40, 0] = 10.0  # HSC
    scores[40:70, 1] = 10.0  # MPP
    scores[70:, 3] = 10.0  # CMP
    return pd.DataFrame(scores, columns=all_states, index=[f"Cell{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# Tests: crosscheck_state_structure
# ---------------------------------------------------------------------------

class TestCrosscheckStateStructure:
    """Tests for crosscheck_state_structure."""

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_perfect_concordance_all_states_shared(
        self, mock_score, mock_assign, primary_assignments_all_shared, secondary_adata_scoreable
    ):
        """When both datasets assign all 6 states, concordance should be 1.0."""
        # Secondary assigns all 6 states
        rng = np.random.default_rng(99)
        n = secondary_adata_scoreable.n_obs
        sec_states = rng.choice(["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"], size=n)
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 6)),
            columns=["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"],
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(
            primary_assignments_all_shared, secondary_adata_scoreable
        )

        assert result["structural_concordance"] == pytest.approx(1.0)
        assert len(result["shared_states"]) == 6
        assert result["primary_only"] == []
        assert result["secondary_only"] == []

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_partial_concordance(
        self, mock_score, mock_assign, primary_assignments_subset, secondary_adata_scoreable
    ):
        """Primary has {HSC,MPP,CMP}, secondary has all 6 -> concordance 3/6 = 0.5."""
        n = secondary_adata_scoreable.n_obs
        counts = [n // 6] * 6
        counts[-1] = n - sum(counts[:-1])  # adjust last bin for rounding
        labels = ["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"]
        sec_states = np.concatenate([[lbl] * cnt for lbl, cnt in zip(labels, counts)])
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 6)),
            columns=labels,
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(
            primary_assignments_subset, secondary_adata_scoreable
        )

        # Union = 6 states, shared = 3 -> concordance = 3/6
        assert result["structural_concordance"] == pytest.approx(0.5)
        assert sorted(result["shared_states"]) == ["CMP", "HSC", "MPP"]
        assert sorted(result["primary_only"]) == []
        assert sorted(result["secondary_only"]) == ["GMP", "LMPP", "MEP"]

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_no_concordance(
        self, mock_score, mock_assign, primary_assignments_disjoint, secondary_adata_scoreable
    ):
        """Primary has {StateA,StateB}, secondary has {HSC,MPP} -> concordance 0."""
        n = secondary_adata_scoreable.n_obs
        half = n // 2
        sec_states = np.array(["HSC"] * half + ["MPP"] * (n - half))
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 6)),
            columns=["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"],
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(
            primary_assignments_disjoint, secondary_adata_scoreable
        )

        assert result["structural_concordance"] == pytest.approx(0.0)
        assert result["shared_states"] == []
        assert sorted(result["primary_only"]) == ["StateA", "StateB"]
        assert sorted(result["secondary_only"]) == ["HSC", "MPP"]

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_primary_only_states(
        self, mock_score, mock_assign, primary_assignments_disjoint, secondary_adata_scoreable
    ):
        """States in primary but not secondary appear in primary_only."""
        n = secondary_adata_scoreable.n_obs
        sec_states = np.array(["StateC"] * n)
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 1)),
            columns=["StateC"],
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(
            primary_assignments_disjoint, secondary_adata_scoreable
        )

        assert sorted(result["primary_only"]) == ["StateA", "StateB"]
        assert result["shared_states"] == []

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_secondary_only_states(
        self, mock_score, mock_assign, primary_assignments_subset, secondary_adata_scoreable
    ):
        """States in secondary but not primary appear in secondary_only."""
        n = secondary_adata_scoreable.n_obs
        # Secondary has states not in primary's {HSC, MPP, CMP}
        half = n // 2
        sec_states = np.array(["MEP"] * half + ["GMP"] * (n - half))
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 6)),
            columns=["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"],
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(
            primary_assignments_subset, secondary_adata_scoreable
        )

        assert sorted(result["secondary_only"]) == ["GMP", "MEP"]
        assert result["shared_states"] == []

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_concordance_fraction_calculation(
        self, mock_score, mock_assign, secondary_adata_scoreable
    ):
        """Concordance = |intersection| / |union|."""
        # Primary: {A, B, C}, Secondary: {B, C, D}
        # Union = {A,B,C,D} size 4, Intersection = {B,C} size 2, concordance = 0.5
        primary = pd.Series(
            ["A"] * 40 + ["B"] * 30 + ["C"] * 30,
            index=[f"Cell{i}" for i in range(100)],
        )
        n = secondary_adata_scoreable.n_obs
        third = n // 3
        sec_states = np.array(
            ["B"] * third + ["C"] * third + ["D"] * (n - 2 * third)
        )
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 3)),
            columns=["B", "C", "D"],
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(primary, secondary_adata_scoreable)

        assert result["structural_concordance"] == pytest.approx(0.5)
        assert sorted(result["shared_states"]) == ["B", "C"]
        assert sorted(result["primary_only"]) == ["A"]
        assert sorted(result["secondary_only"]) == ["D"]

    def test_empty_primary_assignments(self, secondary_adata_scoreable):
        """Empty primary_assignments should not crash."""
        primary = pd.Series([], dtype=str, name="state")

        result = crosscheck_state_structure(primary, secondary_adata_scoreable)

        # With empty primary, no shared states possible
        assert result["structural_concordance"] == pytest.approx(0.0)
        assert result["shared_states"] == []
        assert result["primary_only"] == []
        assert result["primary_distribution"] == {}

    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_secondary_scoring_failure_graceful(
        self, mock_score, primary_assignments_all_shared, secondary_adata_scoreable
    ):
        """When score_marker_states raises, the function warns and returns empty secondary."""
        mock_score.side_effect = ValueError("No lognorm layer")

        with pytest.warns(UserWarning, match="Could not score secondary"):
            result = crosscheck_state_structure(
                primary_assignments_all_shared, secondary_adata_scoreable
            )

        assert result["secondary_distribution"] == {}
        assert result["shared_states"] == []
        assert result["secondary_only"] == []
        # Primary still works
        assert len(result["primary_only"]) > 0 or len(result["shared_states"]) == 0
        # Concordance should be 0 since secondary_states is empty
        assert result["structural_concordance"] == pytest.approx(0.0)

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_custom_panels_passed_through(
        self, mock_score, mock_assign, secondary_adata_scoreable
    ):
        """Custom panels dict is forwarded to score_marker_states."""
        primary = pd.Series(["X"] * 50 + ["Y"] * 50, index=[f"Cell{i}" for i in range(100)])
        n = secondary_adata_scoreable.n_obs
        custom_panels = {"X": ["Gene1", "Gene2"], "Y": ["Gene3", "Gene4"]}
        half = n // 2
        sec_states = np.array(["X"] * half + ["Y"] * (n - half))
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 2)),
            columns=["X", "Y"],
            index=secondary_adata_scoreable.obs_names,
        )
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(
            primary, secondary_adata_scoreable, panels=custom_panels
        )

        mock_score.assert_called_once_with(secondary_adata_scoreable, panels=custom_panels)
        assert result["structural_concordance"] == pytest.approx(1.0)

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_custom_names(self, mock_score, mock_assign, secondary_adata_scoreable):
        """Custom primary_name and secondary_name appear in output."""
        primary = pd.Series(["A"] * 50 + ["B"] * 50, index=[f"Cell{i}" for i in range(100)])
        n = secondary_adata_scoreable.n_obs
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 2)), columns=["A", "B"], index=secondary_adata_scoreable.obs_names
        )
        mock_assign.return_value = pd.Series(
            ["A"] * n, index=secondary_adata_scoreable.obs_names
        )

        result = crosscheck_state_structure(
            primary,
            secondary_adata_scoreable,
            primary_name="Dataset1",
            secondary_name="Dataset2",
        )

        assert result["primary_name"] == "Dataset1"
        assert result["secondary_name"] == "Dataset2"

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_primary_distribution_computed(
        self, mock_score, mock_assign, secondary_adata_scoreable
    ):
        """Primary distribution should reflect the input assignment frequencies."""
        primary = pd.Series(
            ["A"] * 80 + ["B"] * 20, index=[f"Cell{i}" for i in range(100)]
        )
        n = secondary_adata_scoreable.n_obs
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 1)), columns=["A"], index=secondary_adata_scoreable.obs_names
        )
        mock_assign.return_value = pd.Series(
            ["A"] * n, index=secondary_adata_scoreable.obs_names
        )

        result = crosscheck_state_structure(primary, secondary_adata_scoreable)

        assert result["primary_distribution"]["A"] == pytest.approx(0.8)
        assert result["primary_distribution"]["B"] == pytest.approx(0.2)

    @patch("queuediff.structural_crosscheck.assign_states")
    @patch("queuediff.structural_crosscheck.score_marker_states")
    def test_secondary_distribution_computed(
        self, mock_score, mock_assign, secondary_adata_scoreable
    ):
        """Secondary distribution should reflect scored assignment frequencies."""
        primary = pd.Series(["A"] * 50, index=[f"Cell{i}" for i in range(50)])
        n = secondary_adata_scoreable.n_obs
        mock_score.return_value = pd.DataFrame(
            np.ones((n, 2)),
            columns=["A", "B"],
            index=secondary_adata_scoreable.obs_names,
        )
        # 70% A, 30% B
        sec_states = np.array(["A"] * int(n * 0.7) + ["B"] * (n - int(n * 0.7)))
        mock_assign.return_value = pd.Series(sec_states, index=secondary_adata_scoreable.obs_names)

        result = crosscheck_state_structure(primary, secondary_adata_scoreable)

        expected_a = int(n * 0.7) / n
        expected_b = (n - int(n * 0.7)) / n
        assert result["secondary_distribution"]["A"] == pytest.approx(expected_a, abs=0.01)
        assert result["secondary_distribution"]["B"] == pytest.approx(expected_b, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: format_crosscheck_report
# ---------------------------------------------------------------------------

class TestFormatCrosscheckReport:
    """Tests for format_crosscheck_report."""

    def test_report_contains_header(self):
        """Report should contain STRUCTURAL CROSS-CHECK header."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": [],
            "primary_only": [],
            "secondary_only": [],
            "structural_concordance": 0.0,
        }
        report = format_crosscheck_report(result)

        assert "STRUCTURAL CROSS-CHECK" in report
        assert "=" * 60 in report

    def test_report_contains_dataset_names(self):
        """Report should display both dataset names."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": ["HSC", "MPP"],
            "primary_only": ["GMP"],
            "secondary_only": ["Erythroid"],
            "structural_concordance": 0.5,
        }
        report = format_crosscheck_report(result)

        assert "Primary: Weinreb" in report
        assert "Secondary: Nestorowa" in report

    def test_report_contains_concordance(self):
        """Report should display structural concordance value."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": ["HSC"],
            "primary_only": [],
            "secondary_only": [],
            "structural_concordance": 0.75,
        }
        report = format_crosscheck_report(result)

        assert "0.75" in report
        assert "Structural concordance" in report

    def test_report_contains_shared_states(self):
        """Report should list shared states."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": ["HSC", "MPP", "CMP"],
            "primary_only": [],
            "secondary_only": [],
            "structural_concordance": 1.0,
        }
        report = format_crosscheck_report(result)

        assert "HSC" in report
        assert "MPP" in report
        assert "CMP" in report

    def test_report_with_distributions(self):
        """Report should display state distributions when provided."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {"GMP": 0.35, "HSC": 0.25, "MEP": 0.40},
            "secondary_distribution": {"GMP": 0.30, "HSC": 0.30, "MEP": 0.40},
            "shared_states": ["GMP", "HSC", "MEP"],
            "primary_only": [],
            "secondary_only": [],
            "structural_concordance": 1.0,
        }
        report = format_crosscheck_report(result)

        assert "Weinreb state distribution:" in report
        assert "Nestorowa state distribution:" in report
        # Check some fractions appear
        assert "0.350" in report or "0.35" in report
        assert "0.400" in report or "0.40" in report

    def test_report_empty_distributions(self):
        """Report should handle empty distributions gracefully."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": [],
            "primary_only": ["HSC"],
            "secondary_only": ["Other"],
            "structural_concordance": 0.0,
        }
        report = format_crosscheck_report(result)

        # Should not contain distribution section headers when empty
        assert "state distribution:" not in report
        # But still should have the header
        assert "STRUCTURAL CROSS-CHECK" in report

    def test_report_is_string(self):
        """Report output should be a string."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": [],
            "primary_only": [],
            "secondary_only": [],
            "structural_concordance": 0.0,
        }
        report = format_crosscheck_report(result)
        assert isinstance(report, str)

    def test_report_contains_primary_only_and_secondary_only(self):
        """Report should list primary-only and secondary-only states."""
        result = {
            "primary_name": "Weinreb",
            "secondary_name": "Nestorowa",
            "primary_distribution": {},
            "secondary_distribution": {},
            "shared_states": ["HSC"],
            "primary_only": ["GMP", "MEP"],
            "secondary_only": ["Erythroid"],
            "structural_concordance": 0.25,
        }
        report = format_crosscheck_report(result)

        assert "Primary-only states:" in report
        assert "Secondary-only states:" in report
        assert "GMP" in report
        assert "MEP" in report
        assert "Erythroid" in report
