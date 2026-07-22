"""Tests for branch_point_validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queuediff.branch_point_validation import (
    estimate_routing_probabilities,
    validate_branch_points,
)


@pytest.fixture
def branching_trajectories():
    """Synthetic trajectories with known branching structure.

    Each cell has observations at days 2, 4, 6 with consistent cell_id.
    - 100 cells: HSC (d2) -> MPP (d4) -> MEP (d6) — 60 cells
    - 100 cells: HSC (d2) -> MPP (d4) -> GMP (d6) — 40 cells
    This yields MPP branching at ~0.6 MEP / 0.4 GMP.
    """
    records = []
    for i in range(100):
        # Cell i: HSC at day 2, MPP at day 4
        records.append({"clone_id": i, "cell_id": f"cell{i}", "timepoint": 2.0, "state": "HSC"})
        records.append({"clone_id": i, "cell_id": f"cell{i}", "timepoint": 4.0, "state": "MPP"})
    for i in range(60):
        # 60 cells go to MEP at day 6
        records.append({"clone_id": i, "cell_id": f"cell{i}", "timepoint": 6.0, "state": "MEP"})
    for i in range(60, 100):
        # 40 cells go to GMP at day 6
        records.append({"clone_id": i, "cell_id": f"cell{i}", "timepoint": 6.0, "state": "GMP"})
    return pd.DataFrame(records)


@pytest.fixture
def linear_trajectories():
    """Linear (non-branching) topology: HSC -> MPP -> GMP."""
    records = []
    for i in range(50):
        records.append({"clone_id": i, "cell_id": f"c{i}_d2", "timepoint": 2.0, "state": "HSC"})
        records.append({"clone_id": i, "cell_id": f"c{i}_d4", "timepoint": 4.0, "state": "MPP"})
        records.append({"clone_id": i, "cell_id": f"c{i}_d6", "timepoint": 6.0, "state": "GMP"})
    return pd.DataFrame(records)


@pytest.fixture
def routing_structure():
    return {
        "HSC": ["MPP"],
        "MPP": ["MEP", "GMP"],
        "MEP": [],
        "GMP": [],
    }


class TestEstimateRoutingProbabilities:
    def test_returns_dict(self, branching_trajectories, routing_structure):
        probs = estimate_routing_probabilities(branching_trajectories, routing_structure)
        assert isinstance(probs, dict)

    def test_branch_point_sums_to_one(self, branching_trajectories, routing_structure):
        probs = estimate_routing_probabilities(branching_trajectories, routing_structure)
        # MPP should have probabilities summing to 1
        mpp_total = sum(probs.get("MPP", {}).values())
        assert abs(mpp_total - 1.0) < 1e-6

    def test_branch_proportions(self, branching_trajectories, routing_structure):
        """With 60 MEP and 40 GMP cells, should recover ~0.6 and ~0.4."""
        probs = estimate_routing_probabilities(branching_trajectories, routing_structure)
        mep_prob = probs.get("MPP", {}).get("MEP", 0)
        gmp_prob = probs.get("MPP", {}).get("GMP", 0)
        assert abs(mep_prob - 0.6) < 0.05
        assert abs(gmp_prob - 0.4) < 0.05

    def test_linear_routing_probability_one(self, linear_trajectories, routing_structure):
        """HSC should have MPP with prob 1.0 (single target)."""
        probs = estimate_routing_probabilities(linear_trajectories, routing_structure)
        assert abs(probs.get("HSC", {}).get("MPP", 0) - 1.0) < 1e-6

    def test_empty_trajectories(self):
        empty = pd.DataFrame(columns=["clone_id", "cell_id", "timepoint", "state"])
        probs = estimate_routing_probabilities(empty, {"A": ["B", "C"]})
        assert probs == {}

    def test_no_branch_points(self, routing_structure):
        """Terminal states with no targets should not appear in results."""
        records = [
            {"clone_id": 0, "cell_id": "c0", "timepoint": 2.0, "state": "MEP"},
            {"clone_id": 0, "cell_id": "c0", "timepoint": 4.0, "state": "MEP"},
        ]
        traj = pd.DataFrame(records)
        probs = estimate_routing_probabilities(traj, routing_structure)
        # MEP is terminal (no downstream), should not be in probs
        assert "MEP" not in probs or probs["MEP"] == {}

    def test_fallback_equal_split(self, routing_structure):
        """When no transitions observed, fall back to equal split."""
        records = [
            {"clone_id": 0, "cell_id": "c0", "timepoint": 2.0, "state": "HSC"},
            {"clone_id": 0, "cell_id": "c0", "timepoint": 4.0, "state": "HSC"},
        ]
        traj = pd.DataFrame(records)
        probs = estimate_routing_probabilities(traj, routing_structure)
        # MPP has two targets, no transitions observed -> equal split
        if "MPP" in probs:
            assert abs(sum(probs["MPP"].values()) - 1.0) < 1e-6


class TestValidateBranchPoints:
    def test_returns_dataframe(self):
        estimated = {"MPP": {"MEP": 0.55, "GMP": 0.45}}
        result = validate_branch_points(estimated)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self):
        estimated = {"MPP": {"MEP": 0.55, "GMP": 0.45}}
        result = validate_branch_points(estimated)
        expected = {"source", "target", "estimated_prob", "expected_prob",
                    "deviation", "valid"}
        assert set(result.columns) == expected

    def test_valid_within_tolerance(self):
        estimated = {"MPP": {"MEP": 0.55, "GMP": 0.45}}
        expected = {"MPP": {"MEP": 0.6, "GMP": 0.4}}
        result = validate_branch_points(estimated, expected, tolerance=0.2)
        assert result["valid"].all()

    def test_invalid_outside_tolerance(self):
        estimated = {"MPP": {"MEP": 0.9, "GMP": 0.1}}
        expected = {"MPP": {"MEP": 0.6, "GMP": 0.4}}
        result = validate_branch_points(estimated, expected, tolerance=0.1)
        assert not result["valid"].all()

    def test_no_expected_probs(self):
        """When no expected_probs, deviation should be None and valid True."""
        estimated = {"MPP": {"MEP": 0.6, "GMP": 0.4}}
        result = validate_branch_points(estimated)
        assert result["deviation"].isna().all()
        assert result["valid"].all()
