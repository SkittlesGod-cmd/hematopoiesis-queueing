"""Tests for bottleneck_diagnostics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queuediff.bottleneck_diagnostics import (
    compute_bottleneck_ranking,
    generate_bottleneck_report,
)


@pytest.fixture
def traffic_intensities():
    """Example traffic intensities with GMP as highest."""
    return {
        "HSC": 0.15,
        "MPP": 0.20,
        "LMPP": 0.05,
        "CMP": 0.25,
        "MEP": 0.30,
        "GMP": 0.45,
    }


@pytest.fixture
def model_comparison_all_gamma():
    """All states gamma-preferred."""
    return pd.DataFrame({
        "state": ["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"],
        "gamma_preferred": [True, True, True, True, True, True],
        "delta_aic": [50.0, 40.0, 30.0, 60.0, 70.0, 80.0],
        "fdr_pvalue": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        "gamma_shape": [15.0, 12.0, 8.0, 10.0, 18.0, 20.0],
        "gamma_mean": [16.5, 13.2, 8.4, 10.5, 18.4, 19.4],
    })


@pytest.fixture
def model_comparison_mixed():
    """Some states gamma-preferred, some not."""
    return pd.DataFrame({
        "state": ["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"],
        "gamma_preferred": [False, True, False, True, True, True],
        "delta_aic": [1.0, 45.0, 0.5, 55.0, 65.0, 75.0],
        "fdr_pvalue": [0.50, 0.001, 0.60, 0.001, 0.001, 0.001],
        "gamma_shape": [2.0, 12.0, 1.5, 10.0, 18.0, 20.0],
        "gamma_mean": [16.5, 13.2, 8.4, 10.5, 18.4, 19.4],
    })


@pytest.fixture
def residence_summary():
    """Example residence time summary."""
    return pd.DataFrame({
        "state": ["CMP", "GMP", "HSC", "LMPP", "MEP", "MPP"],
        "mean_hours": [10.5, 19.4, 16.6, 8.4, 18.4, 13.0],
        "std_hours": [3.0, 5.0, 4.0, 2.0, 4.5, 3.5],
        "n_observations": [100, 80, 120, 90, 70, 110],
    })


class TestComputeBottleneckRanking:
    def test_returns_dataframe(self, traffic_intensities, model_comparison_all_gamma):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        assert isinstance(ranking, pd.DataFrame)

    def test_ranked_by_traffic_intensity_desc(self, traffic_intensities, model_comparison_all_gamma):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        values = ranking["traffic_intensity"].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_primary_bottleneck_is_highest_rho_gamma(self, traffic_intensities, model_comparison_all_gamma):
        """Primary bottleneck should be the gamma-preferred state with highest ρ."""
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        primary = ranking[ranking["is_primary_bottleneck"]]
        assert len(primary) == 1
        assert primary.iloc[0]["state"] == "GMP"

    def test_no_primary_when_no_gamma_preferred(self, traffic_intensities):
        """When no states are gamma-preferred, no primary bottleneck."""
        mc = pd.DataFrame({
            "state": ["HSC", "MPP", "LMPP", "CMP", "MEP", "GMP"],
            "gamma_preferred": [False] * 6,
            "delta_aic": [1.0] * 6,
            "fdr_pvalue": [0.50] * 6,
            "gamma_shape": [1.0] * 6,
            "gamma_mean": [10.0] * 6,
        })
        ranking = compute_bottleneck_ranking(traffic_intensities, mc)
        assert not ranking["is_primary_bottleneck"].any()

    def test_primary_bottleneck_must_be_gamma(self, traffic_intensities, model_comparison_mixed):
        """Even if LMPP has higher ρ than GMP, only gamma-preferred states qualify."""
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_mixed)
        primary = ranking[ranking["is_primary_bottleneck"]]
        assert len(primary) == 1
        assert primary.iloc[0]["gamma_preferred"]  # Must be gamma-preferred

    def test_correct_columns(self, traffic_intensities, model_comparison_all_gamma):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        expected = {"state", "traffic_intensity", "gamma_preferred",
                    "delta_aic", "fdr_pvalue", "is_primary_bottleneck",
                    "gamma_shape", "gamma_mean"}
        assert expected.issubset(set(ranking.columns))

    def test_empty_model_comparison(self, traffic_intensities):
        """When model_comparison has no 'state' column, gamma_preferred defaults False."""
        empty = pd.DataFrame()
        ranking = compute_bottleneck_ranking(traffic_intensities, empty)
        assert not ranking["gamma_preferred"].any()
        assert not ranking["is_primary_bottleneck"].any()

    def test_single_state_network(self):
        """Single state should be its own bottleneck if gamma-preferred."""
        rho = {"HSC": 1.0}
        mc = pd.DataFrame({
            "state": ["HSC"],
            "gamma_preferred": [True],
            "delta_aic": [50.0],
            "fdr_pvalue": [0.001],
            "gamma_shape": [15.0],
            "gamma_mean": [16.5],
        })
        ranking = compute_bottleneck_ranking(rho, mc)
        assert ranking.iloc[0]["is_primary_bottleneck"]


class TestGenerateBottleneckReport:
    def test_returns_string(self, traffic_intensities, model_comparison_all_gamma):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        report = generate_bottleneck_report(ranking)
        assert isinstance(report, str)

    def test_contains_bottleneck_state(self, traffic_intensities, model_comparison_all_gamma):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        report = generate_bottleneck_report(ranking)
        assert "GMP" in report

    def test_includes_residence_times(self, traffic_intensities, model_comparison_all_gamma,
                                      residence_summary):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        report = generate_bottleneck_report(ranking, residence_summary)
        assert "RESIDENCE TIME" in report

    def test_network_name_in_header(self, traffic_intensities, model_comparison_all_gamma):
        ranking = compute_bottleneck_ranking(traffic_intensities, model_comparison_all_gamma)
        report = generate_bottleneck_report(ranking, network_name="Test Network")
        assert "Test Network" in report

    def test_honest_when_no_bottleneck(self, traffic_intensities):
        """Report should say NONE when no gamma-preferred states exist."""
        mc = pd.DataFrame({
            "state": ["HSC"],
            "gamma_preferred": [False],
            "delta_aic": [0.5],
            "fdr_pvalue": [0.50],
            "gamma_shape": [1.0],
            "gamma_mean": [10.0],
        })
        ranking = compute_bottleneck_ranking(traffic_intensities, mc)
        report = generate_bottleneck_report(ranking)
        assert "NONE" in report or "none" in report.lower()
