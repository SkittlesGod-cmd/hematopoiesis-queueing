import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# likelihood_ratio_test  —  LRT comparing exponential vs gamma
# ---------------------------------------------------------------------------

class TestLikelihoodRatioTest:
    def test_lr_statistic_non_negative(self):
        from queuediff.model_comparison import likelihood_ratio_test
        result = likelihood_ratio_test(exp_loglik=-100.0, gamma_loglik=-90.0)
        assert result["lr_statistic"] >= 0

    def test_p_value_in_unit_interval(self):
        from queuediff.model_comparison import likelihood_ratio_test
        result = likelihood_ratio_test(exp_loglik=-100.0, gamma_loglik=-90.0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_correct_chi_squared_computation(self):
        from queuediff.model_comparison import likelihood_ratio_test
        # For LR=2*(gamma - exp) = 2*(-90 - -100) = 20
        # p = chi2.sf(20, df=1)
        expected_lr = 20.0
        expected_p = stats.chi2.sf(expected_lr, df=1)
        result = likelihood_ratio_test(exp_loglik=-100.0, gamma_loglik=-90.0)
        assert abs(result["lr_statistic"] - expected_lr) < 1e-10
        assert abs(result["p_value"] - expected_p) < 1e-10

    def test_negative_lr_clips_to_zero(self):
        from queuediff.model_comparison import likelihood_ratio_test
        # gamma_loglik < exp_loglik (numerical noise when exponential is true)
        result = likelihood_ratio_test(exp_loglik=-90.0, gamma_loglik=-100.0)
        assert result["lr_statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_df_returned_correctly(self):
        from queuediff.model_comparison import likelihood_ratio_test
        result = likelihood_ratio_test(exp_loglik=-100.0, gamma_loglik=-90.0, df=1)
        assert result["df"] == 1

    def test_custom_df_allowed(self):
        from queuediff.model_comparison import likelihood_ratio_test
        result = likelihood_ratio_test(exp_loglik=-100.0, gamma_loglik=-90.0, df=2)
        assert result["df"] == 2
        expected_p = stats.chi2.sf(20.0, df=2)
        assert abs(result["p_value"] - expected_p) < 1e-10


# ---------------------------------------------------------------------------
# compare_all_states  —  adds LRT columns to fit_all_states output
# ---------------------------------------------------------------------------

class TestCompareAllStates:
    def test_adds_lr_and_p_columns(self):
        from queuediff.model_comparison import compare_all_states
        df = pd.DataFrame({
            "state": [0, 1],
            "exp_log_likelihood": [-100.0, -200.0],
            "gamma_log_likelihood": [-90.0, -190.0],
            "delta_aic": [10.0, 20.0],
            "delta_bic": [8.0, 18.0],
        })
        result = compare_all_states(df)
        assert "lr_statistic" in result.columns
        assert "p_value" in result.columns

    def test_preserves_input_columns(self):
        from queuediff.model_comparison import compare_all_states
        df = pd.DataFrame({
            "state": [0, 1],
            "exp_log_likelihood": [-100.0, -200.0],
            "gamma_log_likelihood": [-90.0, -190.0],
            "delta_aic": [10.0, 20.0],
            "delta_bic": [8.0, 18.0],
            "custom_col": ["a", "b"],
        })
        result = compare_all_states(df)
        assert "custom_col" in result.columns
        assert list(result["custom_col"]) == ["a", "b"]

    def test_preserves_row_count_and_order(self):
        from queuediff.model_comparison import compare_all_states
        df = pd.DataFrame({
            "state": [2, 0, 1],
            "exp_log_likelihood": [-100.0, -200.0, -150.0],
            "gamma_log_likelihood": [-90.0, -190.0, -140.0],
        })
        result = compare_all_states(df)
        assert len(result) == 3
        assert list(result["state"]) == [2, 0, 1]

    def test_raises_on_missing_required_columns(self):
        from queuediff.model_comparison import compare_all_states
        df = pd.DataFrame({"state": [0, 1]})  # missing log-likelihood cols
        with pytest.raises(ValueError, match="Missing required columns"):
            compare_all_states(df)


# ---------------------------------------------------------------------------
# apply_fdr_correction  —  Benjamini-Hochberg correction
# ---------------------------------------------------------------------------

class TestApplyFDRCorrection:
    def test_adds_corrected_and_significant_columns(self):
        from queuediff.model_comparison import apply_fdr_correction
        df = pd.DataFrame({
            "state": [0, 1, 2],
            "p_value": [0.01, 0.03, 0.2],
        })
        result = apply_fdr_correction(df, alpha=0.05)
        assert "p_value_corrected" in result.columns
        assert "significant" in result.columns

    def test_corrected_p_value_ge_original(self):
        from queuediff.model_comparison import apply_fdr_correction
        df = pd.DataFrame({
            "state": [0, 1, 2],
            "p_value": [0.01, 0.03, 0.2],
        })
        result = apply_fdr_correction(df, alpha=0.05)
        # BH correction is conservative: p_corrected >= p_original
        for orig, corr in zip(df["p_value"], result["p_value_corrected"]):
            assert corr >= orig - 1e-10

    def test_significant_column_is_boolean(self):
        from queuediff.model_comparison import apply_fdr_correction
        df = pd.DataFrame({
            "state": [0, 1],
            "p_value": [0.01, 0.2],
        })
        result = apply_fdr_correction(df, alpha=0.05)
        assert result["significant"].dtype == bool
        assert result["significant"].iloc[0] == True
        assert result["significant"].iloc[1] == False

    def test_behavior_at_alpha_boundary(self):
        from queuediff.model_comparison import apply_fdr_correction
        # p_value exactly at alpha should not be significant (< alpha, not <=)
        df = pd.DataFrame({
            "state": [0],
            "p_value": [0.05],
        })
        result = apply_fdr_correction(df, alpha=0.05)
        # With 1 test, corrected p = 0.05, significant = (0.05 < 0.05) = False
        assert result["significant"].iloc[0] == False

    def test_multiple_tests_bh_ordering(self):
        from queuediff.model_comparison import apply_fdr_correction
        df = pd.DataFrame({
            "state": [0, 1, 2, 3, 4],
            "p_value": [0.001, 0.01, 0.03, 0.04, 0.2],
        })
        result = apply_fdr_correction(df, alpha=0.05)
        # BH: sorted p = [0.001, 0.01, 0.03, 0.04, 0.2]
        # critical = [0.01, 0.02, 0.03, 0.04, 0.05] (alpha * i / m)
        # 0.001 <= 0.01: significant
        # 0.01 <= 0.02: significant
        # 0.03 <= 0.03: significant
        # 0.04 <= 0.04: significant
        # 0.2 > 0.05: not significant
        assert result["significant"].sum() == 4

    def test_raises_on_missing_p_value_column(self):
        from queuediff.model_comparison import apply_fdr_correction
        df = pd.DataFrame({"state": [0, 1]})
        with pytest.raises(ValueError, match="must have 'p_value' column"):
            apply_fdr_correction(df)

    def test_preserves_other_columns(self):
        from queuediff.model_comparison import apply_fdr_correction
        df = pd.DataFrame({
            "state": [0, 1],
            "p_value": [0.01, 0.2],
            "delta_aic": [10.0, 5.0],
            "custom": ["x", "y"],
        })
        result = apply_fdr_correction(df, alpha=0.05)
        assert "delta_aic" in result.columns
        assert "custom" in result.columns
        assert list(result["delta_aic"]) == [10.0, 5.0]


# ---------------------------------------------------------------------------
# identify_significant_bottlenecks  —  filter and sort significant states
# ---------------------------------------------------------------------------

class TestIdentifySignificantBottlenecks:
    def test_returns_only_significant_rows(self):
        from queuediff.model_comparison import identify_significant_bottlenecks
        df = pd.DataFrame({
            "state": [0, 1, 2, 3],
            "delta_aic": [10.0, 100.0, 50.0, 5.0],
            "p_value": [0.2, 0.001, 0.01, 0.5],
        })
        result = identify_significant_bottlenecks(df, alpha=0.05)
        assert len(result) == 2
        assert set(result["state"]) == {1, 2}

    def test_sorted_by_delta_aic_descending(self):
        from queuediff.model_comparison import identify_significant_bottlenecks
        df = pd.DataFrame({
            "state": [0, 1, 2],
            "delta_aic": [10.0, 100.0, 50.0],
            "p_value": [0.2, 0.001, 0.01],
        })
        result = identify_significant_bottlenecks(df, alpha=0.05)
        assert list(result["delta_aic"]) == [100.0, 50.0]

    def test_returns_empty_dataframe_when_none_significant(self):
        from queuediff.model_comparison import identify_significant_bottlenecks
        df = pd.DataFrame({
            "state": [0, 1],
            "delta_aic": [10.0, 5.0],
            "p_value": [0.2, 0.5],
        })
        result = identify_significant_bottlenecks(df, alpha=0.05)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Should have same columns as input plus correction columns
        assert "p_value_corrected" in result.columns
        assert "significant" in result.columns

    def test_runs_fdr_internally_if_not_already_corrected(self):
        from queuediff.model_comparison import identify_significant_bottlenecks
        df = pd.DataFrame({
            "state": [0, 1, 2],
            "delta_aic": [10.0, 100.0, 50.0],
            "p_value": [0.2, 0.001, 0.01],
            # No 'significant' or 'p_value_corrected' columns
        })
        result = identify_significant_bottlenecks(df, alpha=0.05)
        assert len(result) == 2
        assert set(result["state"]) == {1, 2}

    @pytest.mark.skip(reason="requires end-to-end pipeline: generate hierarchy → simulate → fit → compare")
    def test_end_to_end_bottleneck_detection_on_synthetic_data(self):
        pass


# ---------------------------------------------------------------------------
# Integration with distribution_fitting
# ---------------------------------------------------------------------------

class TestIntegrationWithDistributionFitting:
    @pytest.mark.skip(reason="awaiting full pipeline integration test")
    def test_full_pipeline_synthetic_bottleneck_recovery(self):
        pass

    @pytest.mark.skip(reason="awaiting full pipeline integration test")
    def test_full_pipeline_fdr_control_on_null_data(self):
        pass