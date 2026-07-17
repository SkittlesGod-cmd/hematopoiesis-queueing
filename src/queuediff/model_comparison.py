"""Model comparison and statistical significance testing for residence time distributions.

This module adds likelihood-ratio tests and FDR correction on top of
AIC/BIC values from distribution_fitting.py to identify statistically
significant bottlenecks (departures from the exponential/memoryless assumption).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
from statsmodels.stats.multitest import multipletests


def likelihood_ratio_test(exp_loglik: float, gamma_loglik: float, df: int = 1) -> Dict:
    """Compute likelihood-ratio test comparing exponential vs gamma.

    The exponential distribution is a special case of gamma (shape=1),
    making them nested models. Under the null (exponential is true),
    LR = 2 * (gamma_loglik - exp_loglik) follows chi-squared(df=1).

    Parameters
    ----------
    exp_loglik : float
        Log-likelihood of the exponential fit.
    gamma_loglik : float
        Log-likelihood of the gamma fit.
    df : int, default=1
        Degrees of freedom for the chi-squared distribution.
        Fixed at 1 because gamma has 1 more free parameter than exponential.

    Returns
    -------
    dict
        Contains: lr_statistic (float), p_value (float), df (int).
        If LR < 0 (numerical noise when true model is exponential),
        clips to 0 and returns p_value = 1.0.
    """
    lr = 2.0 * (gamma_loglik - exp_loglik)

    if lr < 0:
        # Numerical noise when exponential is true; clip to boundary
        lr = 0.0
        p_value = 1.0
    else:
        p_value = stats.chi2.sf(lr, df)

    return {
        'lr_statistic': lr,
        'p_value': p_value,
        'df': df
    }


def compare_all_states(fit_all_states_df: pd.DataFrame) -> pd.DataFrame:
    """Run likelihood-ratio test for each state in the fit results.

    Parameters
    ----------
    fit_all_states_df : pandas.DataFrame
        Output from distribution_fitting.fit_all_states with columns
        exp_log_likelihood and gamma_log_likelihood.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added columns: lr_statistic, p_value.
    """
    required_cols = {'exp_log_likelihood', 'gamma_log_likelihood', 'state'}
    missing = required_cols - set(fit_all_states_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = fit_all_states_df.copy()
    lr_results = []

    for _, row in df.iterrows():
        lr = likelihood_ratio_test(row['exp_log_likelihood'], row['gamma_log_likelihood'])
        lr_results.append(lr)

    lr_df = pd.DataFrame(lr_results)
    return pd.concat([df, lr_df], axis=1)


def apply_fdr_correction(comparison_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to LRT p-values.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Output from compare_all_states with a 'p_value' column.
    alpha : float, default=0.05
        FDR significance threshold.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added columns:
        p_value_corrected (float), significant (bool).
    """
    if 'p_value' not in comparison_df.columns:
        raise ValueError("Input DataFrame must have 'p_value' column")

    df = comparison_df.copy()
    p_values = df['p_value'].to_numpy()

    # statsmodels multipletests returns (reject, pvals_corrected, ...)
    _, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    df['p_value_corrected'] = pvals_corrected
    df['significant'] = pvals_corrected < alpha

    return df


def identify_significant_bottlenecks(comparison_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Identify statistically significant bottlenecks.

    Convenience function that runs FDR correction and filters to
    significant states, sorted by effect size (delta_aic descending).

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Output from compare_all_states.
    alpha : float, default=0.05
        FDR significance threshold.

    Returns
    -------
    pandas.DataFrame
        Subset of states where significant=True, sorted by delta_aic descending.
        Returns empty DataFrame if no significant states.
    """
    corrected = apply_fdr_correction(comparison_df, alpha)
    significant = corrected[corrected['significant']].copy()

    if len(significant) == 0:
        return pd.DataFrame(columns=corrected.columns)

    return significant.sort_values('delta_aic', ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    """Validation: test LRT + FDR on synthetic data with known bottleneck."""
    print("=" * 80)
    print("MODEL COMPARISON VALIDATION")
    print("=" * 80)

    from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
    from queuediff.distribution_fitting import fit_all_states

    # Same 5-state hierarchy as distribution_fitting's sanity check
    hierarchy = generate_hierarchy(
        n_states=5,
        branching_structure={1: [2, 3], 2: [4]},
        seed=42
    )

    # Base gamma params (shapes > 1 mean gamma is truly better than exponential)
    gamma_params = {
        0: (2.0, 1.0),   # Stem: mean=2.0, shape=2.0
        1: (2.0, 1.5),   # Myeloid-Primed: mean=3.0, shape=2.0
        2: (3.0, 1.0),   # Lymphoid-Primed: mean=3.0, shape=3.0
        3: (2.0, 2.0),   # Myeloid-Committed: mean=4.0, shape=2.0
        4: (4.0, 1.0),   # Lymphoid-Committed: mean=4.0, shape=4.0
    }

    bottleneck_state = 1
    bottleneck_severity = 3.0
    true_shape_at_bottleneck = gamma_params[bottleneck_state][0] * bottleneck_severity

    print(f"\n--- TEST 1: WITH BOTTLENECK (state {bottleneck_state}, severity={bottleneck_severity}) ---")

    df = simulate_cells(
        hierarchy=hierarchy,
        n_cells=2000,
        gamma_params_per_state=gamma_params,
        bottleneck_state=bottleneck_state,
        bottleneck_severity=bottleneck_severity,
        seed=123
    )

    # Fit distributions
    fit_df = fit_all_states(df)

    # Run LRT comparison
    comparison_df = compare_all_states(fit_df)

    # Apply FDR correction
    corrected_df = apply_fdr_correction(comparison_df, alpha=0.05)

    # Print full comparison table
    print("\nFull comparison table (all states):")
    display_cols = ['state', 'delta_aic', 'lr_statistic', 'p_value',
                    'p_value_corrected', 'significant']
    print(corrected_df[display_cols].to_string(index=False))

    # Check 1: bottleneck state has the largest delta_aic
    max_delta_aic_state = corrected_df.loc[corrected_df['delta_aic'].idxmax(), 'state']
    has_max_delta_aic = (max_delta_aic_state == bottleneck_state)

    # Check 2: bottleneck state has the highest LR statistic
    max_lr_state = corrected_df.loc[corrected_df['lr_statistic'].idxmax(), 'state']
    has_max_lr = (max_lr_state == bottleneck_state)

    # Check 3: bottleneck state has the smallest p-value
    min_p_state = corrected_df.loc[corrected_df['p_value'].idxmin(), 'state']
    has_min_p = (min_p_state == bottleneck_state)

    # Check 4: bottleneck state is significant after FDR
    bottleneck_row = corrected_df[corrected_df['state'] == bottleneck_state].iloc[0]
    bottleneck_significant = bottleneck_row['significant']

    print(f"\nValidation for bottleneck state {bottleneck_state}:")
    print(f"  Largest delta_aic?     {has_max_delta_aic} (state {max_delta_aic_state})")
    print(f"  Largest LR statistic?  {has_max_lr} (state {max_lr_state})")
    print(f"  Smallest p-value?      {has_min_p} (state {min_p_state})")
    print(f"  Significant after FDR? {bottleneck_significant}")

    test1_pass = has_max_delta_aic and has_max_lr and has_min_p and bottleneck_significant
    print(f"  TEST 1 RESULT: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: Negative control - all exponential (shape=1.0 everywhere)
    # When data is truly exponential, gamma should NOT be significantly preferred
    print(f"\n--- TEST 2: NEGATIVE CONTROL (ALL STATES TRULY EXPONENTIAL, shape=1.0) ---")

    exp_params = {s: (1.0, gamma_params[s][1]) for s in gamma_params}  # shape=1 = exponential

    df_null = simulate_cells(
        hierarchy=hierarchy,
        n_cells=2000,
        gamma_params_per_state=exp_params,
        bottleneck_state=bottleneck_state,
        bottleneck_severity=1.0,  # No extra bottleneck
        seed=456
    )

    fit_df_null = fit_all_states(df_null)
    comparison_df_null = compare_all_states(fit_df_null)
    corrected_df_null = apply_fdr_correction(comparison_df_null, alpha=0.05)

    print("\nFull comparison table (negative control):")
    print(corrected_df_null[display_cols].to_string(index=False))

    n_significant = corrected_df_null['significant'].sum()
    n_states = len(corrected_df_null)
    false_positive_rate = n_significant / n_states

    print(f"\nValidation for negative control:")
    print(f"  Significant states: {n_significant} / {n_states}")
    print(f"  False positive rate: {false_positive_rate:.3f} (expected ~{0.05})")

    # Pass if FPR is controlled (typically 0 or 1 significant by chance)
    test2_pass = n_significant <= 1
    print(f"  TEST 2 RESULT: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: No bottleneck but gamma data (all states gamma, no extra severity)
    # In this case, all states should have similar delta_aic, no single outlier
    print(f"\n--- TEST 3: NO BOTTLENECK, GAMMA DATA (all states gamma, uniform) ---")

    gamma_params_uniform = {s: (2.0, gamma_params[s][1]) for s in gamma_params}  # all shape=2.0

    df_uniform = simulate_cells(
        hierarchy=hierarchy,
        n_cells=2000,
        gamma_params_per_state=gamma_params_uniform,
        bottleneck_state=bottleneck_state,
        bottleneck_severity=1.0,
        seed=789
    )

    fit_df_uniform = fit_all_states(df_uniform)
    comparison_df_uniform = compare_all_states(fit_df_uniform)
    corrected_df_uniform = apply_fdr_correction(comparison_df_uniform, alpha=0.05)

    print("\nFull comparison table (uniform gamma):")
    print(corrected_df_uniform[display_cols].to_string(index=False))

    # Check: no single state should be an extreme outlier in delta_aic
    # The ratio of max to min delta_aic should be reasonable
    delta_aic_max = corrected_df_uniform['delta_aic'].max()
    delta_aic_min = corrected_df_uniform['delta_aic'].min()
    delta_aic_ratio = delta_aic_max / delta_aic_min if delta_aic_min > 0 else np.inf

    # With bottleneck, ratio is ~2158/174 = 12.4
    # Without bottleneck, should be much smaller (similar effect sizes)
    print(f"\nValidation for uniform gamma:")
    print(f"  delta_aic range: {delta_aic_min:.1f} - {delta_aic_max:.1f}")
    print(f"  Max/min ratio:   {delta_aic_ratio:.2f}")

    # Ratio should be small (states have similar evidence for gamma)
    test3_pass = delta_aic_ratio < 5.0  # heuristic threshold
    print(f"  TEST 3 RESULT: {'PASS' if test3_pass else 'FAIL'}")

    print("\n" + "=" * 80)
    print(f"OVERALL: {'PASS' if test1_pass and test2_pass and test3_pass else 'FAIL'}")
    print("=" * 80)