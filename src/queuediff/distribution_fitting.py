"""Distribution fitting for residence times: exponential vs gamma MLE with AIC/BIC comparison.

This module is the statistical core that determines whether the semi-Markov
(gamma) model beats the classical (exponential) baseline for each state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


def extract_residence_times(cell_df: pd.DataFrame, state: int) -> np.ndarray:
    """Extract residence times for a specific state from simulated cell DataFrame.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        DataFrame from synthetic_generator.simulate_cells with columns:
        cell_id, state, entry_time, exit_time, next_state.
    state : int
        State index to filter for.

    Returns
    -------
    numpy.ndarray
        Array of residence times (exit_time - entry_time) for all rows
        where state matches. Empty array if state not found.
    """
    mask = cell_df['state'] == state
    if not mask.any():
        return np.array([])
    return (cell_df.loc[mask, 'exit_time'] - cell_df.loc[mask, 'entry_time']).to_numpy()


def fit_exponential(residence_times: np.ndarray) -> Dict[str, float]:
    """Fit exponential distribution to residence times via MLE (fixed loc=0).

    Parameters
    ----------
    residence_times : numpy.ndarray
        Array of non-negative residence times.

    Returns
    -------
    dict
        Contains: rate (1/scale), log_likelihood, aic, bic, n_obs.
        k=1 free parameter (rate).
        If fewer than 2 valid observations, returns dict with all numeric
        fields set to np.nan and n_obs set to actual count.
    """
    # Drop NaN values
    clean_times = residence_times[~np.isnan(residence_times)]
    n = len(clean_times)

    # If fewer than 2 points, return NaN-filled dict
    if n < 2:
        return {
            'rate': np.nan,
            'scale': np.nan,
            'log_likelihood': np.nan,
            'loglik': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'n_obs': n,
            'n': n
        }

    if n < 10:
        raise ValueError(f"Need at least 10 observations for MLE, got {n}")

    # scipy.stats.expon.fit with floc=0 fixes location at 0, returns (loc, scale)
    # Since we fix loc=0, scale is the MLE estimate of 1/rate
    _, scale = stats.expon.fit(clean_times, floc=0)
    rate = 1.0 / scale

    # Log-likelihood for exponential: sum(log(rate * exp(-rate * x)))
    # = n * log(rate) - rate * sum(x)
    log_likelihood = n * np.log(rate) - rate * np.sum(clean_times)

    k = 1  # one free parameter: rate
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return {
        'rate': rate,
        'scale': scale,
        'log_likelihood': log_likelihood,
        'loglik': log_likelihood,
        'aic': aic,
        'bic': bic,
        'n_obs': n,
        'n': n
    }


def fit_gamma(residence_times: np.ndarray) -> Dict[str, float]:
    """Fit gamma distribution to residence times via MLE (fixed loc=0).

    Parameters
    ----------
    residence_times : numpy.ndarray
        Array of non-negative residence times.

    Returns
    -------
    dict
        Contains: shape, scale, log_likelihood, aic, bic, n_obs.
        k=2 free parameters (shape and scale).
        If fewer than 2 valid observations, returns dict with all numeric
        fields set to np.nan and n_obs set to actual count.
    """
    # Drop NaN values
    clean_times = residence_times[~np.isnan(residence_times)]
    n = len(clean_times)

    # If fewer than 2 points, return NaN-filled dict
    if n < 2:
        return {
            'shape': np.nan,
            'scale': np.nan,
            'log_likelihood': np.nan,
            'loglik': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'n_obs': n,
            'n': n
        }

    if n < 10:
        raise ValueError(f"Need at least 10 observations for MLE, got {n}")

    # scipy.stats.gamma.fit with floc=0 fixes location at 0, returns (shape, loc, scale)
    shape, _, scale = stats.gamma.fit(clean_times, floc=0)

    # Log-likelihood for gamma using scipy's logpdf
    log_likelihood = np.sum(stats.gamma.logpdf(clean_times, shape, loc=0, scale=scale))

    k = 2  # two free parameters: shape and scale
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return {
        'shape': shape,
        'scale': scale,
        'log_likelihood': log_likelihood,
        'loglik': log_likelihood,
        'aic': aic,
        'bic': bic,
        'n_obs': n,
        'n': n
    }


def aic(log_likelihood: float, k: int) -> float:
    """Compute Akaike Information Criterion.

    AIC = 2k - 2*log_likelihood

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    k : int
        Number of free parameters in the model.

    Returns
    -------
    float
        AIC value (lower is better).
    """
    return 2 * k - 2 * log_likelihood


def bic(log_likelihood: float, k: int, n: int) -> float:
    """Compute Bayesian Information Criterion.

    BIC = k*ln(n) - 2*log_likelihood

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    k : int
        Number of free parameters in the model.
    n : int
        Number of observations.

    Returns
    -------
    float
        BIC value (lower is better).
    """
    return k * np.log(n) - 2 * log_likelihood


def fit_distributions_to_state(residence_times: np.ndarray) -> Dict:
    """Fit both exponential and gamma distributions to residence times.

    This is a convenience function that takes raw residence time data
    (not a cell DataFrame) and returns combined fit results.

    Parameters
    ----------
    residence_times : numpy.ndarray
        Array of non-negative residence times.

    Returns
    -------
    dict
        Keys: 'n_obs', 'gamma_shape', 'gamma_scale', 'gamma_loglik',
        'exp_rate', 'exp_loglik', 'gamma_aic', 'exp_aic',
        'gamma_bic', 'exp_bic', 'delta_aic', 'delta_bic'.
    """
    if len(residence_times) < 10:
        raise ValueError(f"Need at least 10 observations for MLE, got {len(residence_times)}")

    exp_fit = fit_exponential(residence_times)
    gamma_fit = fit_gamma(residence_times)

    return {
        'n_obs': exp_fit['n_obs'],
        'gamma_shape': gamma_fit['shape'],
        'gamma_scale': gamma_fit['scale'],
        'gamma_loglik': gamma_fit['log_likelihood'],
        'exp_rate': exp_fit['rate'],
        'exp_loglik': exp_fit['log_likelihood'],
        'gamma_aic': gamma_fit['aic'],
        'exp_aic': exp_fit['aic'],
        'gamma_bic': gamma_fit['bic'],
        'exp_bic': exp_fit['bic'],
        'delta_aic': exp_fit['aic'] - gamma_fit['aic'],
        'delta_bic': exp_fit['bic'] - gamma_fit['bic']
    }


def fit_state_distributions(cell_df: pd.DataFrame, state: int) -> Dict:
    """Fit both exponential and gamma distributions for a single state.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Simulated cell DataFrame from synthetic_generator.simulate_cells.
    state : int
        State index to fit distributions for.

    Returns
    -------
    dict
        Keys: 'exponential' (fit dict), 'gamma' (fit dict),
        'delta_aic' (exp_aic - gamma_aic; positive => gamma better),
        'delta_bic' (exp_bic - gamma_bic; positive => gamma better).
    """
    residence_times = extract_residence_times(cell_df, state)

    if len(residence_times) < 10:
        raise ValueError(f"State {state} has only {len(residence_times)} observations (< 10 minimum)")

    exp_fit = fit_exponential(residence_times)
    gamma_fit = fit_gamma(residence_times)

    return {
        'exponential': exp_fit,
        'gamma': gamma_fit,
        'delta_aic': exp_fit['aic'] - gamma_fit['aic'],
        'delta_bic': exp_fit['bic'] - gamma_fit['bic']
    }


def fit_all_states(
    cell_df: pd.DataFrame,
    state_col: str = 'state',
    time_col: Optional[str] = None
) -> pd.DataFrame:
    """Fit distributions for all unique states in the DataFrame.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        DataFrame containing residence time data.
    state_col : str, default='state'
        Column name containing state identifiers.
    time_col : str, optional
        Column name containing residence times. If None, tries to find
        'residence_time' column, otherwise computes residence time from
        'entry_time' and 'exit_time' columns (as produced by
        synthetic_generator.simulate_cells).

    Returns
    -------
    pandas.DataFrame
        One row per state with flattened fit results columns.
    """
    # Preserve order of first appearance instead of sorting
    _, idx = np.unique(cell_df[state_col], return_index=True)
    states = cell_df[state_col].iloc[np.sort(idx)].unique()
    rows = []

    for state in states:
        mask = cell_df[state_col] == state

        if time_col is not None:
            # User provided explicit residence time column
            residence_times = cell_df.loc[mask, time_col].to_numpy()
        elif 'residence_time' in cell_df.columns:
            # Auto-detect residence_time column
            residence_times = cell_df.loc[mask, 'residence_time'].to_numpy()
        elif 'entry_time' in cell_df.columns and 'exit_time' in cell_df.columns:
            # Default: compute residence time from entry_time and exit_time
            residence_times = (
                cell_df.loc[mask, 'exit_time'] - cell_df.loc[mask, 'entry_time']
            ).to_numpy()
        else:
            raise ValueError(
                "No residence time column found. Provide time_col or ensure "
                "DataFrame has 'residence_time' or 'entry_time'/'exit_time' columns."
            )

        if len(residence_times) < 10:
            raise ValueError(f"State {state} has only {len(residence_times)} observations (< 10 minimum)")

        fit_result = fit_distributions_to_state(residence_times)

        rows.append({
            'state': state,
            'n_obs': fit_result['n_obs'],
            'exp_rate': fit_result['exp_rate'],
            'exp_log_likelihood': fit_result['exp_loglik'],
            'exp_aic': fit_result['exp_aic'],
            'exp_bic': fit_result['exp_bic'],
            'exp_n_obs': fit_result['n_obs'],
            'gamma_shape': fit_result['gamma_shape'],
            'gamma_scale': fit_result['gamma_scale'],
            'gamma_log_likelihood': fit_result['gamma_loglik'],
            'gamma_aic': fit_result['gamma_aic'],
            'gamma_bic': fit_result['gamma_bic'],
            'gamma_n_obs': fit_result['n_obs'],
            'delta_aic': fit_result['delta_aic'],
            'delta_bic': fit_result['delta_bic']
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    """Sanity check: fit distributions to synthetic data with known bottleneck."""
    print("=" * 70)
    print("DISTRIBUTION FITTING SANITY CHECK")
    print("=" * 70)

    from queuediff.synthetic_generator import generate_hierarchy, simulate_cells

    # Same 5-state hierarchy as synthetic_generator's sanity check
    hierarchy = generate_hierarchy(
        n_states=5,
        branching_structure={1: [2, 3], 2: [4]},
        seed=42
    )

    gamma_params = {
        0: (2.0, 1.0),   # Stem: mean=2.0, shape=2.0
        1: (2.0, 1.5),   # Myeloid-Primed: mean=3.0, shape=2.0
        2: (3.0, 1.0),   # Lymphoid-Primed: mean=3.0, shape=3.0
        3: (2.0, 2.0),   # Myeloid-Committed: mean=4.0, shape=2.0
        4: (4.0, 1.0),   # Lymphoid-Committed: mean=4.0, shape=4.0
    }

    # Bottleneck at state 1 with severity 3.0 -> true shape = 2.0 * 3.0 = 6.0
    bottleneck_state = 1
    bottleneck_severity = 3.0
    true_shape_at_bottleneck = gamma_params[bottleneck_state][0] * bottleneck_severity

    df = simulate_cells(
        hierarchy=hierarchy,
        n_cells=2000,
        gamma_params_per_state=gamma_params,
        bottleneck_state=bottleneck_state,
        bottleneck_severity=bottleneck_severity,
        seed=123
    )

    # Fit all states
    fit_df = fit_all_states(df)
    print(f"\nFit results for {len(fit_df)} states:")
    print(fit_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    # Check 1: Fitted gamma shape at bottleneck state matches injected shape
    bottleneck_row = fit_df[fit_df['state'] == bottleneck_state].iloc[0]
    fitted_shape = bottleneck_row['gamma_shape']
    shape_error = abs(fitted_shape - true_shape_at_bottleneck) / true_shape_at_bottleneck
    print(f"\n1. Gamma shape at bottleneck state {bottleneck_state}:")
    print(f"   True injected shape: {true_shape_at_bottleneck:.2f}")
    print(f"   Fitted shape:        {fitted_shape:.2f}")
    print(f"   Relative error:      {shape_error*100:.1f}%")
    shape_pass = shape_error < 0.15  # within 15% is reasonable for MLE
    print(f"   PASS" if shape_pass else f"   FAIL")

    # Check 2: delta_aic favors gamma at bottleneck more strongly than other states
    print(f"\n2. Model comparison (delta_aic = exp_aic - gamma_aic; positive => gamma better):")
    for _, row in fit_df.iterrows():
        marker = " <-- BOTTLENECK" if row['state'] == bottleneck_state else ""
        print(f"   State {int(row['state'])}: delta_aic = {row['delta_aic']:.2f}, delta_bic = {row['delta_bic']:.2f}{marker}")

    bottleneck_delta_aic = bottleneck_row['delta_aic']
    other_states = fit_df[fit_df['state'] != bottleneck_state]
    max_other_delta_aic = other_states['delta_aic'].max()

    aic_check = bottleneck_delta_aic > max_other_delta_aic
    print(f"\n   Bottleneck delta_aic ({bottleneck_delta_aic:.2f}) > max other delta_aic ({max_other_delta_aic:.2f})?")
    print(f"   PASS" if aic_check else f"   FAIL")

    # Check 3: delta_bic also favors gamma at bottleneck
    bottleneck_delta_bic = bottleneck_row['delta_bic']
    max_other_delta_bic = other_states['delta_bic'].max()
    bic_check = bottleneck_delta_bic > max_other_delta_bic
    print(f"\n   Bottleneck delta_bic ({bottleneck_delta_bic:.2f}) > max other delta_bic ({max_other_delta_bic:.2f})?")
    print(f"   PASS" if bic_check else f"   FAIL")

    # Overall
    all_pass = shape_pass and aic_check and bic_check
    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)