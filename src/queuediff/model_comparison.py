"""Model comparison: gamma vs exponential with multiple testing correction.

Compares gamma and exponential fits across multiple states using AIC/BIC
and likelihood ratio tests with Benjamini-Hochberg FDR correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from queuediff.distribution_fitting import (
    FitResult,
    fit_exponential,
    fit_gamma,
    likelihood_ratio_test,
)


def compare_models_per_state(
    residence_times: dict[str, np.ndarray],
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compare gamma vs exponential fits for each state.

    Parameters
    ----------
    residence_times : dict[str, ndarray]
        State name -> array of residence times (hours).
    min_samples : int, default 10
        Minimum number of samples required for reliable fitting.

    Returns
    -------
    pd.DataFrame
        One row per state with columns:
        - state: state name
        - n_samples: number of residence time observations
        - gamma_shape, gamma_scale, gamma_mean: gamma fit parameters
        - exp_scale, exp_mean: exponential fit parameters
        - gamma_aic, exp_aic, delta_aic: AIC values and difference
        - gamma_bic, exp_bic, delta_bic: BIC values and difference
        - lr_statistic, lr_pvalue: likelihood ratio test results
        - gamma_loglik, exp_loglik: log-likelihoods

    Notes
    -----
    delta_aic = exp_aic - gamma_aic (positive favors gamma).
    delta_bic = exp_bic - gamma_bic (positive favors gamma).
    States with fewer than min_samples observations are skipped with a warning.
    """
    results = []

    for state, times in residence_times.items():
        times = np.asarray(times, dtype=np.float64)

        if len(times) < min_samples:
            continue

        # Fit both distributions
        gamma_fit = fit_gamma(times)
        exp_fit = fit_exponential(times)

        # Likelihood ratio test
        lr_stat, lr_p = likelihood_ratio_test(exp_fit, gamma_fit)

        results.append({
            "state": state,
            "n_samples": len(times),
            "gamma_shape": gamma_fit.params["shape"],
            "gamma_scale": gamma_fit.params["scale"],
            "gamma_mean": gamma_fit.mean,
            "gamma_variance": gamma_fit.variance,
            "exp_scale": exp_fit.params["scale"],
            "exp_mean": exp_fit.mean,
            "gamma_aic": gamma_fit.aic,
            "exp_aic": exp_fit.aic,
            "delta_aic": exp_fit.aic - gamma_fit.aic,
            "gamma_bic": gamma_fit.bic,
            "exp_bic": exp_fit.bic,
            "delta_bic": exp_fit.bic - gamma_fit.bic,
            "lr_statistic": lr_stat,
            "lr_pvalue": lr_p,
            "gamma_loglik": gamma_fit.loglik,
            "exp_loglik": exp_fit.loglik,
        })

    return pd.DataFrame(results)


def apply_fdr_correction(
    comparison_df: pd.DataFrame,
    alpha: float = 0.05,
    delta_aic_threshold: float = 2.0,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to model comparison results.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_models_per_state.
    alpha : float, default 0.05
        FDR significance level.
    delta_aic_threshold : float, default 2.0
        Minimum delta AIC to prefer gamma over exponential.

    Returns
    -------
    pd.DataFrame
        Input dataframe augmented with:
        - fdr_pvalue: BH-corrected p-values
        - gamma_preferred: True if delta_aic > threshold AND fdr_p < alpha

    Notes
    -----
    Gamma is preferred when BOTH conditions hold:
    1. delta_aic > delta_aic_threshold (meaningful improvement in fit)
    2. fdr_pvalue < alpha (statistically significant after multiple testing)
    """
    df = comparison_df.copy()

    if len(df) == 0:
        df["fdr_pvalue"] = pd.Series(dtype=float)
        df["gamma_preferred"] = pd.Series(dtype=bool)
        return df

    # BH FDR correction across states
    _, fdr_pvalues, _, _ = multipletests(
        df["lr_pvalue"].values,
        alpha=alpha,
        method="fdr_bh",
    )

    df["fdr_pvalue"] = fdr_pvalues

    # Gamma preferred: both criteria must be met
    df["gamma_preferred"] = (
        (df["delta_aic"] > delta_aic_threshold) &
        (df["fdr_pvalue"] < alpha)
    )

    return df


def summarize_model_comparison(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of model comparison results.

    Parameters
    ----------
    df : pd.DataFrame
        Output from apply_fdr_correction.

    Returns
    -------
    str
        Formatted text summary for terminal output.
    """
    lines = [
        "=" * 70,
        "MODEL COMPARISON: GAMMA vs EXPONENTIAL SERVICE TIMES",
        "=" * 70,
        "",
    ]

    n_gamma = df["gamma_preferred"].sum()
    n_total = len(df)
    lines.append(f"States analyzed: {n_total}")
    lines.append(f"States preferring gamma: {n_gamma}/{n_total}")
    lines.append("")

    lines.append(f"{'State':<8} {'n':>6} {'γ shape':>8} {'γ mean':>8} "
                 f"{'ΔAIC':>8} {'FDR p':>10} {'Prefer':>8}")
    lines.append("-" * 70)

    for _, row in df.iterrows():
        prefer = "GAMMA" if row["gamma_preferred"] else "exp"
        lines.append(
            f"{row['state']:<8} {row['n_samples']:>6} "
            f"{row['gamma_shape']:>8.1f} {row['gamma_mean']:>8.1f}h "
            f"{row['delta_aic']:>8.0f} {row['fdr_pvalue']:>10.2e} "
            f"{prefer:>8}"
        )

    lines.append("")
    if n_gamma == n_total and n_total > 0:
        lines.append(
            "CONCLUSION: All states show gamma-preferred service times,\n"
            "confirming the semi-Markov (non-exponential) queueing model."
        )
    elif n_gamma > 0:
        gamma_states = list(df[df["gamma_preferred"]]["state"])
        lines.append(
            f"CONCLUSION: {gamma_states} show gamma-preferred service times.\n"
            f"Partial support for the semi-Markov queueing model."
        )
    else:
        lines.append(
            "CONCLUSION: No states show gamma-preferred service times.\n"
            "Exponential (Markov) model is sufficient for this data."
        )

    lines.append("=" * 70)
    return "\n".join(lines)
