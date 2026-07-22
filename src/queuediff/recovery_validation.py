"""Recovery validation: verify that the pipeline recovers known parameters.

Uses synthetic data with known ground truth to validate that the pipeline
correctly estimates service-time distributions and identifies bottlenecks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from queuediff.distribution_fitting import fit_gamma
from queuediff.synthetic_generator import (
    SyntheticParameters,
    compute_true_residence_times,
    generate_residence_times,
)


def validate_parameter_recovery(
    params: SyntheticParameters,
    n_samples: int = 1000,
    shape_tolerance: float = 0.3,
    mean_tolerance: float = 0.2,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Validate that gamma MLE recovers true parameters from synthetic data.

    Parameters
    ----------
    params : SyntheticParameters
        Ground-truth parameters.
    n_samples : int, default 1000
        Number of samples per state for testing.
    shape_tolerance : float, default 0.3
        Maximum relative error in shape recovery.
    mean_tolerance : float, default 0.2
        Maximum relative error in mean recovery.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Columns: state, true_shape, fitted_shape, shape_error,
        true_mean, fitted_mean, mean_error, shape_valid, mean_valid.
    """
    if rng is None:
        rng = np.random.default_rng()

    true_means = compute_true_residence_times(params)
    records = []

    for state in params.states:
        # Generate synthetic residence times
        times = generate_residence_times(params, n_samples, state, rng=rng)

        # Fit gamma
        fit = fit_gamma(times)

        true_shape = params.gamma_shapes[state]
        fitted_shape = fit.params["shape"]
        shape_error = abs(fitted_shape - true_shape) / true_shape

        true_mean = true_means[state]
        fitted_mean = fit.mean
        mean_error = abs(fitted_mean - true_mean) / true_mean

        records.append({
            "state": state,
            "true_shape": true_shape,
            "fitted_shape": fitted_shape,
            "shape_error": shape_error,
            "true_mean": true_mean,
            "fitted_mean": fitted_mean,
            "mean_error": mean_error,
            "shape_valid": shape_error <= shape_tolerance,
            "mean_valid": mean_error <= mean_tolerance,
        })

    return pd.DataFrame(records)


def validate_bottleneck_recovery(
    params: SyntheticParameters,
    detected_bottleneck: str,
    true_bottleneck: str | None = None,
) -> dict[str, any]:
    """Validate that the detected bottleneck matches expectation.

    Parameters
    ----------
    params : SyntheticParameters
        Ground-truth parameters.
    detected_bottleneck : str
        State identified as primary bottleneck.
    true_bottleneck : str, optional
        Known true bottleneck. If None, computed from params.

    Returns
    -------
    dict
        Keys: true_bottleneck, detected_bottleneck, match, all_intensities.
    """
    from queuediff.synthetic_generator import compute_true_traffic_intensity

    intensities = compute_true_traffic_intensity(params)

    if true_bottleneck is None:
        # True bottleneck = highest traffic intensity
        true_bottleneck = max(intensities, key=intensities.get)

    return {
        "true_bottleneck": true_bottleneck,
        "detected_bottleneck": detected_bottleneck,
        "match": detected_bottleneck == true_bottleneck,
        "all_intensities": intensities,
    }


def recovery_summary(
    param_recovery: pd.DataFrame,
    bottleneck_recovery: dict,
) -> str:
    """Generate text summary of recovery validation.

    Parameters
    ----------
    param_recovery : pd.DataFrame
        From validate_parameter_recovery.
    bottleneck_recovery : dict
        From validate_bottleneck_recovery.

    Returns
    -------
    str
        Formatted text report.
    """
    lines = [
        "=" * 60,
        "RECOVERY VALIDATION REPORT",
        "=" * 60,
        "",
        "Parameter Recovery (gamma shape and mean):",
        f"{'State':<8} {'True k':>8} {'Fit k':>8} {'Error':>8} {'Valid':>6}",
        "-" * 50,
    ]

    for _, row in param_recovery.iterrows():
        valid = "✓" if row["shape_valid"] else "✗"
        lines.append(
            f"{row['state']:<8} {row['true_shape']:>8.1f} "
            f"{row['fitted_shape']:>8.1f} {row['shape_error']:>8.3f} {valid:>6}"
        )

    lines.append("")
    lines.append(f"Shape recovery: {param_recovery['shape_valid'].sum()}/{len(param_recovery)} states within tolerance")
    lines.append(f"Mean recovery: {param_recovery['mean_valid'].sum()}/{len(param_recovery)} states within tolerance")
    lines.append("")

    match = "MATCH" if bottleneck_recovery["match"] else "MISMATCH"
    lines.append(f"Bottleneck detection: {match}")
    lines.append(f"  True bottleneck:     {bottleneck_recovery['true_bottleneck']}")
    lines.append(f"  Detected bottleneck: {bottleneck_recovery['detected_bottleneck']}")
    lines.append("=" * 60)

    return "\n".join(lines)
