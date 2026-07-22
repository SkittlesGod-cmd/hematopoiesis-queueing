"""Run synthetic data sweep to validate parameter recovery.

Generates synthetic data with known ground-truth parameters and
validates that the pipeline correctly recovers them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_synthetic_sweep(
    output_dir: str | Path,
    n_samples_list: list[int] | None = None,
    n_repeats: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run parameter recovery sweep across sample sizes.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output files.
    n_samples_list : list[int], optional
        Sample sizes to test. Defaults to [100, 500, 1000, 5000].
    n_repeats : int, default 5
        Number of repeats per sample size.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Recovery results across all conditions.
    """
    from queuediff.synthetic_generator import (
        default_hematopoiesis_params,
        generate_residence_times,
    )
    from queuediff.distribution_fitting import fit_gamma, fit_exponential
    from queuediff.model_comparison import compare_models_per_state, apply_fdr_correction
    from queuediff.recovery_validation import (
        validate_parameter_recovery,
        validate_bottleneck_recovery,
        recovery_summary,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_samples_list is None:
        n_samples_list = [100, 500, 1000, 5000]

    params = default_hematopoiesis_params()
    all_results = []

    for n_samples in n_samples_list:
        for repeat in range(n_repeats):
            rng = np.random.default_rng(42 * (repeat + 1) + n_samples)

            if verbose:
                print(f"  n={n_samples}, repeat={repeat+1}/{n_repeats}")

            # Parameter recovery
            recovery = validate_parameter_recovery(
                params, n_samples=n_samples, rng=rng
            )
            recovery["n_samples"] = n_samples
            recovery["repeat"] = repeat

            all_results.append(recovery)

    combined = pd.concat(all_results, ignore_index=True)

    # Summary statistics
    if verbose:
        print("\nRecovery Summary:")
        print(f"{'N':<8} {'Shape Valid %':>15} {'Mean Valid %':>15}")
        print("-" * 40)
        for n in n_samples_list:
            subset = combined[combined["n_samples"] == n]
            shape_pct = subset["shape_valid"].mean() * 100
            mean_pct = subset["mean_valid"].mean() * 100
            print(f"{n:<8} {shape_pct:>14.1f}% {mean_pct:>14.1f}%")

    # Save
    combined.to_csv(output_dir / "synthetic_sweep_results.csv", index=False)
    if verbose:
        print(f"\nResults saved to: {output_dir / 'synthetic_sweep_results.csv'}")

    return combined


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "results"
    run_synthetic_sweep(output_dir)
