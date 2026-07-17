"""
Synthetic severity sweep — validation of pipeline recovery accuracy.

Generates simulated differentiation hierarchies with known ground-truth
bottleneck locations and continuously-varied severity factors, then runs
the pipeline (distribution fitting → queueing network → bottleneck
diagnostics) on each synthetic dataset and measures how accurately the
true bottleneck is recovered.

Outputs are saved to ``results/synthetic_sweep/``.

Usage::

    python scripts/run_synthetic_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import sys

RESULTS_DIR = Path("results")
SWEEP_DIR = RESULTS_DIR / "synthetic_sweep"


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Sweep parameters
    #
    #    Define which states to bottleneck, which severity factors to
    #    sweep, how many replicates per condition, and how many cells
    #    to simulate per synthetic dataset.
    # ------------------------------------------------------------------
    print("[1/5] Configuring sweep parameters …")

    # Which differentiation stages to artificially bottleneck.
    bottleneck_states = ["HSC", "MPP", "CMP", "LMPP", "MEP", "GMP"]

    # Severity = μ_healthy / μ_bottleneck.  A factor of 1.0 means no
    # bottleneck; larger values mean progressively more severe throttling.
    severity_factors = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    # Number of independent synthetic datasets per (state, severity) cell.
    n_replicates = 5

    # Number of cells in each synthetic dataset.
    n_cells = 5000

    total = len(bottleneck_states) * len(severity_factors) * n_replicates
    print(f"  Sweep size : {total} datasets ({bottleneck_states} × "
          f"{severity_factors} × {n_replicates})")

    # ------------------------------------------------------------------
    # 2. Generate all synthetic datasets
    #
    #    Input:
    #       num_cells          : int  — cells per synthetic dataset.
    #       severity_range     : list[float]
    #       bottleneck_states  : list[str]
    #       n_replicates       : int
    #       seed               : int  — base RNG seed (incremented per replicate).
    #
    #    Returns:
    #       sweep  : dict[(bottleneck_state, severity, replicate), AnnData]
    #
    #    Each AnnData has:
    #       .obs['state']               : str   (true native state)
    #       .obs['residence_time']      : float (gamma-distributed)
    #       .obs['true_traffic_intensity'] : float
    #       .uns['true_hierarchy']      : dict  (service rates, routing)
    #       .uns['true_bottleneck_state'] : str
    #       .uns['true_severity']       : float
    #       .uns['replicate']           : int
    #
    #    TODO: implemented in src/queuediff/synthetic_generator.py
    # ------------------------------------------------------------------
    print("[2/5] Generating synthetic datasets …")
    try:
        from queuediff.synthetic_generator import generate_sweep

        sweep = generate_sweep(
            num_cells=n_cells,
            severity_range=severity_factors,
            bottleneck_states=bottleneck_states,
            n_replicates=n_replicates,
            seed=42,
        )
    except Exception as exc:
        print(f"  FAILED to generate sweep: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Generated {len(sweep)} datasets")

    # ------------------------------------------------------------------
    # 3. For every dataset in the sweep: fit distributions, build queueing
    #    network, and identify the inferred bottleneck.
    #
    #    Per-dataset steps:
    #
    #    a) fit_all_states (distribution_fitting)
    #       Input:  residence_time column from adata.obs
    #       Output: per-state fit results (gamma_shape, exp_rate, aic, …)
    #
    #    b) build_from_data (queueing_network)
    #       Input:  exp_rate per state + routing probabilities
    #       Output: QueueingNetwork + summary DataFrame with traffic_intensity
    #
    #    c) rank_bottlenecks (bottleneck_diagnostics)
    #       Input:  queueing summary
    #       Output: ranked DataFrame; the state with the highest ρ is the
    #               inferred bottleneck.
    #
    #    TODO: implemented in src/queuediff/distribution_fitting.py
    #          implemented in src/queuediff/queueing_network.py
    #          implemented in src/queuediff/bottleneck_diagnostics.py
    # ------------------------------------------------------------------
    print("[3/5] Running pipeline on each synthetic dataset …")

    from queuediff.distribution_fitting import fit_distributions_to_state
    from queuediff.queueing_network import build_from_data
    from queuediff.bottleneck_diagnostics import rank_bottlenecks

    records: list[dict] = []
    processed = 0
    failed = 0

    for (state, sev, rep), adata in sweep.items():
        processed += 1
        if processed % 10 == 0 or processed == total:
            print(f"    {processed}/{total}  …", end="\r")

        try:
            # a) Fit gamma + exponential to each state's residence times
            true_states = adata.obs["state"].unique()
            est_rates: dict[str, float] = {}
            for s in true_states:
                mask = adata.obs["state"] == s
                times = adata.obs.loc[mask, "residence_time"].values
                fit = fit_distributions_to_state(times)
                # Use exponential rate (μ) for the queueing network.
                # The exponential is the Markov baseline; gamma shape will
                # be compared later via AIC.
                est_rates[s] = fit["exp_rate"]

            # b) Build queueing network and compute traffic intensities
            routing = adata.uns["true_routing"]
            qn = build_from_data(est_rates, routing, name=f"synth_{state}_sev{sev}")
            summary = qn.summary()

            # c) Rank states by traffic intensity
            ranked = rank_bottlenecks(summary)
            inferred_bottleneck = ranked.iloc[0]["state"]
            true_bottleneck_rank = (
                ranked[ranked["state"] == state].index[0] + 1
                if state in ranked["state"].values
                else 999
            )

        except Exception as exc:
            failed += 1
            inferred_bottleneck = "ERROR"
            true_bottleneck_rank = -1
            print(f"\n  WARNING: dataset ({state}, sev={sev}, rep={rep}) "
                  f"failed: {exc}")

        records.append({
            "true_bottleneck": state,
            "true_severity": sev,
            "replicate": rep,
            "inferred_bottleneck": inferred_bottleneck,
            "inferred_rank_of_true": true_bottleneck_rank,
        })

    print(f"\n  Pipeline complete: {processed} processed, {failed} failed")

    import pandas as pd

    sweep_results = pd.DataFrame(records)
    sweep_results.to_csv(SWEEP_DIR / "sweep_results.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Recovery-rate aggregation
    #
    #    Compute how often the true bottleneck is ranked in the top-1
    #    (or top-k) positions, stratified by severity factor and by state.
    #
    #    Input:
    #       results  : pd.DataFrame with columns:
    #                    true_bottleneck       : str
    #                    true_severity         : float
    #                    inferred_rank_of_true : int
    #
    #    Returns:
    #       by_severity  : pd.DataFrame  — recovery_rate per severity factor.
    #       by_state     : pd.DataFrame  — recovery_rate per target state.
    #
    #    TODO: implemented in src/queuediff/recovery_validation.py
    # ------------------------------------------------------------------
    print("[4/5] Computing recovery rates …")
    try:
        from queuediff.recovery_validation import (
            recovery_by_severity,
            recovery_by_state,
            recovery_rate,
            confusion_matrix,
        )

        top1 = recovery_rate(sweep_results, k=1)
        top3 = recovery_rate(sweep_results, k=3)
        print(f"  Overall top-1 recovery : {top1:.3f}")
        print(f"  Overall top-3 recovery : {top3:.3f}")

        by_sev = recovery_by_severity(sweep_results, k=1)
        by_state = recovery_by_state(sweep_results)
        conf = confusion_matrix(sweep_results)

    except Exception as exc:
        print(f"  FAILED recovery rate computation: {exc}", file=sys.stderr)
        sys.exit(1)

    by_sev.to_csv(SWEEP_DIR / "recovery_by_severity.csv")
    by_state.to_csv(SWEEP_DIR / "recovery_by_state.csv")
    conf.to_csv(SWEEP_DIR / "confusion_matrix.csv")

    print(f"  Saved to {SWEEP_DIR}/")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    print("[5/5] Summary")
    print()
    print("=" * 60)
    print("  Recovery rate by severity factor")
    print("=" * 60)
    print(f"  {'Severity':>10s}  {'Recovery':>10s}  {'Std':>8s}  {'Trials':>8s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for sev, row in by_sev.iterrows():
        print(f"  {sev:>10.1f}  {row['recovery_rate']:>10.3f}  "
              f"{row['recovery_std']:>8.3f}  {int(row['n_trials']):>8d}")
    print()
    print("=" * 60)
    print("  Recovery rate by bottleneck state")
    print("=" * 60)
    print(f"  {'State':<20s}  {'Recovery':>10s}  {'Std':>8s}  {'Trials':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*8}  {'-'*8}")
    for state, row in by_state.iterrows():
        print(f"  {state:<20s}  {row['recovery_rate']:>10.3f}  "
              f"{row['recovery_std']:>8.3f}  {int(row['n_trials']):>8d}")

    print()
    print("=" * 60)
    print("  Synthetic sweep complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
