"""Run the full queuediff pipeline on Weinreb et al. 2020 data.

End-to-end: load -> preprocess -> discretize -> estimate residence times
-> fit distributions -> build queueing network -> detect bottleneck.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    verbose: bool = True,
) -> dict:
    """Run the complete analysis pipeline.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing Weinreb data files.
    output_dir : str or Path
        Directory for output files (tables, reports).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Pipeline results including residence times, model comparison,
        bottleneck ranking, and diagnostics.
    """
    from queuediff.data_loading import load_weinreb, preprocess_standard
    from queuediff.state_discretization import (
        assign_states,
        calibrate_division_death_rates,
        score_apoptosis,
        score_cell_cycle,
        score_marker_states,
    )
    from queuediff.clonal_residence_time import (
        compute_normalized_arrival_rates,
        compute_residence_time_summary,
        estimate_residence_times_clonal,
        extract_clone_trajectories,
    )
    from queuediff.flux_residence_time import (
        compute_state_occupancy,
        fit_transition_rates,
        flux_residence_time_summary,
        identify_degenerate_states,
    )
    from queuediff.distribution_fitting import fit_gamma, fit_exponential
    from queuediff.model_comparison import (
        apply_fdr_correction,
        compare_models_per_state,
        summarize_model_comparison,
    )
    from queuediff.queueing_network import build_from_data
    from queuediff.bottleneck_diagnostics import (
        compute_bottleneck_ranking,
        generate_bottleneck_report,
    )
    from queuediff.branch_point_validation import (
        estimate_routing_probabilities,
        validate_branch_points,
    )

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Step 1: Load data ─────────────────────────────────────────────────
    if verbose:
        print("Step 1: Loading Weinreb data...")
    adata = load_weinreb(data_dir, include_clones=True)
    if verbose:
        print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"  Timepoints: {sorted(adata.obs['Time_point'].unique())}")

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    if verbose:
        print("Step 2: Preprocessing...")
    # already_normalized=True: Weinreb data is pre-normalized, never call normalize_total
    adata = preprocess_standard(adata, already_normalized=True)
    if verbose:
        print(f"  After preprocessing: {adata.shape[0]} cells × {adata.shape[1]} HVGs")
        print(f"  lognorm_full shape: {adata.obsm['lognorm_full'].shape}")

    # ── Step 3: State discretization ──────────────────────────────────────
    if verbose:
        print("Step 3: State discretization...")
    scores = score_marker_states(adata)
    state_assignments = assign_states(scores)
    if verbose:
        print("  State distribution:")
        for state, count in state_assignments.value_counts().sort_index().items():
            print(f"    {state}: {count} ({count/len(state_assignments)*100:.1f}%)")

    results["state_assignments"] = state_assignments
    results["state_scores"] = scores

    # ── Step 4: Cell cycle and apoptosis scoring ──────────────────────────
    if verbose:
        print("Step 4: Cell cycle and apoptosis scoring...")
    # MUST use lognorm_full (full gene set). Using lognorm silently gives zero.
    cc_scores = score_cell_cycle(adata)
    apop_scores = score_apoptosis(adata)
    if verbose:
        print(f"  Mean cycling score: {cc_scores['cycling_score'].mean():.4f}")
        print(f"  Mean net apoptotic: {apop_scores['net_apoptotic_score'].mean():.4f}")

    results["cell_cycle_scores"] = cc_scores
    results["apoptosis_scores"] = apop_scores

    # ── Step 5: Division/death rate calibration ───────────────────────────
    if verbose:
        print("Step 5: Division/death rate calibration...")
    # Population-dynamics calibration (NOT fraction-above-threshold)
    rates = calibrate_division_death_rates(
        adata, state_assignments,
        cc_scores["cycling_score"],
        apop_scores["net_apoptotic_score"],
    )
    if verbose:
        print("  Rates per state:")
        for _, row in rates.iterrows():
            shrink = " [shrinking]" if row["net_shrinking"] else ""
            print(f"    {row['state']}: div={row['division_rate']:.6f}/h, "
                  f"death={row['death_rate']:.6f}/h{shrink}")

    results["division_death_rates"] = rates

    # ── Step 6: Clone trajectory extraction ───────────────────────────────
    if verbose:
        print("Step 6: Extracting clone trajectories...")
    trajectories = extract_clone_trajectories(adata, state_assignments)
    if verbose:
        n_clones = trajectories["clone_id"].nunique()
        print(f"  Clones with trajectories: {n_clones}")
        print(f"  Total observations: {len(trajectories)}")

    results["trajectories"] = trajectories

    # ── Step 7: Clonal residence time estimation ──────────────────────────
    if verbose:
        print("Step 7: Estimating clonal residence times...")
    # Timepoints are in days; time_unit_hours=24 converts to hours
    residence_times = estimate_residence_times_clonal(trajectories, time_unit_hours=24.0)
    residence_summary = compute_residence_time_summary(residence_times)
    if verbose:
        print("  Residence times (mean hours):")
        for _, row in residence_summary.sort_values("mean_hours", ascending=False).iterrows():
            print(f"    {row['state']}: {row['mean_hours']:.1f}h "
                  f"(n={row['n_observations']})")

    results["residence_times"] = residence_times
    results["residence_summary"] = residence_summary

    # ── Step 8: Normalized arrival rates ──────────────────────────────────
    if verbose:
        print("Step 8: Computing normalized arrival rates...")
    arrival_rates = compute_normalized_arrival_rates(
        trajectories, state_assignments, time_unit_hours=24.0
    )
    if verbose:
        for state, rate in sorted(arrival_rates.items()):
            print(f"    {state}: λ_norm = {rate:.6f}/h")

    results["arrival_rates"] = arrival_rates

    # ── Step 9: Distribution fitting and model comparison ─────────────────
    if verbose:
        print("Step 9: Fitting distributions...")
    comparison = compare_models_per_state(residence_times, min_samples=10)
    comparison = apply_fdr_correction(comparison)
    mc_summary = summarize_model_comparison(comparison)
    if verbose:
        print(mc_summary)

    results["model_comparison"] = comparison
    results["model_comparison_text"] = mc_summary

    # ── Routing structure (shared by flux ODE and branch point validation) ──
    routing_structure = {
        "HSC": ["MPP"],
        "MPP": ["CMP", "LMPP"],
        "CMP": ["MEP", "GMP"],
        "LMPP": [],
        "MEP": [],
        "GMP": [],
    }

    # ── Step 10: Branch point validation (before flux ODE) ────────────────
    if verbose:
        print("Step 10: Branch point validation...")
    est_probs = estimate_routing_probabilities(trajectories, routing_structure)
    branch_validation = validate_branch_points(est_probs)
    if verbose and not branch_validation.empty:
        for _, row in branch_validation.iterrows():
            print(f"    {row['source']} -> {row['target']}: "
                  f"p = {row['estimated_prob']:.3f}")

    results["routing_probabilities"] = est_probs
    results["branch_validation"] = branch_validation

    # ── Step 11: Flux-based estimation (primary, uses routing probabilities) ─
    if verbose:
        print("Step 11: Flux-based residence time estimation (primary)...")
    occupancy = compute_state_occupancy(adata, state_assignments)
    # Pass estimated routing probabilities into ODE solver for accurate system matrix
    flux_results = fit_transition_rates(
        occupancy, routing_structure,
        routing_probs=est_probs if est_probs else None,
    )
    degenerate = identify_degenerate_states(flux_results)
    if verbose:
        print("  Flux residence times:")
        for _, row in flux_results.iterrows():
            flag = " [DEGENERATE]" if row["is_degenerate"] else ""
            print(f"    {row['state']}: {row['residence_time_hours']:.1f}h{flag}")
        if degenerate:
            print(f"  Degenerate states: {degenerate}")

    flux_summary = flux_residence_time_summary(residence_summary, flux_results)
    results["flux_results"] = flux_results
    results["flux_summary"] = flux_summary

    # ── Step 12: Build queueing network ───────────────────────────────────
    if verbose:
        print("Step 12: Building queueing network...")

    # Service rates: flux ODE primary, clonal fallback for degenerate states
    service_rates = {}
    flux_degenerate_states = set(degenerate)
    for _, row in flux_results.iterrows():
        state = row["state"]
        if row["is_degenerate"]:
            # Fallback to clonal residence time for degenerate states
            clonal_row = residence_summary[residence_summary["state"] == state]
            if not clonal_row.empty:
                service_rates[state] = 1.0 / clonal_row.iloc[0]["mean_hours"]
            else:
                service_rates[state] = row["exit_rate_per_hour"]
        else:
            service_rates[state] = row["exit_rate_per_hour"]

    if verbose and flux_degenerate_states:
        print(f"  Using clonal fallback for degenerate states: {flux_degenerate_states}")

    # Routing probabilities from branch point estimation (dict format)
    routing_dict = {}
    for src, targets in est_probs.items():
        if isinstance(targets, dict):
            routing_dict[src] = targets
        elif isinstance(targets, list):
            if targets:
                routing_dict[src] = {t: 1.0 / len(targets) for t in targets}

    network = build_from_data(service_rates, routing_dict, name="Weinreb Hematopoiesis")

    # Compute traffic intensity via network (handles servers parameter c)
    traffic_intensities = network.traffic_intensity(arrival_rates)

    if verbose:
        print("  Traffic intensities:")
        for state in sorted(traffic_intensities, key=traffic_intensities.get, reverse=True):
            rho = traffic_intensities[state]
            print(f"    {state}: ρ = {rho:.6f}")

    results["network"] = network
    results["traffic_intensities"] = traffic_intensities

    # ── Step 13: Bottleneck diagnostics ───────────────────────────────────
    if verbose:
        print("Step 13: Bottleneck diagnostics...")
    ranking = compute_bottleneck_ranking(traffic_intensities, comparison)
    # Use flux-based residence summary for display (primary method)
    flux_display_summary = flux_results[["state", "residence_time_hours"]].copy()
    flux_display_summary = flux_display_summary.rename(columns={"residence_time_hours": "mean_hours"})
    flux_display_summary["std_hours"] = 0.0
    flux_display_summary["n_observations"] = 0
    flux_display_summary["method"] = "flux_ode"
    report = generate_bottleneck_report(ranking, flux_display_summary, "Weinreb Hematopoiesis")
    if verbose:
        print(report)

    results["bottleneck_ranking"] = ranking
    results["bottleneck_report"] = report

    # ── Save outputs ──────────────────────────────────────────────────────
    if verbose:
        print("\nSaving results...")

    residence_summary.to_csv(output_dir / "residence_times.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    ranking.to_csv(output_dir / "bottleneck_ranking.csv", index=False)
    flux_summary.to_csv(output_dir / "flux_comparison.csv", index=False)
    rates.to_csv(output_dir / "division_death_rates.csv", index=False)

    # Save state assignments (needed by generate_figures.py and nestorowa cross-check)
    state_assignments.to_frame("state").to_csv(output_dir / "state_assignments.csv")

    # Save residence times as JSON (dict of arrays, needed by generate_figures.py fig2)
    import json
    residence_times_json = {k: v.tolist() for k, v in residence_times.items()}
    with open(output_dir / "residence_times.json", "w") as f:
        json.dump(residence_times_json, f)

    # Save routing probabilities as JSON (needed by generate_figures.py fig5)
    with open(output_dir / "routing_probabilities.json", "w") as f:
        json.dump(est_probs, f)

    with open(output_dir / "bottleneck_report.txt", "w") as f:
        f.write(report)
    with open(output_dir / "model_comparison_report.txt", "w") as f:
        f.write(mc_summary)

    if verbose:
        print(f"  Results saved to: {output_dir}")
        print("\nPipeline complete.")

    return results


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "raw" / "weinreb"
    output_dir = script_dir.parent / "results"

    if not data_dir.exists():
        print(f"Data not found at {data_dir}")
        print("Run 'python scripts/download_weinreb.py' first.")
        sys.exit(1)

    run_pipeline(data_dir, output_dir)
