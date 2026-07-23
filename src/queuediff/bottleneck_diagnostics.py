"""Bottleneck diagnostics: identify rate-limiting states.

Reports two separate findings:
  1. Traffic intensity ranking (ρ per state)
  2. Gamma vs exponential preference per state

Primary bottleneck: highest ρ AND gamma_preferred = True.
Reports honestly even if no state satisfies both conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bottleneck_ranking(
    traffic_intensities: dict[str, float],
    model_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank states by traffic intensity and annotate with gamma preference.

    Parameters
    ----------
    traffic_intensities : dict[str, float]
        State -> traffic intensity ρ (normalized, dimensionless).
    model_comparison_df : pd.DataFrame
        Output from model_comparison.apply_fdr_correction.
        Must have columns: state, gamma_preferred, delta_aic, fdr_pvalue.

    Returns
    -------
    pd.DataFrame
        Sorted by traffic_intensity descending with columns:
        state, traffic_intensity, gamma_preferred, delta_aic, fdr_pvalue,
        is_primary_bottleneck.
    """
    # Build traffic intensity dataframe
    rho_df = pd.DataFrame([
        {"state": s, "traffic_intensity": rho}
        for s, rho in traffic_intensities.items()
    ])

    # Merge with model comparison
    if "state" in model_comparison_df.columns:
        mc_cols = ["state", "gamma_preferred", "delta_aic", "fdr_pvalue",
                   "gamma_shape", "gamma_mean"]
        available = [c for c in mc_cols if c in model_comparison_df.columns]
        merged = rho_df.merge(model_comparison_df[available], on="state", how="left")
    else:
        merged = rho_df.copy()
        merged["gamma_preferred"] = False
        merged["delta_aic"] = 0.0
        merged["fdr_pvalue"] = 1.0

    # Fill missing gamma_preferred with False
    merged["gamma_preferred"] = merged["gamma_preferred"].fillna(False)

    # Sort by traffic intensity descending
    merged = merged.sort_values("traffic_intensity", ascending=False).reset_index(drop=True)

    # Primary bottleneck: highest ρ AND gamma_preferred
    merged["is_primary_bottleneck"] = False
    gamma_states = merged[merged["gamma_preferred"]]
    if not gamma_states.empty:
        # Highest ρ among gamma-preferred states
        top_idx = gamma_states["traffic_intensity"].idxmax()
        merged.loc[top_idx, "is_primary_bottleneck"] = True

    return merged


def generate_bottleneck_report(
    ranking: pd.DataFrame,
    residence_summary: pd.DataFrame | None = None,
    network_name: str = "",
) -> str:
    """Generate plain-text bottleneck report.

    Parameters
    ----------
    ranking : pd.DataFrame
        From compute_bottleneck_ranking.
    residence_summary : pd.DataFrame, optional
        Residence time summary from clonal analysis.
    network_name : str
        Name for the report header.

    Returns
    -------
    str
        Formatted text suitable for terminal output and .txt file.
    """
    lines = [
        "=" * 70,
        f"BOTTLENECK DIAGNOSTIC REPORT{f': {network_name}' if network_name else ''}",
        "=" * 70,
        "",
    ]

    # Finding 1: Traffic intensity ranking
    lines.append("FINDING 1: TRAFFIC INTENSITY RANKING")
    lines.append("-" * 50)
    lines.append(f"{'Rank':<6} {'State':<8} {'ρ':>10} {'Status':>12}")
    lines.append("-" * 50)

    for rank, (_, row) in enumerate(ranking.iterrows(), 1):
        rho_str = f"{row['traffic_intensity']:.4f}" if np.isfinite(row['traffic_intensity']) else "∞"
        status = "★ PRIMARY" if row.get("is_primary_bottleneck", False) else ""
        lines.append(f"{rank:<6} {row['state']:<8} {rho_str:>10} {status:>12}")

    lines.append("")

    # Finding 2: Gamma vs exponential preference
    lines.append("FINDING 2: GAMMA vs EXPONENTIAL PREFERENCE")
    lines.append("-" * 50)
    lines.append(f"{'State':<8} {'γ pref':>8} {'ΔAIC':>8} {'FDR p':>12} {'γ shape':>8}")
    lines.append("-" * 50)

    for _, row in ranking.iterrows():
        prefer = "YES" if row.get("gamma_preferred", False) else "no"
        delta = f"{row.get('delta_aic', 0):.0f}" if pd.notna(row.get("delta_aic")) else "N/A"
        fdr = f"{row.get('fdr_pvalue', 1):.2e}" if pd.notna(row.get("fdr_pvalue")) else "N/A"
        shape = f"{row.get('gamma_shape', 0):.1f}" if pd.notna(row.get("gamma_shape")) else "N/A"
        lines.append(f"{row['state']:<8} {prefer:>8} {delta:>8} {fdr:>12} {shape:>8}")

    lines.append("")

    # Residence time summary if available
    if residence_summary is not None and not residence_summary.empty:
        lines.append("RESIDENCE TIME ESTIMATES (flux ODE primary)")
        lines.append("-" * 50)
        lines.append(f"{'State':<8} {'Mean (h)':>10} {'Std (h)':>10} {'N':>6}")
        lines.append("-" * 50)

        for _, row in residence_summary.sort_values("mean_hours", ascending=False).iterrows():
            lines.append(
                f"{row['state']:<8} {row['mean_hours']:>10.1f} "
                f"{row['std_hours']:>10.1f} {row['n_observations']:>6}"
            )
        lines.append("")

    # Summary
    primary = ranking[ranking.get("is_primary_bottleneck", False) == True]
    n_gamma = ranking.get("gamma_preferred", pd.Series(dtype=bool)).sum()
    n_total = len(ranking)

    lines.append("SUMMARY")
    lines.append("-" * 50)
    lines.append(f"States analyzed: {n_total}")
    lines.append(f"States with gamma-preferred service times: {n_gamma}/{n_total}")

    if not primary.empty:
        ps = primary.iloc[0]
        lines.append(
            f"Primary bottleneck: {ps['state']} "
            f"(ρ = {ps['traffic_intensity']:.4f}, gamma-preferred)"
        )
    else:
        # Report honestly
        if n_gamma == 0:
            lines.append("Primary bottleneck: NONE (no states have gamma-preferred service times)")
        else:
            highest_rho = ranking.iloc[0]
            lines.append(
                f"Highest ρ: {highest_rho['state']} (ρ = {highest_rho['traffic_intensity']:.4f})"
            )
            if not highest_rho.get("gamma_preferred", False):
                lines.append(
                    f"  Note: {highest_rho['state']} does not have gamma-preferred service times."
                )

    lines.append("=" * 70)
    return "\n".join(lines)
