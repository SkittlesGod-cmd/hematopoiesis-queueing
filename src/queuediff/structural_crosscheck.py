"""Structural cross-check using Nestorowa et al. 2016 data.

Single-timepoint data used ONLY for structural validation, NOT for
quantitative bottleneck analysis. Confirms that the state discretization
and marker panels produce biologically consistent assignments on an
independent dataset.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd

from queuediff.state_discretization import MARKER_PANELS, score_marker_states, assign_states


def crosscheck_state_structure(
    primary_assignments: pd.Series,
    secondary_adata: ad.AnnData,
    primary_name: str = "Weinreb",
    secondary_name: str = "Nestorowa",
    panels: dict[str, list[str]] | None = None,
) -> dict[str, any]:
    """Cross-check state structure between primary and secondary datasets.

    Parameters
    ----------
    primary_assignments : pd.Series
        State assignments from the primary dataset.
    secondary_adata : AnnData
        Secondary dataset (e.g., Nestorowa). Must have layers['lognorm'].
    primary_name : str
        Name of the primary dataset for reporting.
    secondary_name : str
        Name of the secondary dataset for reporting.
    panels : dict, optional
        Marker panels to use. Defaults to MARKER_PANELS.

    Returns
    -------
    dict
        Keys: primary_distribution, secondary_distribution,
        shared_states, primary_only, secondary_only,
        structural_concordance (fraction of states shared).

    Notes
    -----
    This is a structural comparison only. The secondary dataset is
    single-timepoint and CANNOT be used for quantitative analysis.
    """
    if panels is None:
        panels = MARKER_PANELS

    # Primary state distribution
    primary_dist = primary_assignments.value_counts(normalize=True).to_dict()
    primary_states = set(primary_assignments.unique())

    # Score secondary dataset
    try:
        secondary_scores = score_marker_states(secondary_adata, panels=panels)
        secondary_assignments = assign_states(secondary_scores)
        secondary_dist = secondary_assignments.value_counts(normalize=True).to_dict()
        secondary_states = set(secondary_assignments.unique())
    except Exception as e:
        warnings.warn(
            f"Could not score secondary dataset ({secondary_name}): {e}",
            stacklevel=2,
        )
        secondary_dist = {}
        secondary_states = set()

    shared = primary_states & secondary_states
    concordance = len(shared) / max(len(primary_states | secondary_states), 1)

    return {
        "primary_name": primary_name,
        "secondary_name": secondary_name,
        "primary_distribution": primary_dist,
        "secondary_distribution": secondary_dist,
        "shared_states": sorted(shared),
        "primary_only": sorted(primary_states - secondary_states),
        "secondary_only": sorted(secondary_states - primary_states),
        "structural_concordance": concordance,
    }


def format_crosscheck_report(result: dict) -> str:
    """Format structural cross-check result as text.

    Parameters
    ----------
    result : dict
        Output from crosscheck_state_structure.

    Returns
    -------
    str
        Formatted report.
    """
    lines = [
        "=" * 60,
        "STRUCTURAL CROSS-CHECK",
        f"Primary: {result['primary_name']}",
        f"Secondary: {result['secondary_name']}",
        "=" * 60,
        "",
        f"Shared states: {result['shared_states']}",
        f"Primary-only states: {result['primary_only']}",
        f"Secondary-only states: {result['secondary_only']}",
        f"Structural concordance: {result['structural_concordance']:.2f}",
        "",
    ]

    if result["primary_distribution"]:
        lines.append(f"{result['primary_name']} state distribution:")
        for state, frac in sorted(result["primary_distribution"].items()):
            lines.append(f"  {state}: {frac:.3f}")
        lines.append("")

    if result["secondary_distribution"]:
        lines.append(f"{result['secondary_name']} state distribution:")
        for state, frac in sorted(result["secondary_distribution"].items()):
            lines.append(f"  {state}: {frac:.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)
