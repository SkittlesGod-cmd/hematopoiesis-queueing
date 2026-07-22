"""Branch point validation for the queueing network topology.

Validates that the inferred routing probabilities at branch points
(where one state feeds into multiple downstream states) are
biologically consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_routing_probabilities(
    trajectories: pd.DataFrame,
    routing_structure: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Estimate routing probabilities from clone trajectories.

    For each branch point (state with multiple downstream states),
    estimates the fraction of individual cells that route to each
    downstream state between consecutive timepoints.

    Parameters
    ----------
    trajectories : pd.DataFrame
        Clone trajectories with columns: clone_id, cell_id, timepoint, state.
    routing_structure : dict[str, list[str]]
        Expected routing: source -> list of possible target states.

    Returns
    -------
    dict[str, dict[str, float]]
        source_state -> {target_state: probability}.
        Probabilities sum to 1.0 for each source.
    """
    if trajectories.empty:
        return {}

    timepoints = sorted(trajectories["timepoint"].unique())
    result = {}

    for source, targets in routing_structure.items():
        if len(targets) < 2:
            # Not a branch point (single or no downstream)
            if targets:
                result[source] = {targets[0]: 1.0}
            continue

        # Count individual cell transitions from source to each target
        # Per-cell tracking avoids overcounting when multiple cells
        # from the same clone are at different states
        transition_counts = {t: 0 for t in targets}
        total_transitions = 0

        for cell_id, cell_df in trajectories.groupby("cell_id"):
            cell_df = cell_df.sort_values("timepoint")
            for i in range(len(cell_df) - 1):
                curr = cell_df.iloc[i]
                nxt = cell_df.iloc[i + 1]
                if curr["state"] == source and nxt["state"] in targets:
                    transition_counts[nxt["state"]] += 1
                    total_transitions += 1

        # Normalize to probabilities
        if total_transitions > 0:
            result[source] = {t: c / total_transitions for t, c in transition_counts.items()}
        else:
            # Equal split as fallback (no observed transitions)
            n = len(targets)
            result[source] = {t: 1.0 / n for t in targets}

    return result


def validate_branch_points(
    estimated_probs: dict[str, dict[str, float]],
    expected_probs: dict[str, dict[str, float]] | None = None,
    tolerance: float = 0.2,
) -> pd.DataFrame:
    """Validate estimated routing probabilities against expectations.

    Parameters
    ----------
    estimated_probs : dict
        From estimate_routing_probabilities.
    expected_probs : dict, optional
        Expected probabilities for comparison. If None, only reports estimates.
    tolerance : float, default 0.2
        Maximum acceptable deviation from expected probabilities.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, estimated_prob, expected_prob, deviation, valid.
    """
    records = []

    for source, targets in estimated_probs.items():
        for target, est_prob in targets.items():
            exp_prob = None
            deviation = None
            valid = True

            if expected_probs and source in expected_probs:
                exp_prob = expected_probs[source].get(target)
                if exp_prob is not None:
                    deviation = abs(est_prob - exp_prob)
                    valid = deviation <= tolerance

            records.append({
                "source": source,
                "target": target,
                "estimated_prob": est_prob,
                "expected_prob": exp_prob,
                "deviation": deviation,
                "valid": valid,
            })

    return pd.DataFrame(records)
