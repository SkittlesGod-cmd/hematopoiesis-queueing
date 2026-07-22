"""Clonal residence time estimation from LARRY lineage barcodes.

Primary method for residence time estimation. Uses clone-traced cell
trajectories across timepoints to measure how long cells spend in
each hematopoietic state.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def extract_clone_trajectories(
    adata,
    state_assignments: pd.Series,
    timepoint_col: str = "Time_point",
    min_clone_size: int = 2,
) -> pd.DataFrame:
    """Extract per-clone state trajectories across timepoints.

    Uses the clone matrix in adata.obsm['clone_matrix'] which was attached
    at load time (survives subsetting automatically, no position-based
    alignment needed).

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['clone_matrix'] (cells x clones binary matrix).
    state_assignments : pd.Series
        State label per cell (index = cell barcodes).
    timepoint_col : str
        Column in adata.obs with timepoint values (days).
    min_clone_size : int, default 2
        Minimum cells in a clone to include it.

    Returns
    -------
    pd.DataFrame
        Columns: clone_id, cell_id, timepoint, state.
        One row per cloned cell per timepoint observation.

    Notes
    -----
    Clone matrix rows align to cells via obsm (position-safe after any filtering).
    Only cells that are both cloned AND have a state assignment are included.
    """
    if "clone_matrix" not in adata.obsm:
        raise ValueError(
            "adata.obsm['clone_matrix'] required. "
            "Load data with include_clones=True."
        )

    clone_matrix = adata.obsm["clone_matrix"]
    if issparse(clone_matrix):
        clone_matrix = clone_matrix.toarray()
    clone_matrix = np.asarray(clone_matrix)

    n_cells, n_clones = clone_matrix.shape
    cell_ids = list(adata.obs_names)
    timepoints = adata.obs[timepoint_col].values

    records = []
    for clone_idx in range(n_clones):
        # Find cells belonging to this clone
        cell_mask = clone_matrix[:, clone_idx] > 0
        clone_cell_indices = np.where(cell_mask)[0]

        if len(clone_cell_indices) < min_clone_size:
            continue

        for cell_pos in clone_cell_indices:
            cell_id = cell_ids[cell_pos]
            if cell_id not in state_assignments.index:
                continue

            records.append({
                "clone_id": clone_idx,
                "cell_id": cell_id,
                "timepoint": float(timepoints[cell_pos]),
                "state": state_assignments[cell_id],
            })

    return pd.DataFrame(records)


def estimate_residence_times_clonal(
    trajectories: pd.DataFrame,
    time_unit_hours: float = 24.0,
) -> dict[str, np.ndarray]:
    """Estimate per-state residence times from clone trajectories.

    For each clone, tracks which states its cells occupy at each timepoint.
    The residence time for a state is the duration a clone's cells remain
    in that state (measured as consecutive timepoint intervals).

    Parameters
    ----------
    trajectories : pd.DataFrame
        Output from extract_clone_trajectories.
        Columns: clone_id, cell_id, timepoint, state.
    time_unit_hours : float, default 24.0
        Hours per timepoint unit (timepoints are in days, so 24.0).

    Returns
    -------
    dict[str, ndarray]
        State -> array of residence times in HOURS.
        Each entry is one clone's measured residence duration in that state.

    Notes
    -----
    Method: For each clone, compute the time span during which cells of that
    clone appear in a given state. This gives one residence time estimate per
    clone per state-occupancy episode.
    """
    if trajectories.empty:
        return {}

    states = sorted(trajectories["state"].unique())
    timepoints = sorted(trajectories["timepoint"].unique())
    residence_times: dict[str, list[float]] = {s: [] for s in states}

    for clone_id, clone_df in trajectories.groupby("clone_id"):
        # For each state, find intervals where this clone has cells in that state
        for state in states:
            state_cells = clone_df[clone_df["state"] == state]
            if state_cells.empty:
                continue

            # Timepoints where this clone has cells in this state
            occupied_tps = sorted(state_cells["timepoint"].unique())

            # Compute residence as consecutive intervals
            episodes = _find_consecutive_episodes(occupied_tps, timepoints)
            for duration_days in episodes:
                duration_hours = duration_days * time_unit_hours
                if duration_hours > 0:
                    residence_times[state].append(duration_hours)

    return {s: np.array(v) for s, v in residence_times.items() if len(v) > 0}


def compute_residence_time_summary(
    residence_times: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Summarize residence time estimates per state.

    Parameters
    ----------
    residence_times : dict[str, ndarray]
        State -> array of residence times in hours.

    Returns
    -------
    pd.DataFrame
        Columns: state, n_observations, mean_hours, std_hours, median_hours,
        min_hours, max_hours.
    """
    records = []
    for state, times in sorted(residence_times.items()):
        records.append({
            "state": state,
            "n_observations": len(times),
            "mean_hours": float(np.mean(times)),
            "std_hours": float(np.std(times)),
            "median_hours": float(np.median(times)),
            "min_hours": float(np.min(times)),
            "max_hours": float(np.max(times)),
        })
    return pd.DataFrame(records)


def compute_normalized_arrival_rates(
    trajectories: pd.DataFrame,
    state_assignments: pd.Series,
    time_unit_hours: float = 24.0,
) -> dict[str, float]:
    """Compute normalized arrival rates per state from clone trajectories.

    λ_normalized = inflow_cells_per_hour / total_cells_in_state

    This normalization makes λ and μ comparable for traffic intensity
    calculation. Without it, ρ = λ/μ has mismatched units and gives
    values in the thousands.

    Parameters
    ----------
    trajectories : pd.DataFrame
        Clone trajectories with columns: clone_id, cell_id, timepoint, state.
    state_assignments : pd.Series
        Full state assignment for all cells (for population counts).
    time_unit_hours : float, default 24.0
        Hours per timepoint unit.

    Returns
    -------
    dict[str, float]
        State -> normalized arrival rate (per hour, dimensionless when
        combined with μ = 1/mean_residence_time).
    """
    if trajectories.empty:
        return {}

    timepoints = sorted(trajectories["timepoint"].unique())
    states = sorted(trajectories["state"].unique())

    # Pre-compute cell sets per (state, timepoint) — single O(n) pass instead of O(states × tp) queries
    grouped = trajectories.groupby(["state", "timepoint"])["cell_id"].apply(set).to_dict()

    arrival_rates = {}
    for state in states:
        total_arrivals = 0
        total_time_hours = 0.0

        for i in range(len(timepoints) - 1):
            tp_prev = timepoints[i]
            tp_next = timepoints[i + 1]
            delta_hours = (tp_next - tp_prev) * time_unit_hours

            cells_prev = grouped.get((state, tp_prev), set())
            cells_next = grouped.get((state, tp_next), set())
            arrivals = len(cells_next - cells_prev)
            total_arrivals += arrivals
            total_time_hours += delta_hours

        # Raw inflow rate
        if total_time_hours > 0:
            raw_rate = total_arrivals / total_time_hours
        else:
            raw_rate = 0.0

        # Normalize by total cells in state
        total_in_state = (state_assignments == state).sum()
        if total_in_state > 0:
            arrival_rates[state] = raw_rate / total_in_state
        else:
            arrival_rates[state] = 0.0

    return arrival_rates


# ── Private helpers ───────────────────────────────────────────────────────────


def _find_consecutive_episodes(
    occupied_tps: list[float],
    all_timepoints: list[float],
) -> list[float]:
    """Find consecutive residence episodes from observed timepoints.

    An episode is a sequence of consecutive timepoints where the clone
    has cells in a state. The duration is first-to-last timepoint in
    the episode (minimum resolution = 1 interval).

    Returns durations in timepoint units (days).
    """
    if not occupied_tps:
        return []

    # Map timepoints to indices
    tp_to_idx = {tp: i for i, tp in enumerate(all_timepoints)}
    occupied_indices = sorted(tp_to_idx[tp] for tp in occupied_tps if tp in tp_to_idx)

    if not occupied_indices:
        return []

    # Find consecutive runs
    episodes = []
    start_idx = occupied_indices[0]
    prev_idx = occupied_indices[0]

    for idx in occupied_indices[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx
        else:
            # End of episode
            duration = all_timepoints[prev_idx] - all_timepoints[start_idx]
            if duration > 0:
                episodes.append(duration)
            start_idx = idx
            prev_idx = idx

    # Last episode -- for a single-timepoint observation, assign minimum interval
    duration = all_timepoints[prev_idx] - all_timepoints[start_idx]
    if duration > 0:
        episodes.append(duration)
    elif len(all_timepoints) > 1:
        # Single timepoint: use minimum interval as lower bound
        min_interval = min(
            all_timepoints[i + 1] - all_timepoints[i]
            for i in range(len(all_timepoints) - 1)
        )
        episodes.append(min_interval)

    return episodes
