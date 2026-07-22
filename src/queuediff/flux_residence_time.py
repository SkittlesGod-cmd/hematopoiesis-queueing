"""Flux-based residence time estimation via ODE fitting.

Secondary method for residence time estimation. Fits an ODE model to
population-level state occupancy dynamics over time to estimate
transition rates and derive residence times.

Known limitations (document, do not try to fix):
1. Source state (HSC): no upstream inflow → transition rate unidentifiable
2. Fast-draining terminal states with only 3 timepoints: exit rate driven
   to lower bound → degenerate residence time
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


def compute_state_occupancy(
    adata,
    state_assignments: pd.Series,
    timepoint_col: str = "Time_point",
) -> pd.DataFrame:
    """Compute fraction of cells in each state at each timepoint.

    Parameters
    ----------
    adata : AnnData
        Must have timepoint column in obs.
    state_assignments : pd.Series
        State per cell.
    timepoint_col : str
        Column in adata.obs with timepoint values (in days).

    Returns
    -------
    pd.DataFrame
        Index = timepoints (days), columns = states, values = cell fractions.
    """
    df = pd.DataFrame({
        "timepoint": adata.obs[timepoint_col].values,
        "state": state_assignments.values,
    })

    # Count cells per state per timepoint
    counts = df.groupby(["timepoint", "state"]).size().unstack(fill_value=0)

    # Normalize to fractions
    fractions = counts.div(counts.sum(axis=1), axis=0)

    return fractions


def fit_transition_rates(
    occupancy: pd.DataFrame,
    routing_structure: dict[str, list[str]],
    rate_bounds: tuple[float, float] = (1e-4, 1.0),
) -> pd.DataFrame:
    """Fit transition rates by minimizing ODE residuals against observed occupancy.

    Parameters
    ----------
    occupancy : pd.DataFrame
        State occupancy fractions over time (from compute_state_occupancy).
        Index = timepoints in DAYS, columns = state names.
    routing_structure : dict[str, list[str]]
        For each state, list of downstream states it feeds into.
        Terminal states have empty lists.
    rate_bounds : tuple[float, float]
        (lower, upper) bounds for transition rates (per hour).

    Returns
    -------
    pd.DataFrame
        Columns: state, exit_rate_per_hour, residence_time_hours, is_degenerate.

    Notes
    -----
    Timepoints are in DAYS. Rates are in per HOUR.
    Solves the linear ODE via matrix exponential (scipy.linalg.expm),
    which gives the exact solution y(t) = expm(A τ) @ y0 without the
    numerical integration error of RK45-based methods.
    Timepoints are converted from days to hours (×24) before solving.
    Failure to do this causes 24x unit mismatch -- all ODE fits degenerate.
    """
    states = list(occupancy.columns)
    timepoints_days = np.array(occupancy.index, dtype=np.float64)
    # Convert to HOURS for ODE integration (critical: prevents 24x unit mismatch)
    timepoints_hours = timepoints_days * 24.0
    observed = occupancy.values  # shape: (n_timepoints, n_states)

    # Initial conditions (fractions at first timepoint)
    y0 = observed[0]

    # Optimize exit rates for all non-source states
    n_states = len(states)
    x0 = np.full(n_states, 0.01)  # initial guess: 0.01/hr

    def objective(rates):
        """Sum of squared residuals between ODE solution (via expm) and observed fractions."""
        rates = np.clip(rates, rate_bounds[0], rate_bounds[1])
        predicted = _solve_ode(y0, rates, states, routing_structure, timepoints_hours)
        if predicted is None:
            return 1e10
        return float(np.sum((predicted - observed) ** 2))

    bounds = [(rate_bounds[0], rate_bounds[1])] * n_states
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    fitted_rates = np.clip(result.x, rate_bounds[0], rate_bounds[1])

    # Build results
    records = []
    for i, state in enumerate(states):
        rate = fitted_rates[i]
        residence_time = 1.0 / rate if rate > 0 else np.inf

        # Identify degenerate states (at or near bounds)
        is_degenerate = (
            abs(rate - rate_bounds[0]) < 1e-6 or
            abs(rate - rate_bounds[1]) < 1e-6
        )

        records.append({
            "state": state,
            "exit_rate_per_hour": float(rate),
            "residence_time_hours": float(residence_time),
            "is_degenerate": is_degenerate,
        })

    return pd.DataFrame(records)


def identify_degenerate_states(
    fit_results: pd.DataFrame,
) -> list[str]:
    """Identify states where fitted rate is at or near the lower bound.

    These states have unreliable residence time estimates. Common causes:
    - Source state (HSC): no upstream inflow modeled
    - Fast-draining terminal states: only 3 timepoints insufficient

    Parameters
    ----------
    fit_results : pd.DataFrame
        Output from fit_transition_rates.

    Returns
    -------
    list[str]
        Names of degenerate states.
    """
    return list(fit_results[fit_results["is_degenerate"]]["state"])


def flux_residence_time_summary(
    clonal_summary: pd.DataFrame,
    flux_results: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-reference clonal and flux residence time estimates.

    Parameters
    ----------
    clonal_summary : pd.DataFrame
        From clonal_residence_time.compute_residence_time_summary.
    flux_results : pd.DataFrame
        From fit_transition_rates.

    Returns
    -------
    pd.DataFrame
        Merged summary with both estimates per state and their ratio.
    """
    merged = clonal_summary[["state", "mean_hours"]].merge(
        flux_results[["state", "residence_time_hours", "is_degenerate"]],
        on="state",
        how="outer",
    )
    merged = merged.rename(columns={
        "mean_hours": "clonal_mean_hours",
        "residence_time_hours": "flux_mean_hours",
    })

    # Compute ratio (clonal / flux) for concordance
    merged["ratio_clonal_flux"] = merged["clonal_mean_hours"] / merged["flux_mean_hours"]

    return merged


# ── Private helpers ───────────────────────────────────────────────────────────


def _solve_ode(
    y0: np.ndarray,
    rates: np.ndarray,
    states: list[str],
    routing_structure: dict[str, list[str]],
    timepoints_hours: np.ndarray,
) -> np.ndarray | None:
    """Solve the state-occupancy ODE system via matrix exponential.

    dy/dt = A @ y  where A is the linear system matrix.

    For a linear time-invariant ODE, the matrix exponential gives the
    exact solution: y(t) = expm(A * (t - t0)) @ y0, avoiding numerical
    integration error from methods like RK45.

    Parameters
    ----------
    y0 : ndarray
        Initial state fractions.
    rates : ndarray
        Exit rates per state (per hour).
    states : list[str]
        State names.
    routing_structure : dict
        Downstream connections per state.
    timepoints_hours : ndarray
        Evaluation times in HOURS.

    Returns
    -------
    ndarray or None
        Predicted fractions at each timepoint (n_timepoints x n_states).
        None if computation fails.
    """
    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    # Build routing matrix (equal split among downstream states)
    routing = np.zeros((n_states, n_states))
    for src, targets in routing_structure.items():
        if src not in state_idx:
            continue
        if targets:
            prob = 1.0 / len(targets)
            for tgt in targets:
                if tgt in state_idx:
                    routing[state_idx[tgt], state_idx[src]] = prob

    # Build the system matrix A: dy_i/dt = -rates[i] * y_i + sum_j(routing[i,j] * rates[j] * y_j)
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                A[i, j] = routing[i, j] * rates[j]  # inflow from j to i
        A[i, i] = -rates[i]  # outflow from i

    # Exact solution via matrix exponential: y(t) = expm(A * τ) @ y0
    try:
        t0 = timepoints_hours[0]
        predicted = np.zeros((len(timepoints_hours), n_states))
        for k, t in enumerate(timepoints_hours):
            tau = t - t0
            predicted[k] = expm(A * tau) @ y0
        return predicted
    except Exception:
        return None
