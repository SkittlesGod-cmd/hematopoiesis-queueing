"""Synthetic data generator for validation of the queueing network pipeline.

Generates synthetic clone trajectory data with known ground-truth parameters
(gamma-distributed service times, known routing probabilities) to validate
that the pipeline recovers true parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist


@dataclass
class SyntheticParameters:
    """Ground-truth parameters for synthetic data generation.

    Attributes
    ----------
    states : list[str]
        Ordered state names.
    gamma_shapes : dict[str, float]
        Gamma shape parameter (k) per state. k=1 is exponential.
    gamma_scales : dict[str, float]
        Gamma scale parameter (theta) per state. Mean = k * theta (hours).
    routing_probs : dict[str, dict[str, float]]
        Transition probabilities. routing_probs[src][dst] = probability.
    source_state : str
        Entry point for cells into the network.
    n_clones : int
        Number of clones to simulate.
    cells_per_clone : int
        Number of cells per clone.
    observation_times : list[float]
        Timepoints at which cells are observed (hours).
    """

    states: list[str]
    gamma_shapes: dict[str, float]
    gamma_scales: dict[str, float]
    routing_probs: dict[str, dict[str, float]]
    source_state: str
    n_clones: int = 50
    cells_per_clone: int = 20
    observation_times: list[float] = field(default_factory=lambda: [48.0, 96.0, 144.0])


def default_hematopoiesis_params() -> SyntheticParameters:
    """Default synthetic parameters mimicking hematopoietic differentiation.

    Uses biologically-plausible values inspired by real pipeline outputs.
    Gamma shapes > 1 for all states (semi-Markov claim).
    """
    states = ["HSC", "MPP", "CMP", "LMPP", "MEP", "GMP"]
    return SyntheticParameters(
        states=states,
        gamma_shapes={
            "HSC": 15.0, "MPP": 12.0, "CMP": 10.0,
            "LMPP": 8.0, "MEP": 18.0, "GMP": 20.0,
        },
        gamma_scales={
            "HSC": 1.1, "MPP": 1.1, "CMP": 1.05,
            "LMPP": 1.05, "MEP": 1.02, "GMP": 0.97,
        },
        routing_probs={
            "HSC": {"MPP": 1.0},
            "MPP": {"CMP": 0.5, "LMPP": 0.5},
            "CMP": {"MEP": 0.6, "GMP": 0.4},
            "LMPP": {},  # terminal
            "MEP": {},   # terminal
            "GMP": {},   # terminal
        },
        source_state="HSC",
        n_clones=50,
        cells_per_clone=20,
        observation_times=[48.0, 96.0, 144.0],
    )


def generate_residence_times(
    params: SyntheticParameters,
    n_samples: int,
    state: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate gamma-distributed residence times for a single state.

    Parameters
    ----------
    params : SyntheticParameters
        Parameters defining the gamma distribution.
    n_samples : int
        Number of residence time samples to generate.
    state : str
        Which state to generate for.
    rng : Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    ndarray
        Array of residence times in hours.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = params.gamma_shapes[state]
    theta = params.gamma_scales[state]

    return rng.gamma(shape=k, scale=theta, size=n_samples)


def simulate_clone_trajectories(
    params: SyntheticParameters,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate clone-traced cell trajectories through the queueing network.

    Each cell enters at source_state, spends a gamma-distributed time there,
    then routes to the next state according to routing_probs, until reaching
    a terminal state (no outgoing transitions).

    Cells are observed at each observation_time. The observed state is whichever
    state the cell is in at that timepoint.

    Parameters
    ----------
    params : SyntheticParameters
        Ground-truth parameters for simulation.
    rng : Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: clone_id, cell_id, timepoint, state, entry_time, exit_time.
        One row per cell per observation timepoint where the cell is observed.
    """
    if rng is None:
        rng = np.random.default_rng()

    records = []

    for clone_idx in range(params.n_clones):
        for cell_idx in range(params.cells_per_clone):
            cell_id = f"clone{clone_idx}_cell{cell_idx}"

            # Simulate trajectory: sequence of (state, entry_time, exit_time)
            trajectory = _simulate_single_cell(params, rng)

            # Observe at each timepoint
            for tp in params.observation_times:
                state_at_tp = _state_at_time(trajectory, tp)
                if state_at_tp is not None:
                    records.append({
                        "clone_id": clone_idx,
                        "cell_id": cell_id,
                        "timepoint": tp,
                        "state": state_at_tp,
                    })

    return pd.DataFrame(records)


def compute_true_residence_times(
    params: SyntheticParameters,
) -> dict[str, float]:
    """Compute true mean residence times from parameters.

    Mean of gamma(k, theta) = k * theta.

    Parameters
    ----------
    params : SyntheticParameters
        Ground-truth parameters.

    Returns
    -------
    dict
        state -> mean residence time in hours.
    """
    return {
        state: params.gamma_shapes[state] * params.gamma_scales[state]
        for state in params.states
    }


def compute_true_traffic_intensity(
    params: SyntheticParameters,
    external_arrival_rate: float = 1.0,
) -> dict[str, float]:
    """Compute true traffic intensity from parameters.

    ρ = λ / μ where μ = 1 / mean_residence_time.

    Parameters
    ----------
    params : SyntheticParameters
        Ground-truth parameters.
    external_arrival_rate : float
        Arrival rate into the source state.

    Returns
    -------
    dict
        state -> traffic intensity.
    """
    # Compute arrival rates via topological propagation
    arrival_rates = {s: 0.0 for s in params.states}
    arrival_rates[params.source_state] = external_arrival_rate

    # Simple topological propagation (assumes DAG)
    visited = set()
    queue = [params.source_state]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in params.routing_probs:
            for target, prob in params.routing_probs[current].items():
                arrival_rates[target] += arrival_rates[current] * prob
                if target not in visited:
                    queue.append(target)

    # Traffic intensity: ρ = λ * mean_residence_time (since μ = 1/mean_time)
    mean_times = compute_true_residence_times(params)
    return {
        state: arrival_rates[state] * mean_times[state]
        for state in params.states
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _simulate_single_cell(
    params: SyntheticParameters,
    rng: np.random.Generator,
) -> list[tuple[str, float, float]]:
    """Simulate one cell's trajectory through the network.

    Returns list of (state, entry_time, exit_time) tuples.
    """
    trajectory = []
    current_state = params.source_state
    current_time = 0.0

    max_steps = 100  # prevent infinite loops
    for _ in range(max_steps):
        # Sample residence time
        k = params.gamma_shapes[current_state]
        theta = params.gamma_scales[current_state]
        residence = float(rng.gamma(shape=k, scale=theta))

        entry = current_time
        exit_time = current_time + residence
        trajectory.append((current_state, entry, exit_time))

        current_time = exit_time

        # Route to next state
        routes = params.routing_probs.get(current_state, {})
        if not routes:
            break  # terminal state

        targets = list(routes.keys())
        probs = list(routes.values())
        current_state = rng.choice(targets, p=probs)

    return trajectory


def _state_at_time(
    trajectory: list[tuple[str, float, float]],
    time: float,
) -> str | None:
    """Find which state a cell is in at the given time.

    Returns None if the cell has exited the system before `time`.
    """
    for state, entry, exit_time in trajectory:
        if entry <= time < exit_time:
            return state

    # If time >= last exit, cell is still in the last (terminal) state
    if trajectory:
        last_state, _, last_exit = trajectory[-1]
        if time >= last_exit:
            # Check if it's a terminal state (stays there)
            return last_state

    return None
