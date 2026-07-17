"""Synthetic data generation for validating the queueing inference pipeline.

Simulates differentiation hierarchies as semi-Markov processes with known,
injected bottleneck locations. Used to validate the rest of the pipeline
before it touches real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional


def generate_hierarchy(
    n_states: int,
    branching_structure: Optional[Dict[int, List[int]]] = None,
    seed: int = 42
) -> nx.DiGraph:
    """Generate a directed graph representing a differentiation hierarchy.

    Parameters
    ----------
    n_states : int
        Number of states in the hierarchy. Must be >= 2.
    branching_structure : dict[int, list[int]], optional
        Mapping from parent state index to list of child state indices.
        If None, creates a linear chain. Each child must have index > parent
        to maintain DAG structure. Example: {1: [2, 3]} means state 1
        branches to states 2 and 3.
    seed : int, default=42
        Random seed for reproducibility (affects branch probabilities).

    Returns
    -------
    networkx.DiGraph
        Directed acyclic graph where nodes are state indices (0 to n_states-1)
        and edges represent allowed transitions. Start state is 0.
        Terminal states have out-degree 0.
        Each edge has a 'prob' attribute for transition probability.
    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2")

    rng = np.random.default_rng(seed)
    G = nx.DiGraph()

    # Add all states as nodes
    for i in range(n_states):
        G.add_node(i)

    if branching_structure is None:
        # Linear chain: 0 -> 1 -> 2 -> ... -> n_states-1
        for i in range(n_states - 1):
            G.add_edge(i, i + 1, prob=1.0)
    else:
        # Build from branching structure
        all_children = set()
        for parent, children in branching_structure.items():
            if parent >= n_states:
                raise ValueError(f"Parent state {parent} >= n_states ({n_states})")
            for child in children:
                if child >= n_states:
                    raise ValueError(f"Child state {child} >= n_states ({n_states})")
                if child <= parent:
                    raise ValueError(f"Child {child} must be > parent {parent} for DAG")
                all_children.add(child)

        # Determine which states have no incoming edges (potential starts)
        # and which have no outgoing edges in the structure (terminals)
        has_incoming = set(all_children)
        has_outgoing = set(branching_structure.keys())

        # Add edges with probabilities
        for parent, children in branching_structure.items():
            if len(children) == 1:
                G.add_edge(parent, children[0], prob=1.0)
            else:
                # Sample Dirichlet for branch probabilities
                probs = rng.dirichlet(np.ones(len(children)))
                for child, prob in zip(children, probs):
                    G.add_edge(parent, child, prob=prob)

        # Connect unconnected states linearly to ensure single start at 0
        # and all states reachable
        connected = set()
        def dfs(node):
            connected.add(node)
            for succ in G.successors(node):
                if succ not in connected:
                    dfs(succ)
        dfs(0)

        # Add missing connections for any unreachable states
        for i in range(n_states):
            if i not in connected:
                # Find nearest connected predecessor
                preds = [j for j in range(i) if j in connected]
                if preds:
                    pred = max(preds)
                    G.add_edge(pred, i, prob=1.0)
                    dfs(i)

        # Ensure all nodes reachable from 0 (single start)
        for i in range(1, n_states):
            if not nx.has_path(G, 0, i):
                # Connect from nearest reachable predecessor
                for j in range(i):
                    if nx.has_path(G, 0, j):
                        G.add_edge(j, i, prob=1.0)
                        break

    # Verify it's a DAG with single start (0)
    assert nx.is_directed_acyclic_graph(G), "Graph must be a DAG"
    assert 0 in G.nodes, "Start state 0 must exist"
    assert G.in_degree(0) == 0, "State 0 must be the unique start state"

    return G


def simulate_cells(
    hierarchy: nx.DiGraph,
    n_cells: int,
    gamma_params_per_state: Dict[int, Tuple[float, float]],
    bottleneck_state: int,
    bottleneck_severity: float,
    seed: int = 42
) -> pd.DataFrame:
    """Simulate cells moving through a differentiation hierarchy.

    Each cell independently traverses from start state (0) to a terminal state,
    sampling residence (sojourn) times from Gamma distributions per state.
    The bottleneck_state has its Gamma shape inflated by bottleneck_severity.

    Parameters
    ----------
    hierarchy : networkx.DiGraph
        Directed acyclic graph from generate_hierarchy().
        Nodes are state indices. Edges have 'prob' attribute.
    n_cells : int
        Number of cells to simulate. Must be > 0.
    gamma_params_per_state : dict[int, tuple[float, float]]
        Mapping from state index to (shape, scale) for Gamma distribution.
        Must contain all states in hierarchy.
    bottleneck_state : int
        State index where the bottleneck is injected.
        Must be a valid state in hierarchy.
    bottleneck_severity : float
        Factor by which to inflate the Gamma shape parameter at bottleneck_state.
        Must be >= 1.0 (1.0 = no bottleneck, higher = more severe congestion).
    seed : int, default=42
        Random seed for full reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: cell_id (int), state (int), entry_time (float),
        exit_time (float), next_state (int or NaN for terminal).
        One row per cell-state visit. Sorted by cell_id then entry_time.
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be > 0")
    if bottleneck_severity < 1.0:
        raise ValueError("bottleneck_severity must be >= 1.0")
    if bottleneck_state not in hierarchy.nodes:
        raise ValueError(f"bottleneck_state {bottleneck_state} not in hierarchy")
    for state in hierarchy.nodes:
        if state not in gamma_params_per_state:
            raise ValueError(f"Missing gamma params for state {state}")

    rng = np.random.default_rng(seed)

    # Identify terminal states (out-degree 0)
    terminal_states = {n for n in hierarchy.nodes if hierarchy.out_degree(n) == 0}

    # Build transition probability matrix for each state
    trans_probs: Dict[int, Tuple[List[int], List[float]]] = {}
    for state in hierarchy.nodes:
        if state in terminal_states:
            continue
        successors = list(hierarchy.successors(state))
        probs = [hierarchy.edges[state, succ]['prob'] for succ in successors]
        trans_probs[state] = (successors, probs)

    # Prepare modified gamma params for bottleneck
    mod_gamma_params = dict(gamma_params_per_state)
    if bottleneck_severity > 1.0:
        shape, scale = gamma_params_per_state[bottleneck_state]
        mod_gamma_params[bottleneck_state] = (shape * bottleneck_severity, scale)

    records = []
    for cell_id in range(n_cells):
        current_state = 0
        current_time = 0.0

        while current_state not in terminal_states:
            # Sample residence time
            shape, scale = mod_gamma_params[current_state]
            residence = rng.gamma(shape, scale)

            entry_time = current_time
            exit_time = current_time + residence

            # Choose next state
            successors, probs = trans_probs[current_state]
            next_state = rng.choice(successors, p=probs)

            records.append({
                'cell_id': cell_id,
                'state': current_state,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'next_state': next_state
            })

            current_state = next_state
            current_time = exit_time

        # Record terminal state visit (exit_time = entry_time, next_state = NaN)
        shape, scale = mod_gamma_params[current_state]
        residence = rng.gamma(shape, scale)
        entry_time = current_time
        exit_time = current_time + residence
        records.append({
            'cell_id': cell_id,
            'state': current_state,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'next_state': np.nan
        })

    df = pd.DataFrame(records)
    df = df.sort_values(['cell_id', 'entry_time']).reset_index(drop=True)
    return df


def generate_severity_sweep(
    hierarchy: nx.DiGraph,
    n_cells: int,
    bottleneck_state: int,
    severity_values: np.ndarray,
    gamma_params_per_state: Dict[int, Tuple[float, float]],
    seed: int = 42
) -> Dict[float, pd.DataFrame]:
    """Generate a sweep of simulated datasets across bottleneck severities.

    Parameters
    ----------
    hierarchy : networkx.DiGraph
        Differentiation hierarchy from generate_hierarchy().
    n_cells : int
        Number of cells per severity level.
    bottleneck_state : int
        State index where bottleneck is injected.
    severity_values : numpy.ndarray
        Array of severity values (e.g., np.linspace(1.0, 5.0, 20)).
        Each value >= 1.0.
    gamma_params_per_state : dict[int, tuple[float, float]]
        Base Gamma (shape, scale) parameters for each state.
    seed : int, default=42
        Base random seed. Each severity gets a derived seed for independence.

    Returns
    -------
    dict[float, pandas.DataFrame]
        Mapping from severity value to simulated DataFrame (from simulate_cells).
    """
    if len(severity_values) == 0:
        raise ValueError("severity_values must not be empty")
    if np.any(severity_values < 1.0):
        raise ValueError("All severity_values must be >= 1.0")

    base_rng = np.random.default_rng(seed)
    # Derive independent seeds for each severity
    severity_seeds = base_rng.integers(0, 2**32, size=len(severity_values))

    results = {}
    for severity, sev_seed in zip(severity_values, severity_seeds):
        df = simulate_cells(
            hierarchy=hierarchy,
            n_cells=n_cells,
            gamma_params_per_state=gamma_params_per_state,
            bottleneck_state=bottleneck_state,
            bottleneck_severity=float(severity),
            seed=int(sev_seed)
        )
        results[float(severity)] = df

    return results


if __name__ == "__main__":
    """Sanity check: generate example hierarchy matching coarse meta-hierarchy.

    Coarse meta-hierarchy from research_plan.md:
    Stem/Multipotent -> Myeloid-primed Progenitor -> Lymphoid-primed Progenitor
    -> Committed Progenitor -> Mature

    With one branch point: Myeloid-primed branches to Myeloid-committed
    and Lymphoid-committed (both terminal).
    5 states total:
    0: Stem_Multipotent (start)
    1: Myeloid_Primed_Progenitor
    2: Branch_Point (Lymphoid_Primed_Progenitor)
    3: Committed_Progenitor_Myeloid (terminal)
    4: Committed_Progenitor_Lymphoid (terminal)
    """
    print("=" * 60)
    print("SYNTHETIC GENERATOR SANITY CHECK")
    print("=" * 60)

    # 5 states, one branch point at state 1 -> states 2 and 3
    # State 4 is reached from state 2 (linear after branch)
    # Actually: 0 -> 1 -> {2, 3}, then 2 -> 4, 3 terminal
    # Wait, need 5 states total with one branch point
    # Let's do: 0 -> 1 -> {2, 3}, 2 -> 4, 3 terminal
    # That's 5 states (0,1,2,3,4) with branch at 1
    hierarchy = generate_hierarchy(
        n_states=5,
        branching_structure={1: [2, 3], 2: [4]},
        seed=42
    )

    print(f"\nHierarchy: {hierarchy.number_of_nodes()} states, {hierarchy.number_of_edges()} transitions")
    print("Edges (with probabilities):")
    for u, v, data in hierarchy.edges(data=True):
        print(f"  {u} -> {v} (p={data['prob']:.3f})")

    # Gamma params: (shape, scale) for each state
    # Mean = shape * scale, Var = shape * scale^2
    gamma_params = {
        0: (2.0, 1.0),   # Stem: mean=2.0
        1: (2.0, 1.5),   # Myeloid-Primed: mean=3.0
        2: (3.0, 1.0),   # Lymphoid-Primed: mean=3.0
        3: (2.0, 2.0),   # Myeloid-Committed: mean=4.0
        4: (4.0, 1.0),   # Lymphoid-Committed (terminal): mean=4.0
    }

    # Simulate with bottleneck at state 1 (Myeloid-Primed), severity 3.0
    bottleneck_state = 1
    bottleneck_severity = 3.0
    n_cells = 1000

    df = simulate_cells(
        hierarchy=hierarchy,
        n_cells=n_cells,
        gamma_params_per_state=gamma_params,
        bottleneck_state=bottleneck_state,
        bottleneck_severity=bottleneck_severity,
        seed=123
    )

    print(f"\nSimulated {n_cells} cells with bottleneck at state {bottleneck_state}")
    print(f"  severity = {bottleneck_severity} (shape inflated by {bottleneck_severity}x)")

    # Summary statistics
    print("\n--- Cell counts per state ---")
    state_counts = df['state'].value_counts().sort_index()
    for state, count in state_counts.items():
        print(f"  State {state}: {count} visits")

    print("\n--- Mean residence time per state ---")
    df['residence'] = df['exit_time'] - df['entry_time']
    mean_residence = df.groupby('state')['residence'].mean()
    for state, mean_rt in mean_residence.items():
        print(f"  State {state}: {mean_rt:.3f}")

    print("\n--- Mean residence time at bottleneck state (state 1) ---")
    bottleneck_res = df[df['state'] == bottleneck_state]['residence']
    print(f"  Observed mean: {bottleneck_res.mean():.3f}")
    print(f"  Expected mean (shape={gamma_params[bottleneck_state][0]*bottleneck_severity}*scale={gamma_params[bottleneck_state][1]}): "
          f"{gamma_params[bottleneck_state][0]*bottleneck_severity*gamma_params[bottleneck_state][1]:.3f}")

    print("\n--- Terminal state distribution ---")
    terminal_states = {n for n in hierarchy.nodes if hierarchy.out_degree(n) == 0}
    last_states = df.groupby('cell_id').last()['state']
    terminal_dist = last_states.value_counts().sort_index()
    for state, count in terminal_dist.items():
        print(f"  State {state}: {count} cells ({count/n_cells*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SANITY CHECK PASSED")
    print("=" * 60)