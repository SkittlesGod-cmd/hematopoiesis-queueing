"""Queueing network model for hematopoietic differentiation.

Backed by networkx DiGraph. Each node is a hematopoietic state with
a service rate (μ = 1/mean_residence_time). Each edge has a routing
probability. Traffic intensity ρ = λ/(c·μ) identifies bottlenecks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


class QueueingNetwork:
    """Semi-Markov queueing network for differentiation analysis.

    Attributes
    ----------
    graph : nx.DiGraph
        Internal directed graph storing states and transitions.
    name : str
        Human-readable name for the network.
    """

    def __init__(self, name: str = "") -> None:
        self.graph = nx.DiGraph()
        self.name = name

    def add_state(
        self,
        name: str,
        service_rate: float = 0.0,
        servers: int = 1,
    ) -> None:
        """Add a state (node) to the network.

        Parameters
        ----------
        name : str
            State name (e.g., 'HSC', 'MPP').
        service_rate : float, default 0.0
            Service rate μ = 1/mean_residence_time (per hour).
        servers : int, default 1
            Number of parallel servers (c). For biological states,
            typically 1 (single effective server per state).
        """
        self.graph.add_node(name, service_rate=service_rate, servers=servers)

    def add_transition(
        self,
        source: str,
        target: str,
        probability: float,
    ) -> None:
        """Add a transition (edge) between states.

        Parameters
        ----------
        source : str
            Source state name.
        target : str
            Target state name.
        probability : float
            Routing probability (0 to 1).
        """
        if source not in self.graph:
            raise ValueError(f"Source state '{source}' not in network.")
        if target not in self.graph:
            raise ValueError(f"Target state '{target}' not in network.")
        self.graph.add_edge(source, target, probability=probability)

    @property
    def states(self) -> list[str]:
        """Return state names in insertion order."""
        return list(self.graph.nodes)

    def routing_matrix(self) -> pd.DataFrame:
        """Compute the routing probability matrix.

        Returns
        -------
        pd.DataFrame
            States × states matrix where entry (i,j) is the probability
            of routing from state i to state j.
        """
        states = self.states
        n = len(states)
        matrix = np.zeros((n, n))

        state_idx = {s: i for i, s in enumerate(states)}
        for src, tgt, data in self.graph.edges(data=True):
            i, j = state_idx[src], state_idx[tgt]
            matrix[i, j] = data.get("probability", 0.0)

        return pd.DataFrame(matrix, index=states, columns=states)

    def arrival_rates(
        self,
        external_arrival: float = 0.0,
        source: str | None = None,
    ) -> dict[str, float]:
        """Propagate arrival rates through the network via topological sort.

        Parameters
        ----------
        external_arrival : float, default 0.0
            External arrival rate into the source state (per hour, normalized).
        source : str, optional
            Source state. If None, uses the first state with no predecessors.

        Returns
        -------
        dict[str, float]
            State -> arrival rate.

        Raises
        ------
        ValueError
            If the graph has cycles.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                "Arrival rate propagation requires a DAG (no cycles). "
                "The differentiation network should be acyclic."
            )

        # Find source state
        if source is None:
            roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
            if not roots:
                raise ValueError("No root state found (all states have predecessors).")
            source = roots[0]

        rates = {s: 0.0 for s in self.states}
        rates[source] = external_arrival

        # Topological order propagation
        for node in nx.topological_sort(self.graph):
            for _, successor, data in self.graph.out_edges(node, data=True):
                prob = data.get("probability", 0.0)
                rates[successor] += rates[node] * prob

        return rates

    def traffic_intensity(
        self,
        arrival_rates: dict[str, float],
    ) -> dict[str, float]:
        """Compute traffic intensity ρ = λ/(c·μ) per state.

        Parameters
        ----------
        arrival_rates : dict[str, float]
            Per-state arrival rates (from self.arrival_rates or external).

        Returns
        -------
        dict[str, float]
            State -> traffic intensity. Returns np.inf when service_rate=0.
        """
        result = {}
        for state in self.states:
            lam = arrival_rates.get(state, 0.0)
            mu = self.graph.nodes[state].get("service_rate", 0.0)
            c = self.graph.nodes[state].get("servers", 1)

            # Handle service_rate=0 explicitly (not via raw division)
            if mu == 0.0:
                result[state] = np.inf if lam > 0 else 0.0
            else:
                result[state] = lam / (c * mu)

        return result

    def summary(
        self,
        arrival_rates: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Generate summary DataFrame for the network.

        Parameters
        ----------
        arrival_rates : dict, optional
            Pre-computed arrival rates. If None, computed automatically.

        Returns
        -------
        pd.DataFrame
            Columns: state, service_rate, servers, arrival_rate, traffic_intensity.
        """
        if arrival_rates is None:
            try:
                arrival_rates = self.arrival_rates()
            except ValueError:
                arrival_rates = {s: 0.0 for s in self.states}

        rho = self.traffic_intensity(arrival_rates)

        records = []
        for state in self.states:
            records.append({
                "state": state,
                "service_rate": self.graph.nodes[state].get("service_rate", 0.0),
                "servers": self.graph.nodes[state].get("servers", 1),
                "arrival_rate": arrival_rates.get(state, 0.0),
                "traffic_intensity": rho.get(state, 0.0),
            })

        return pd.DataFrame(records)


def build_from_data(
    service_rates: dict[str, float],
    routing_probs: dict[str, dict[str, float]],
    name: str = "",
) -> QueueingNetwork:
    """Factory to build a QueueingNetwork from fitted data.

    Parameters
    ----------
    service_rates : dict[str, float]
        State -> service rate μ (per hour).
    routing_probs : dict[str, dict[str, float]]
        Source -> {target: probability} routing matrix.
    name : str, optional
        Network name.

    Returns
    -------
    QueueingNetwork
        Populated network.

    Notes
    -----
    States in routing_probs not in service_rates are auto-added
    with service_rate=0.0.
    """
    network = QueueingNetwork(name=name)

    # Collect all states
    all_states = set(service_rates.keys())
    for src, targets in routing_probs.items():
        all_states.add(src)
        all_states.update(targets.keys())

    # Add states
    for state in sorted(all_states):
        rate = service_rates.get(state, 0.0)
        network.add_state(state, service_rate=rate)

    # Add transitions
    for src, targets in routing_probs.items():
        for tgt, prob in targets.items():
            network.add_transition(src, tgt, prob)

    return network
