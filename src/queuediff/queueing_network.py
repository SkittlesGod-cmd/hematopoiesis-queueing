"""Queueing network for hematopoiesis differentiation states.

A DAG-based queueing network where states are nodes with service rates
and servers, and transitions are edges with routing probabilities.
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.exception import NetworkXUnfeasible


class QueueingNetwork:
    """Queueing network representing differentiation states and transitions."""

    def __init__(self, name: str = ""):
        self.name = name
        self.graph = nx.DiGraph()

    def add_state(self, name: str, service_rate: float = 0.0, servers: int = 1):
        """Add a state to the network.

        Args:
            name: State name (node identifier).
            service_rate: Service rate (mu) for this state. Default 0.0.
            servers: Number of parallel servers (c). Default 1.
        """
        self.graph.add_node(name, service_rate=service_rate, servers=servers)

    def add_transition(self, source: str, target: str, probability: float):
        """Add a directed transition between states.

        Args:
            source: Source state name.
            target: Target state name.
            probability: Routing probability from source to target.
        """
        if source not in self.graph:
            raise ValueError(f"Source state '{source}' not found")
        if target not in self.graph:
            raise ValueError(f"Target state '{target}' not found")
        self.graph.add_edge(source, target, probability=probability)

    @property
    def states(self) -> list[str]:
        """Return list of state names in insertion order."""
        return list(self.graph.nodes())

    def routing_matrix(self) -> pd.DataFrame:
        """Return routing probability matrix as DataFrame.

        Rows = source states, columns = target states.
        Missing edges get 0.0 probability.
        """
        states = self.states
        if not states:
            return pd.DataFrame(index=[], columns=[])

        mat = pd.DataFrame(0.0, index=states, columns=states)
        for source, target, data in self.graph.edges(data=True):
            mat.loc[source, target] = data.get("probability", 0.0)
        return mat

    def arrival_rates(self, external_arrival: float = 0.0, source: str = None) -> dict[str, float]:
        """Compute arrival rates at each state via topological propagation.

        Args:
            external_arrival: Arrival rate injected at the source state.
            source: Source state name where external_arrival is injected.
                   If None, all states get 0.0.

        Returns:
            Dict mapping state name to arrival rate (lambda).

        Raises:
            NetworkXUnfeasible: If the graph contains a cycle.
        """
        rates = {state: 0.0 for state in self.states}

        if source is None:
            return rates

        if source not in self.graph:
            raise ValueError(f"Source state '{source}' not found in network")

        # Topological sort to ensure correct propagation order
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except NetworkXUnfeasible as e:
            raise NetworkXUnfeasible("Graph contains a cycle; differentiation must be a DAG") from e

        rates[source] = external_arrival

        for node in topo_order:
            if node == source:
                continue

            # Sum arrivals from all predecessors
            total = 0.0
            for pred in self.graph.predecessors(node):
                prob = self.graph.edges[(pred, node)].get("probability", 0.0)
                total += rates[pred] * prob
            rates[node] = total

        return rates

    def traffic_intensity(self, arrival_rates: dict[str, float]) -> dict[str, float]:
        """Compute traffic intensity (rho) for each state.

        rho = lambda / (servers * service_rate)

        Args:
            arrival_rates: Dict mapping state name to arrival rate (lambda).

        Returns:
            Dict mapping state name to traffic intensity (rho).
            Returns np.inf for states with service_rate == 0.
        """
        rho = {}
        for state in self.states:
            rate = arrival_rates.get(state, 0.0)
            service_rate = self.graph.nodes[state].get("service_rate", 0.0)
            servers = self.graph.nodes[state].get("servers", 1)

            if service_rate == 0.0:
                rho[state] = np.inf if rate > 0.0 else 0.0
            else:
                rho[state] = rate / (servers * service_rate)
        return rho

    def summary(self, arrival_rates: dict = None) -> pd.DataFrame:
        """Return summary DataFrame with one row per state.

        Columns: state, service_rate, servers, arrival_rate, traffic_intensity

        Args:
            arrival_rates: Optional dict of arrival rates. If None, uses zeros.
        """
        if arrival_rates is None:
            arrival_rates = {state: 0.0 for state in self.states}

        rho = self.traffic_intensity(arrival_rates)

        rows = []
        for state in self.states:
            rows.append({
                "state": state,
                "service_rate": self.graph.nodes[state].get("service_rate", 0.0),
                "servers": self.graph.nodes[state].get("servers", 1),
                "arrival_rate": arrival_rates.get(state, 0.0),
                "traffic_intensity": rho.get(state, 0.0),
            })

        return pd.DataFrame(rows)


def build_from_data(
    service_rates: dict,
    routing_probs: dict[tuple, float],
    name: str = ""
) -> QueueingNetwork:
    """Build a QueueingNetwork from dictionaries.

    Args:
        service_rates: Dict mapping state name -> service_rate.
        routing_probs: Dict mapping (source, target) tuple -> probability.
        name: Network name.

    Returns:
        QueueingNetwork instance.
    """
    qn = QueueingNetwork(name=name)

    # Add states from service_rates
    for state, rate in service_rates.items():
        qn.add_state(state, service_rate=rate)

    # Add transitions from routing_probs
    for (source, target), prob in routing_probs.items():
        # Auto-add missing states with service_rate=0.0
        if source not in qn.graph:
            qn.add_state(source, service_rate=0.0)
        if target not in qn.graph:
            qn.add_state(target, service_rate=0.0)
        qn.add_transition(source, target, prob)

    return qn