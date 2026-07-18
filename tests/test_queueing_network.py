import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# QueueingNetwork  —  DAG of differentiation states with service rates,
#                     routing probabilities, arrival-rate propagation, and
#                     traffic-intensity computation.
#
#   Expected public API:
#       QueueingNetwork(name="")
#       .add_state(name, service_rate, servers)
#       .add_transition(source, target, probability)
#       .states              -> list[str]
#       .routing_matrix()    -> pd.DataFrame
#       .arrival_rates(external_arrival, source) -> dict[str, float]
#       .traffic_intensity(arrival_rates)        -> dict[str, float]
#       .summary(arrival_rates)                  -> pd.DataFrame
#
#   build_from_data(service_rates, routing_probs, name) -> QueueingNetwork
# ---------------------------------------------------------------------------


class TestQueueingNetworkEmpty:
    
    def test_empty_network_has_no_states(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        assert qn.states == []

    
    def test_empty_network_routing_matrix_is_empty(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        mat = qn.routing_matrix()
        assert mat.shape == (0, 0)


class TestQueueingNetworkAddState:
    
    def test_add_single_state(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC", service_rate=1.0, servers=1)
        assert "HSC" in qn.states
        assert len(qn.states) == 1

    
    def test_service_rate_stored(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC", service_rate=2.5, servers=1)
        assert qn.graph.nodes["HSC"]["service_rate"] == 2.5

    
    def test_default_servers_is_one(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC", service_rate=1.0)
        assert qn.graph.nodes["HSC"]["servers"] == 1

    
    def test_add_multiple_states(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        for s in ["HSC", "MPP", "CMP", "LMPP"]:
            qn.add_state(s, service_rate=1.0)
        assert len(qn.states) == 4


class TestQueueingNetworkAddTransition:
    
    def test_add_transition(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC")
        qn.add_state("MPP")
        qn.add_transition("HSC", "MPP", probability=1.0)
        assert qn.graph.has_edge("HSC", "MPP")

    
    def test_transition_probability_stored(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("MPP")
        qn.add_state("CMP")
        qn.add_transition("MPP", "CMP", probability=0.5)
        assert qn.graph.edges[("MPP", "CMP")]["probability"] == 0.5

    
    def test_total_probability_may_be_one(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("MPP")
        qn.add_state("CMP")
        qn.add_state("LMPP")
        qn.add_transition("MPP", "CMP", 0.5)
        qn.add_transition("MPP", "LMPP", 0.5)
        mat = qn.routing_matrix()
        total_out = mat.loc["MPP"].sum()
        assert abs(total_out - 1.0) < 1e-12


class TestQueueingNetworkRoutingMatrix:
    
    def test_shape(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        for s in ["HSC", "MPP", "CMP"]:
            qn.add_state(s)
        qn.add_transition("HSC", "MPP", 1.0)
        qn.add_transition("MPP", "CMP", 0.5)
        mat = qn.routing_matrix()
        assert mat.shape == (3, 3)
        assert list(mat.index) == ["HSC", "MPP", "CMP"]

    
    def test_zero_for_no_edge(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC")
        qn.add_state("CMP")
        mat = qn.routing_matrix()
        assert mat.loc["HSC", "CMP"] == 0.0


class TestQueueingNetworkArrivalRates:
    
    def test_external_arrival_set_on_source(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC")
        qn.add_state("MPP")
        qn.add_transition("HSC", "MPP", 1.0)
        rates = qn.arrival_rates(external_arrival=10.0, source="HSC")
        assert rates["HSC"] == 10.0

    
    def test_arrival_propagates_downstream(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC")
        qn.add_state("MPP")
        qn.add_transition("HSC", "MPP", 1.0)
        rates = qn.arrival_rates(external_arrival=5.0, source="HSC")
        assert abs(rates["MPP"] - 5.0) < 1e-12

    
    def test_split_routing(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("MPP")
        qn.add_state("CMP")
        qn.add_state("LMPP")
        qn.add_transition("MPP", "CMP", 0.4)
        qn.add_transition("MPP", "LMPP", 0.6)
        rates = qn.arrival_rates(external_arrival=10.0, source="MPP")
        assert abs(rates["CMP"] - 4.0) < 1e-12
        assert abs(rates["LMPP"] - 6.0) < 1e-12

    
    def test_no_source_returns_zeros(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC")
        rates = qn.arrival_rates()
        assert rates["HSC"] == 0.0

    
    def test_all_arrival_rates_non_negative(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        for s in ["HSC", "MPP", "CMP", "LMPP"]:
            qn.add_state(s)
        qn.add_transition("HSC", "MPP", 1.0)
        qn.add_transition("MPP", "CMP", 0.5)
        qn.add_transition("MPP", "LMPP", 0.5)
        rates = qn.arrival_rates(external_arrival=10.0, source="HSC")
        for v in rates.values():
            assert v >= 0.0, f"Negative arrival rate: {v}"


class TestQueueingNetworkTrafficIntensity:
    
    def test_rho_non_negative(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC", service_rate=2.0)
        rates = {"HSC": 1.0}
        rho = qn.traffic_intensity(rates)
        assert rho["HSC"] >= 0.0

    
    def test_stable_state_rho_less_than_one(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("HSC", service_rate=10.0, servers=1)
        qn.add_state("MPP", service_rate=5.0, servers=1)
        qn.add_transition("HSC", "MPP", 1.0)
        rates = {"HSC": 1.0, "MPP": 1.0}
        rho = qn.traffic_intensity(rates)
        assert rho["HSC"] < 1.0, f"HSC ρ={rho['HSC']:.3f} should be < 1"
        assert rho["MPP"] < 1.0, f"MPP ρ={rho['MPP']:.3f} should be < 1"

    
    def test_unstable_state_rho_greater_than_one(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("Bottleneck", service_rate=0.5, servers=1)
        rates = {"Bottleneck": 1.0}
        rho = qn.traffic_intensity(rates)
        assert rho["Bottleneck"] > 1.0, (
            f"Bottleneck ρ={rho['Bottleneck']:.3f} should be > 1 "
            "when λ=1.0 and μ=0.5"
        )

    
    def test_rho_formula(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("Test", service_rate=4.0, servers=2)
        rates = {"Test": 3.0}
        rho = qn.traffic_intensity(rates)
        expected = 3.0 / (2 * 4.0)
        assert abs(rho["Test"] - expected) < 1e-12, (
            f"ρ should equal λ/(c·μ) = {expected}, got {rho['Test']}"
        )

    
    def test_zero_service_rate_returns_inf(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("Stuck", service_rate=0.0)
        rates = {"Stuck": 1.0}
        rho = qn.traffic_intensity(rates)
        assert np.isinf(rho["Stuck"])


class TestQueueingNetworkSummary:
    
    def test_returns_dataframe(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("A", service_rate=1.0)
        qn.add_state("B", service_rate=2.0)
        df = qn.summary()
        assert isinstance(df, pd.DataFrame)

    
    def test_columns_present(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        qn.add_state("A", service_rate=1.0)
        df = qn.summary()
        for col in ("state", "service_rate", "servers", "arrival_rate",
                     "traffic_intensity"):
            assert col in df.columns, f"Missing column: {col}"

    
    def test_one_row_per_state(self):
        from queuediff.queueing_network import QueueingNetwork
        qn = QueueingNetwork()
        for s in ["X", "Y", "Z"]:
            qn.add_state(s, service_rate=1.0)
        df = qn.summary()
        assert list(df["state"]) == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# build_from_data  —  convenience factory
# ---------------------------------------------------------------------------

class TestBuildFromData:
    
    def test_returns_QueueingNetwork(self):
        from queuediff.queueing_network import build_from_data
        service_rates = {"HSC": 1.0, "MPP": 2.0}
        routing = {("HSC", "MPP"): 1.0}
        qn = build_from_data(service_rates, routing, name="test")
        from queuediff.queueing_network import QueueingNetwork
        assert isinstance(qn, QueueingNetwork)

    
    def test_name_set(self):
        from queuediff.queueing_network import build_from_data
        qn = build_from_data({"HSC": 1.0}, {}, name="my_network")
        assert qn.name == "my_network"

    
    def test_states_from_service_rates(self):
        from queuediff.queueing_network import build_from_data
        service_rates = {"HSC": 1.0, "MPP": 2.0, "CMP": 3.0}
        qn = build_from_data(service_rates, {}, name="test")
        assert set(qn.states) == {"HSC", "MPP", "CMP"}

    
    def test_transitions_added(self):
        from queuediff.queueing_network import build_from_data
        qn = build_from_data(
            {"HSC": 1.0, "MPP": 2.0},
            {("HSC", "MPP"): 1.0},
        )
        assert qn.graph.has_edge("HSC", "MPP")

    
    def test_missing_service_rate_defaults_to_zero(self):
        from queuediff.queueing_network import build_from_data
        qn = build_from_data(
            {"HSC": 1.0},
            {("HSC", "MPP"): 1.0},
        )
        assert "MPP" in qn.states
        assert qn.graph.nodes["MPP"]["service_rate"] == 0.0

    
    def test_missing_routing_state_added(self):
        from queuediff.queueing_network import build_from_data
        qn = build_from_data(
            {"HSC": 1.0},
            {("HSC", "MPP"): 1.0},
        )
        assert "MPP" in qn.states
