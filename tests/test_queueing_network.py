"""Tests for queueing_network module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queuediff.queueing_network import QueueingNetwork, build_from_data


@pytest.fixture
def simple_network():
    """Three-state linear queueing network."""
    net = QueueingNetwork(name="test")
    net.add_state("HSC", service_rate=0.05)
    net.add_state("MPP", service_rate=0.08)
    net.add_state("GMP", service_rate=0.04)
    net.add_transition("HSC", "MPP", probability=1.0)
    net.add_transition("MPP", "GMP", probability=1.0)
    return net


@pytest.fixture
def branching_network():
    """Branching network with two terminals."""
    net = QueueingNetwork(name="branch")
    net.add_state("HSC", service_rate=0.06)
    net.add_state("MPP", service_rate=0.08)
    net.add_state("MEP", service_rate=0.05)
    net.add_state("GMP", service_rate=0.04)
    net.add_transition("HSC", "MPP", probability=1.0)
    net.add_transition("MPP", "MEP", probability=0.6)
    net.add_transition("MPP", "GMP", probability=0.4)
    return net


class TestQueueingNetwork:
    def test_add_state(self):
        net = QueueingNetwork()
        net.add_state("HSC", service_rate=0.05)
        assert "HSC" in net.states

    def test_states_property_order(self, simple_network):
        assert simple_network.states == ["HSC", "MPP", "GMP"]

    def test_add_transition_invalid_source(self, simple_network):
        with pytest.raises(ValueError, match="Source"):
            simple_network.add_transition("INVALID", "MPP", probability=1.0)

    def test_add_transition_invalid_target(self, simple_network):
        with pytest.raises(ValueError, match="Target"):
            simple_network.add_transition("HSC", "INVALID", probability=1.0)

    def test_routing_matrix_shape(self, simple_network):
        rm = simple_network.routing_matrix()
        assert rm.shape == (3, 3)

    def test_routing_matrix_values(self, simple_network):
        rm = simple_network.routing_matrix()
        assert rm.loc["HSC", "MPP"] == 1.0
        assert rm.loc["MPP", "GMP"] == 1.0
        assert rm.loc["GMP", "HSC"] == 0.0

    def test_routing_matrix_branching(self, branching_network):
        rm = branching_network.routing_matrix()
        assert rm.loc["MPP", "MEP"] == 0.6
        assert rm.loc["MPP", "GMP"] == 0.4


class TestArrivalRates:
    def test_linear_propagation(self, simple_network):
        rates = simple_network.arrival_rates(external_arrival=1.0, source="HSC")
        assert rates["HSC"] == 1.0
        assert rates["MPP"] == 1.0
        assert rates["GMP"] == 1.0

    def test_branching_propagation(self, branching_network):
        rates = branching_network.arrival_rates(external_arrival=1.0, source="HSC")
        assert rates["HSC"] == 1.0
        assert rates["MPP"] == 1.0
        assert abs(rates["MEP"] - 0.6) < 1e-10
        assert abs(rates["GMP"] - 0.4) < 1e-10

    def test_cycle_raises(self):
        net = QueueingNetwork()
        net.add_state("A", service_rate=0.1)
        net.add_state("B", service_rate=0.1)
        net.add_transition("A", "B", probability=0.5)
        net.add_transition("B", "A", probability=0.5)
        with pytest.raises(ValueError, match="DAG"):
            net.arrival_rates(external_arrival=1.0)

    def test_zero_external_gives_zero(self, simple_network):
        rates = simple_network.arrival_rates(external_arrival=0.0, source="HSC")
        assert all(v == 0.0 for v in rates.values())


class TestTrafficIntensity:
    def test_computation(self, simple_network):
        rates = simple_network.arrival_rates(external_arrival=0.01, source="HSC")
        rho = simple_network.traffic_intensity(rates)
        # ρ = λ/(c*μ), c=1 for all
        assert abs(rho["HSC"] - 0.01 / 0.05) < 1e-10
        assert abs(rho["MPP"] - 0.01 / 0.08) < 1e-10
        assert abs(rho["GMP"] - 0.01 / 0.04) < 1e-10

    def test_zero_service_rate_gives_inf(self):
        net = QueueingNetwork()
        net.add_state("A", service_rate=0.0)
        rho = net.traffic_intensity({"A": 1.0})
        assert rho["A"] == np.inf

    def test_zero_arrival_zero_service(self):
        net = QueueingNetwork()
        net.add_state("A", service_rate=0.0)
        rho = net.traffic_intensity({"A": 0.0})
        assert rho["A"] == 0.0


class TestSummary:
    def test_returns_dataframe(self, simple_network):
        rates = simple_network.arrival_rates(external_arrival=0.01, source="HSC")
        summary = simple_network.summary(arrival_rates=rates)
        assert isinstance(summary, pd.DataFrame)

    def test_correct_columns(self, simple_network):
        rates = simple_network.arrival_rates(external_arrival=0.01, source="HSC")
        summary = simple_network.summary(arrival_rates=rates)
        expected = {"state", "service_rate", "servers", "arrival_rate", "traffic_intensity"}
        assert set(summary.columns) == expected


class TestBuildFromData:
    def test_creates_network(self):
        service_rates = {"HSC": 0.06, "MPP": 0.08, "GMP": 0.05}
        routing = {"HSC": {"MPP": 1.0}, "MPP": {"GMP": 1.0}}
        net = build_from_data(service_rates, routing, name="test")
        assert isinstance(net, QueueingNetwork)
        assert set(net.states) == {"HSC", "MPP", "GMP"}

    def test_auto_adds_missing_states(self):
        """States in routing but not in service_rates get added with rate=0."""
        service_rates = {"HSC": 0.06}
        routing = {"HSC": {"MPP": 1.0}}
        net = build_from_data(service_rates, routing)
        assert "MPP" in net.states
        # MPP auto-added with service_rate=0
        assert net.graph.nodes["MPP"]["service_rate"] == 0.0

    def test_preserves_service_rates(self):
        service_rates = {"HSC": 0.06, "MPP": 0.08}
        routing = {"HSC": {"MPP": 1.0}}
        net = build_from_data(service_rates, routing)
        assert net.graph.nodes["HSC"]["service_rate"] == 0.06
        assert net.graph.nodes["MPP"]["service_rate"] == 0.08
