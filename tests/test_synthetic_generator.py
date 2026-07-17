import numpy as np
import pandas as pd
import pytest
import networkx as nx


# ---------------------------------------------------------------------------
# generate_hierarchy  —  returns a networkx.DiGraph representing states and
#                        allowed transitions with edge probabilities.
# ---------------------------------------------------------------------------


class TestGenerateHierarchy:
    def test_returns_digraph(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, seed=42)
        assert isinstance(G, nx.DiGraph)

    def test_correct_number_of_states(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, seed=42)
        assert G.number_of_nodes() == 5

    def test_states_are_consecutive_integers(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, seed=42)
        assert set(G.nodes()) == {0, 1, 2, 3, 4}

    def test_start_state_is_zero(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, seed=42)
        assert G.in_degree(0) == 0

    def test_is_dag(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, seed=42)
        assert nx.is_directed_acyclic_graph(G)

    def test_linear_chain_by_default(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=4, seed=42)
        edges = list(G.edges())
        assert edges == [(0, 1), (1, 2), (2, 3)]

    def test_branch_point_creates_multiple_children(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, branching_structure={1: [2, 3], 2: [4]}, seed=42)
        assert set(G.successors(1)) == {2, 3}

    def test_edge_probabilities_sum_to_one(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G = generate_hierarchy(n_states=5, branching_structure={1: [2, 3], 2: [4]}, seed=42)
        for node in G.nodes():
            if G.out_degree(node) > 0:
                probs = [G.edges[node, succ]['prob'] for succ in G.successors(node)]
                assert abs(sum(probs) - 1.0) < 1e-12

    def test_reproducible_with_seed(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G1 = generate_hierarchy(n_states=5, branching_structure={1: [2, 3]}, seed=42)
        G2 = generate_hierarchy(n_states=5, branching_structure={1: [2, 3]}, seed=42)
        assert list(G1.edges()) == list(G2.edges())
        for u, v in G1.edges():
            assert G1.edges[u, v]['prob'] == G2.edges[u, v]['prob']

    def test_different_seeds_produce_different_branch_probs(self):
        from queuediff.synthetic_generator import generate_hierarchy
        G1 = generate_hierarchy(n_states=5, branching_structure={1: [2, 3]}, seed=42)
        G2 = generate_hierarchy(n_states=5, branching_structure={1: [2, 3]}, seed=123)
        probs1 = [G1.edges[1, s]['prob'] for s in G1.successors(1)]
        probs2 = [G2.edges[1, s]['prob'] for s in G2.successors(1)]
        assert probs1 != probs2


# ---------------------------------------------------------------------------
# simulate_cells  —  returns a pandas DataFrame with columns:
#                    cell_id, state, entry_time, exit_time, next_state
# ---------------------------------------------------------------------------


class TestSimulateCells:
    def test_returns_dataframe(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                            bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                            bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        expected_cols = {'cell_id', 'state', 'entry_time', 'exit_time', 'next_state'}
        assert set(df.columns) == expected_cols

    def test_correct_number_of_rows(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        n_cells = 500
        df = simulate_cells(G, n_cells=n_cells, gamma_params_per_state=gamma_params,
                            bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        # Each cell visits each state on its path once, plus terminal state
        # For linear chain 0->1->2, each cell has 3 rows
        assert len(df) == n_cells * 3

    def test_cell_ids_are_consecutive(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                            bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        assert list(df['cell_id'].unique()) == list(range(100))

    def test_residence_times_positive(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                            bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        df['residence'] = df['exit_time'] - df['entry_time']
        assert (df['residence'] > 0).all()

    def test_next_state_nan_for_terminal(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                            bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        terminal_rows = df[df['state'] == 2]
        assert terminal_rows['next_state'].isna().all()

    def test_reproducible_with_seed(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df1 = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                             bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        df2 = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                             bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        df1 = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                             bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        df2 = simulate_cells(G, n_cells=100, gamma_params_per_state=gamma_params,
                             bottleneck_state=1, bottleneck_severity=1.0, seed=123)
        # At least some residence times should differ
        assert not df1['exit_time'].equals(df2['exit_time'])

    def test_bottleneck_increases_mean_residence_time(self):
        from queuediff.synthetic_generator import generate_hierarchy, simulate_cells
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        # No bottleneck
        df_no = simulate_cells(G, n_cells=2000, gamma_params_per_state=gamma_params,
                               bottleneck_state=1, bottleneck_severity=1.0, seed=42)
        # With bottleneck severity 5.0
        df_bottleneck = simulate_cells(G, n_cells=2000, gamma_params_per_state=gamma_params,
                                       bottleneck_state=1, bottleneck_severity=5.0, seed=42)
        mean_no = df_no[df_no['state'] == 1]['exit_time'].mean() - \
                  df_no[df_no['state'] == 1]['entry_time'].mean()
        mean_bott = df_bottleneck[df_bottleneck['state'] == 1]['exit_time'].mean() - \
                    df_bottleneck[df_bottleneck['state'] == 1]['entry_time'].mean()
        assert mean_bott > mean_no


# ---------------------------------------------------------------------------
# generate_severity_sweep  —  returns dict mapping severity -> DataFrame
# ---------------------------------------------------------------------------


class TestGenerateSeveritySweep:
    def test_returns_dict(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_severity_sweep
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        sweep = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                        severity_values=np.array([1.0, 2.0, 3.0]),
                                        gamma_params_per_state=gamma_params, seed=42)
        assert isinstance(sweep, dict)

    def test_correct_number_of_severities(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_severity_sweep
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        sevs = np.linspace(1.0, 5.0, 10)
        sweep = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                        severity_values=sevs,
                                        gamma_params_per_state=gamma_params, seed=42)
        assert len(sweep) == len(sevs)

    def test_keys_are_severity_values(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_severity_sweep
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        sevs = np.array([1.0, 2.5, 4.0])
        sweep = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                        severity_values=sevs,
                                        gamma_params_per_state=gamma_params, seed=42)
        assert set(sweep.keys()) == {1.0, 2.5, 4.0}

    def test_each_value_is_dataframe(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_severity_sweep
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        sweep = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                        severity_values=np.array([1.0, 2.0]),
                                        gamma_params_per_state=gamma_params, seed=42)
        for df in sweep.values():
            assert isinstance(df, pd.DataFrame)
            expected_cols = {'cell_id', 'state', 'entry_time', 'exit_time', 'next_state'}
            assert set(df.columns) == expected_cols

    def test_reproducible_with_seed(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_severity_sweep
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        sweep1 = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                         severity_values=np.array([1.0, 2.0]),
                                         gamma_params_per_state=gamma_params, seed=42)
        sweep2 = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                         severity_values=np.array([1.0, 2.0]),
                                         gamma_params_per_state=gamma_params, seed=42)
        for sev in sweep1:
            pd.testing.assert_frame_equal(sweep1[sev], sweep2[sev])

    def test_different_seeds_produce_different_sweeps(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_severity_sweep
        G = generate_hierarchy(n_states=3, seed=42)
        gamma_params = {0: (2.0, 1.0), 1: (2.0, 1.0), 2: (2.0, 1.0)}
        sweep1 = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                         severity_values=np.array([1.0, 2.0]),
                                         gamma_params_per_state=gamma_params, seed=42)
        sweep2 = generate_severity_sweep(G, n_cells=100, bottleneck_state=1,
                                         severity_values=np.array([1.0, 2.0]),
                                         gamma_params_per_state=gamma_params, seed=123)
        for sev in sweep1:
            assert not sweep1[sev]['exit_time'].equals(sweep2[sev]['exit_time'])


# ---------------------------------------------------------------------------
# End-to-end recovery test  —  verify that the pipeline can detect the
#                               injected bottleneck.
#
# This test simulates the full inference loop (fit → queueing → rank)
# with a strong bottleneck and checks that the true bottleneck state is
# ranked first. Kept skipped because downstream modules don't exist yet.
# ---------------------------------------------------------------------------


class TestEndToEndRecovery:
    @pytest.mark.skip(reason="awaiting implementation of distribution_fitting, queueing_network, bottleneck_diagnostics")
    def test_detects_strong_bottleneck(self):
        pass

    @pytest.mark.skip(reason="awaiting implementation of distribution_fitting, queueing_network, bottleneck_diagnostics")
    def test_no_bottleneck_when_severity_is_one(self):
        pass