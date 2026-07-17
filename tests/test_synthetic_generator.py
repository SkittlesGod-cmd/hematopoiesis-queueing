import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# generate_hierarchy  —  returns a dict with keys:
#                           "hierarchy"     : dict of state configs
#                           "routing_probs" : dict of (src, tgt) -> prob
#                           "states"        : list of state names
# ---------------------------------------------------------------------------


class TestGenerateHierarchy:
    @pytest.mark.skip(reason="awaiting implementation")
    def test_returns_correct_keys(self):
        from queuediff.synthetic_generator import generate_hierarchy
        result = generate_hierarchy(seed=42)
        for k in ("hierarchy", "routing_probs", "states"):
            assert k in result, f"Missing key: {k}"

    @pytest.mark.skip(reason="awaiting implementation")
    def test_hierarchy_contains_service_rates(self):
        from queuediff.synthetic_generator import generate_hierarchy
        result = generate_hierarchy(seed=42)
        for state_name, config in result["hierarchy"].items():
            assert "service_rate" in config, (
                f"State {state_name} missing service_rate"
            )
            assert config["service_rate"] > 0, (
                f"State {state_name} has non-positive service_rate"
            )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_hierarchy_contains_children(self):
        from queuediff.synthetic_generator import generate_hierarchy
        result = generate_hierarchy(seed=42)
        for state_name, config in result["hierarchy"].items():
            assert "children" in config, (
                f"State {state_name} missing children list"
            )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_hsc_has_external_arrival(self):
        from queuediff.synthetic_generator import generate_hierarchy
        result = generate_hierarchy(seed=42)
        hsc = result["hierarchy"]["HSC"]
        assert hsc.get("arrival", 0) > 0, (
            "HSC should have a positive external arrival rate as the root"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_terminal_states_have_no_children(self):
        from queuediff.synthetic_generator import generate_hierarchy
        result = generate_hierarchy(seed=42)
        for state_name, config in result["hierarchy"].items():
            if not config["children"]:
                assert state_name.startswith("Mature"), (
                    f"Only Mature* states should be terminal, "
                    f"but {state_name} has no children"
                )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_all_states_listed(self):
        from queuediff.synthetic_generator import generate_hierarchy
        result = generate_hierarchy(seed=42)
        for state_name in result["hierarchy"]:
            assert state_name in result["states"], (
                f"{state_name} in hierarchy but not in states list"
            )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_reproducible_with_seed(self):
        from queuediff.synthetic_generator import generate_hierarchy
        a = generate_hierarchy(seed=42)
        b = generate_hierarchy(seed=42)
        assert a["states"] == b["states"]


# ---------------------------------------------------------------------------
# compute_traffic_intensity  —  analytically solve for ρ = λ / (c·μ) given
#                               the hierarchy and routing probabilities.
#                               Returns dict[str, float].
# ---------------------------------------------------------------------------


class TestComputeTrafficIntensity:
    @pytest.mark.skip(reason="awaiting implementation")
    def test_returns_dict(self):
        from queuediff.synthetic_generator import generate_hierarchy, compute_traffic_intensity
        base = generate_hierarchy(seed=42)
        rho = compute_traffic_intensity(base["hierarchy"], base["routing_probs"])
        assert isinstance(rho, dict)

    @pytest.mark.skip(reason="awaiting implementation")
    def test_all_states_have_rho(self):
        from queuediff.synthetic_generator import generate_hierarchy, compute_traffic_intensity
        base = generate_hierarchy(seed=42)
        rho = compute_traffic_intensity(base["hierarchy"], base["routing_probs"])
        for s in base["states"]:
            assert s in rho, f"Missing traffic intensity for {s}"

    @pytest.mark.skip(reason="awaiting implementation")
    def test_rho_non_negative(self):
        from queuediff.synthetic_generator import generate_hierarchy, compute_traffic_intensity
        base = generate_hierarchy(seed=42)
        rho = compute_traffic_intensity(base["hierarchy"], base["routing_probs"])
        for s, v in rho.items():
            assert v >= 0, f"Negative traffic intensity for {s}: {v}"

    @pytest.mark.skip(reason="awaiting implementation")
    def test_hsc_rho_correct(self):
        from queuediff.synthetic_generator import generate_hierarchy, compute_traffic_intensity
        base = generate_hierarchy(seed=42)
        rho = compute_traffic_intensity(base["hierarchy"], base["routing_probs"])
        hsc = base["hierarchy"]["HSC"]
        expected = hsc["arrival"] / (hsc["servers"] * hsc["service_rate"])
        assert abs(rho["HSC"] - expected) < 1e-12, (
            f"HSC ρ should be {expected}, got {rho['HSC']}"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_downstream_rho_propagates(self):
        from queuediff.synthetic_generator import generate_hierarchy, compute_traffic_intensity
        base = generate_hierarchy(seed=42)
        rho = compute_traffic_intensity(base["hierarchy"], base["routing_probs"])
        # MPP arrival should be HSC.arrival * P(HSC->MPP) = arrival
        mpp = base["hierarchy"]["MPP"]
        hsc_arrival = base["hierarchy"]["HSC"]["arrival"]
        expected_mpp_lam = hsc_arrival * base["routing_probs"][("HSC", "MPP")]
        expected_mpp_rho = expected_mpp_lam / (mpp["servers"] * mpp["service_rate"])
        assert abs(rho["MPP"] - expected_mpp_rho) < 1e-12, (
            f"MPP ρ should be {expected_mpp_rho}, got {rho['MPP']}"
        )


# ---------------------------------------------------------------------------
# introduce_bottleneck  —  reduces the service rate of *bottleneck_state*
#                          by *severity_factor*, returns modified hierarchy
#                          without mutating the original.
# ---------------------------------------------------------------------------


class TestIntroduceBottleneck:
    @pytest.mark.skip(reason="awaiting implementation")
    def test_reduces_service_rate(self):
        from queuediff.synthetic_generator import generate_hierarchy, introduce_bottleneck
        base = generate_hierarchy(seed=42)
        mod = introduce_bottleneck(base["hierarchy"], "HSC", severity_factor=5.0)
        assert mod["HSC"]["service_rate"] < base["hierarchy"]["HSC"]["service_rate"], (
            "Bottleneck should reduce service rate"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_does_not_mutate_original(self):
        from queuediff.synthetic_generator import generate_hierarchy, introduce_bottleneck
        base = generate_hierarchy(seed=42)
        orig_rate = base["hierarchy"]["HSC"]["service_rate"]
        introduce_bottleneck(base["hierarchy"], "HSC", severity_factor=5.0)
        assert base["hierarchy"]["HSC"]["service_rate"] == orig_rate, (
            "Original hierarchy should not be mutated"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_other_states_unchanged(self):
        from queuediff.synthetic_generator import generate_hierarchy, introduce_bottleneck
        base = generate_hierarchy(seed=42)
        orig_mpp_rate = base["hierarchy"]["MPP"]["service_rate"]
        mod = introduce_bottleneck(base["hierarchy"], "HSC", severity_factor=5.0)
        assert mod["MPP"]["service_rate"] == orig_mpp_rate, (
            "Only the bottlenecked state should be modified"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_severity_factor_correct(self):
        from queuediff.synthetic_generator import generate_hierarchy, introduce_bottleneck
        base = generate_hierarchy(seed=42)
        factor = 4.0
        mod = introduce_bottleneck(base["hierarchy"], "GMP", severity_factor=factor)
        expected = base["hierarchy"]["GMP"]["service_rate"] / factor
        assert abs(mod["GMP"]["service_rate"] - expected) < 1e-12, (
            f"Service rate should be original / {factor} = {expected}, "
            f"got {mod['GMP']['service_rate']}"
        )


# ---------------------------------------------------------------------------
# generate_cells  —  returns an AnnData object with synthetic cells whose
#                    ground-truth state labels, residence times, and traffic
#                    intensities are known.
# ---------------------------------------------------------------------------


class TestGenerateCells:
    @pytest.mark.skip(reason="awaiting implementation")
    def test_returns_adata(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=100, seed=42)
        import scanpy as sc
        assert isinstance(adata, sc.AnnData)

    @pytest.mark.skip(reason="awaiting implementation")
    def test_correct_number_of_cells(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        n = 5000
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=n, seed=42)
        assert adata.n_obs == n

    @pytest.mark.skip(reason="awaiting implementation")
    def test_obs_has_state_column(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=100, seed=42)
        assert "state" in adata.obs.columns

    @pytest.mark.skip(reason="awaiting implementation")
    def test_obs_has_residence_time_column(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=100, seed=42)
        assert "residence_time" in adata.obs.columns

    @pytest.mark.skip(reason="awaiting implementation")
    def test_obs_has_true_traffic_intensity_column(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=100, seed=42)
        assert "true_traffic_intensity" in adata.obs.columns

    @pytest.mark.skip(reason="awaiting implementation")
    def test_residence_times_positive(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=1000, seed=42)
        assert (adata.obs["residence_time"] > 0).all()

    @pytest.mark.skip(reason="awaiting implementation")
    def test_all_states_represented(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=10000, seed=42)
        for s in base["states"]:
            assert s in adata.obs["state"].unique(), (
                f"State {s} has zero cells"
            )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_hsc_has_largest_arrival_proportion(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=5000, seed=42)
        proportions = adata.obs["state"].value_counts(normalize=True)
        assert proportions.idxmax() == "HSC", (
            "HSC should be the most abundant state because it is "
            "the only state with external arrival"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_uns_has_true_hierarchy(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=100, seed=42)
        assert "true_hierarchy" in adata.uns
        assert "true_routing" in adata.uns
        assert "true_traffic_intensity" in adata.uns

    @pytest.mark.skip(reason="awaiting implementation")
    def test_generate_cells_with_bottleneck(self):
        from queuediff.synthetic_generator import (
            generate_hierarchy, introduce_bottleneck, generate_cells,
        )
        base = generate_hierarchy(seed=42)
        mod = introduce_bottleneck(base["hierarchy"], "HSC", severity_factor=5.0)
        adata = generate_cells(mod, base["routing_probs"], num_cells=1000, seed=42)
        assert adata.n_obs == 1000
        # The bottleneck should increase traffic intensity at HSC
        base_adata = generate_cells(
            base["hierarchy"], base["routing_probs"], num_cells=1000, seed=42
        )
        bottleneck_rho = adata.uns["true_traffic_intensity"]["HSC"]
        base_rho = base_adata.uns["true_traffic_intensity"]["HSC"]
        assert bottleneck_rho > base_rho, (
            f"Bottlenecked HSC ρ ({bottleneck_rho:.3f}) should be "
            f"higher than baseline HSC ρ ({base_rho:.3f})"
        )


# ---------------------------------------------------------------------------
# generate_sweep  —  returns a dict keyed by (bottleneck_state, severity,
#                    replicate) of AnnData objects with known ground-truth
#                    bottleneck location and severity.
# ---------------------------------------------------------------------------


class TestGenerateSweep:
    @pytest.mark.skip(reason="awaiting implementation")
    def test_returns_dict(self):
        from queuediff.synthetic_generator import generate_sweep
        sweep = generate_sweep(
            num_cells=500,
            severity_range=[1.0, 2.0],
            bottleneck_states=["HSC", "MPP"],
            n_replicates=2,
            seed=42,
        )
        assert isinstance(sweep, dict)

    @pytest.mark.skip(reason="awaiting implementation")
    def test_correct_number_of_datasets(self):
        from queuediff.synthetic_generator import generate_sweep
        states = ["HSC", "MPP", "CMP"]
        severities = [1.0, 2.0, 5.0]
        reps = 3
        sweep = generate_sweep(
            num_cells=500,
            severity_range=severities,
            bottleneck_states=states,
            n_replicates=reps,
            seed=42,
        )
        expected = len(states) * len(severities) * reps
        assert len(sweep) == expected, (
            f"Expected {expected} datasets, got {len(sweep)}"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_keys_are_tuples(self):
        from queuediff.synthetic_generator import generate_sweep
        sweep = generate_sweep(
            num_cells=100,
            severity_range=[1.0],
            bottleneck_states=["HSC"],
            n_replicates=1,
            seed=42,
        )
        key = list(sweep.keys())[0]
        assert isinstance(key, tuple)
        assert len(key) == 3  # (state, severity, replicate)

    @pytest.mark.skip(reason="awaiting implementation")
    def test_each_adata_has_ground_truth(self):
        from queuediff.synthetic_generator import generate_sweep
        sweep = generate_sweep(
            num_cells=100,
            severity_range=[1.0, 2.0],
            bottleneck_states=["HSC"],
            n_replicates=2,
            seed=42,
        )
        for (state, sev, rep), adata in sweep.items():
            assert adata.uns["true_bottleneck_state"] == state
            assert adata.uns["true_severity"] == sev
            assert adata.uns["replicate"] == rep

    @pytest.mark.skip(reason="awaiting implementation")
    def test_bottleneck_increases_rho_for_target_state(self):
        from queuediff.synthetic_generator import generate_sweep
        # Compare severity=1.0 (no bottleneck) vs severity=10.0 for MPP
        sweep = generate_sweep(
            num_cells=2000,
            severity_range=[1.0, 10.0],
            bottleneck_states=["MPP"],
            n_replicates=3,
            seed=42,
        )
        base_rhos = []
        bott_rhos = []
        for (state, sev, rep), adata in sweep.items():
            rho = adata.uns["true_traffic_intensity"]["MPP"]
            if sev == 1.0:
                base_rhos.append(rho)
            else:
                bott_rhos.append(rho)
        mean_base = np.mean(base_rhos)
        mean_bott = np.mean(bott_rhos)
        assert mean_bott > mean_base, (
            f"Mean MPP ρ with bottleneck ({mean_bott:.3f}) should exceed "
            f"baseline ρ ({mean_base:.3f})"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_replicates_use_different_seeds(self):
        from queuediff.synthetic_generator import generate_sweep
        sweep = generate_sweep(
            num_cells=500,
            severity_range=[2.0],
            bottleneck_states=["HSC"],
            n_replicates=3,
            seed=42,
        )
        n_cells_per_rep = [
            adata.obs["state"].value_counts(normalize=True).values
            for adata in sweep.values()
        ]
        # Replicates with different seeds should not produce identical
        # cell-type proportion vectors.
        for i in range(len(n_cells_per_rep) - 1):
            for j in range(i + 1, len(n_cells_per_rep)):
                if np.array_equal(n_cells_per_rep[i], n_cells_per_rep[j]):
                    msg = (
                        f"Replicates {i} and {j} produced identical "
                        "proportions — seeds may not vary"
                    )
                    pytest.fail(msg)


# ---------------------------------------------------------------------------
# End-to-end recovery test  —  verify that the pipeline can detect the
#                               injected bottleneck.
#
# This test simulates the full inference loop (fit → queueing → rank)
# with a strong bottleneck and checks that the true bottleneck state is
# ranked first.
# ---------------------------------------------------------------------------


class TestEndToEndRecovery:
    @pytest.mark.skip(reason="awaiting implementation")
    def test_detects_strong_bottleneck(self):
        from queuediff.synthetic_generator import generate_hierarchy, introduce_bottleneck, generate_cells
        from queuediff.distribution_fitting import fit_distributions_to_state
        from queuediff.queueing_network import build_from_data
        from queuediff.bottleneck_diagnostics import rank_bottlenecks

        base = generate_hierarchy(seed=42)
        mod = introduce_bottleneck(base["hierarchy"], "CMP", severity_factor=10.0)
        adata = generate_cells(mod, base["routing_probs"], num_cells=5000, seed=42)

        est_rates = {}
        for s in adata.obs["state"].unique():
            mask = adata.obs["state"] == s
            times = adata.obs.loc[mask, "residence_time"].values
            fit = fit_distributions_to_state(times)
            est_rates[s] = fit["exp_rate"]

        qn = build_from_data(est_rates, base["routing_probs"], name="recovery_test")
        summary = qn.summary()
        ranked = rank_bottlenecks(summary)

        inferred_bottleneck = ranked.iloc[0]["state"]
        assert inferred_bottleneck == "CMP", (
            f"Expected CMP as top bottleneck, got {inferred_bottleneck}. "
            f"Ranking: {ranked['state'].tolist()}"
        )

    @pytest.mark.skip(reason="awaiting implementation")
    def test_no_bottleneck_when_severity_is_one(self):
        from queuediff.synthetic_generator import generate_hierarchy, generate_cells
        from queuediff.distribution_fitting import fit_distributions_to_state
        from queuediff.queueing_network import build_from_data
        from queuediff.bottleneck_diagnostics import rank_bottlenecks

        base = generate_hierarchy(seed=42)
        adata = generate_cells(base["hierarchy"], base["routing_probs"],
                               num_cells=5000, seed=42)

        est_rates = {}
        for s in adata.obs["state"].unique():
            mask = adata.obs["state"] == s
            times = adata.obs.loc[mask, "residence_time"].values
            fit = fit_distributions_to_state(times)
            est_rates[s] = fit["exp_rate"]

        qn = build_from_data(est_rates, base["routing_probs"], name="no_bottleneck")
        summary = qn.summary()
        ranked = rank_bottlenecks(summary)

        # With no bottleneck, the state with highest ρ is the one with
        # lowest service rate / highest arrival — the exact identity is
        # design-dependent, but it should be stable across replicates.
        top_state = ranked.iloc[0]["state"]
        top_rho = ranked.iloc[0]["traffic_intensity"]
        assert top_rho < 1.0, (
            f"Without bottleneck, ρ should be < 1 for all states. "
            f"Top state {top_state} has ρ={top_rho:.3f}"
        )
