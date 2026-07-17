"""
Weinreb et al. 2020 — full quantitative pipeline.

Loads the primary dataset, assigns states, estimates residence times via
both clonal (barcode) and flux-based methods, fits service-time distributions,
compares semi-Markov (gamma) vs. Markov (exponential) models, builds the
queueing network, flags bottlenecks, and validates congestion at the monocyte
branch point.

Save paths are set to ``results/weinreb_*.csv``.

Usage::

    python scripts/run_pipeline_weinreb.py
"""

from __future__ import annotations

from pathlib import Path

import sys

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/raw/weinreb")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw data into an AnnData object
    #
    #    Inputs (from download_weinreb.py):
    #       data/raw/weinreb/stateFate_inVitro_normed_counts.mtx.gz
    #       data/raw/weinreb/stateFate_inVitro_gene_names.txt.gz
    #       data/raw/weinreb/stateFate_inVitro_metadata.txt.gz
    #
    #    Returns:
    #       adata  : AnnData, shape (n_cells, n_genes)
    #                .X       = sparse count matrix  (cells × genes)
    #                .var_names = gene symbols
    #                .obs     = cell-level metadata (timepoint, clone info, …)
    #                .obsm['clone_matrix'] = sparse barcode matrix  (cells × clones)
    #
    #    TODO: implemented in src/queuediff/data_loading.py
    # ------------------------------------------------------------------
    print("[1/9] Loading Weinreb raw data …")
    try:
        from queuediff.data_loading import load_weinreb

        adata = load_weinreb(DATA_DIR)
    except Exception as exc:
        print(f"  FAILED to load data: {exc}", file=sys.stderr)
        print(f"  Ensure files exist under {DATA_DIR} (run scripts/download_weinreb.py first)")
        sys.exit(1)

    print(f"  Cells × Genes  : {adata.n_obs} × {adata.n_vars}")

    # ------------------------------------------------------------------
    # 2. State discretization — marker-gene scoring + Leiden cross-check
    #
    #    Input:
    #       adata        : AnnData (raw counts in .X)
    #
    #    Adds to adata:
    #       adata.obs['marker_state']    : categorical, one of HSC/MPP/LMPP/CMP/MEP/GMP
    #       adata.obs['leiden_cluster']  : categorical, Leiden cluster label
    #       adata.uns['state_contingency'] : pd.crosstab(marker_state, leiden_cluster)
    #
    #    TODO: implemented in src/queuediff/state_discretization.py
    #          (already written — loads, filters, normalises, log1p, HVG, PCA,
    #           clusters via Leiden, and assigns marker states)
    # ------------------------------------------------------------------
    print("[2/9] Assigning states …")
    try:
        from queuediff.state_discretization import run as run_state_discretization

        adata = run_state_discretization(
            adata,
            min_genes=200,
            min_cells=3,
            n_top_genes=2000,
            n_pcs=30,
            leiden_resolution=1.0,
        )
    except Exception as exc:
        print(f"  FAILED state discretisation: {exc}", file=sys.stderr)
        sys.exit(1)

    n_states = adata.obs["marker_state"].nunique()
    n_clusters = adata.obs["leiden_cluster"].nunique()
    print(f"  Marker states   : {n_states}")
    print(f"  Leiden clusters : {n_clusters}")

    # ------------------------------------------------------------------
    # 3. Clonal residence-time estimation (direct, barcode-based)
    #
    #    Input:
    #       adata    : AnnData with .obs['marker_state'], .obs['timepoint'],
    #                  .obsm['clone_matrix'] (sparse csr, cells × clones)
    #
    #    Returns:
    #       res_df   : pd.DataFrame, columns:
    #                    state             : str
    #                    clone_idx         : int
    #                    first_time        : float
    #                    last_time         : float
    #                    residence_time    : float   (last_time - first_time)
    #
    #    TODO: implemented in src/queuediff/clonal_residence_time.py
    # ------------------------------------------------------------------
    print("[3/9] Estimating clonal residence times …")
    try:
        from queuediff.clonal_residence_time import estimate_residence_times

        clonal_res_df = estimate_residence_times(
            adata,
            state_col="marker_state",
            time_col="timepoint",
        )
    except Exception as exc:
        print(f"  FAILED clonal residence time estimation: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  States with barcode data : {len(clonal_res_df)}")
    clonal_res_df.to_csv(RESULTS_DIR / "weinreb_clonal_residence_times.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Flux-based residence-time estimation (population-flux fallback)
    #
    #    Input:
    #       adata        : AnnData with .obs['marker_state'], .obs['timepoint']
    #       upstream_map : dict mapping each state → its immediate upstream
    #                      source state (so flux can be attributed).
    #
    #    Returns:
    #       flux_df      : pd.DataFrame, columns:
    #                        state          : str
    #                        mean_N         : float  (mean cell count)
    #                        mean_flux      : float
    #                        service_rate   : float  (μ = flux / N)
    #
    #    TODO: implemented in src/queuediff/flux_residence_time.py
    # ------------------------------------------------------------------
    print("[4/9] Estimating flux-based residence times …")
    try:
        from queuediff.flux_residence_time import estimate_service_rates

        # Hierarchical structure for the Weinreb dataset:
        #   HSC → MPP → {CMP, LMPP} → {MEP, GMP} → Mature
        flux_upstream = {
            "CMP": "MPP",
            "LMPP": "MPP",
            "MEP": "CMP",
            "GMP": "CMP",
        }
        flux_res_df = estimate_service_rates(
            adata,
            state_col="marker_state",
            time_col="timepoint",
            upstream_map=flux_upstream,
        )
    except Exception as exc:
        print(f"  FAILED flux-based estimation: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  States with flux estimates : {len(flux_res_df)}")
    flux_res_df.to_csv(RESULTS_DIR / "weinreb_flux_service_rates.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Distribution fitting — gamma vs. exponential per state
    #
    #    Input:
    #       residence_df : pd.DataFrame with columns ['state', 'residence_time']
    #                      (e.g., from step 3 or 4).  For Weinreb, use clonal.
    #
    #    Returns:
    #       fit_results  : pd.DataFrame, one row per state, columns:
    #                        state          : str
    #                        n_obs          : int
    #                        gamma_shape    : float  (MLE)
    #                        gamma_scale    : float
    #                        gamma_loglik   : float
    #                        exp_rate       : float  (1/scale, MLE)
    #                        exp_loglik     : float
    #                        gamma_aic      : float
    #                        gamma_bic      : float
    #                        exp_aic        : float
    #                        exp_bic        : float
    #
    #    TODO: implemented in src/queuediff/distribution_fitting.py
    # ------------------------------------------------------------------
    print("[5/9] Fitting service-time distributions …")
    try:
        from queuediff.distribution_fitting import fit_all_states

        fit_results = fit_all_states(
            clonal_res_df,
            state_col="state",
            time_col="residence_time",
        )
    except Exception as exc:
        print(f"  FAILED distribution fitting: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  States fitted : {len(fit_results)}")
    fit_results.to_csv(RESULTS_DIR / "weinreb_distribution_fits.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Model comparison — AIC / BIC + likelihood-ratio test + BH correction
    #
    #    Input:
    #       fit_results  : pd.DataFrame (output of step 5).
    #
    #    Returns:
    #       comparison   : pd.DataFrame, one row per state, columns:
    #                        state              : str
    #                        delta_aic          : float (gamma − exp;  <0 → gamma preferred)
    #                        delta_bic          : float
    #                        lr_pvalue          : float  (likelihood-ratio test)
    #                        q_value_bh         : float  (Benjamini-Hochberg FDR)
    #                        rejected_at_alpha  : bool
    #                        preferred_model    : str   ("gamma (semi-Markov)" or "exponential (Markov)")
    #
    #    TODO: implemented in src/queuediff/model_comparison.py
    # ------------------------------------------------------------------
    print("[6/9] Comparing models …")
    try:
        from queuediff.model_comparison import compare_models

        comparison = compare_models(fit_results, alpha=0.05)
    except Exception as exc:
        print(f"  FAILED model comparison: {exc}", file=sys.stderr)
        sys.exit(1)

    n_gamma = (comparison["preferred_model"] == "gamma (semi-Markov)").sum()
    n_exp = (comparison["preferred_model"] == "exponential (Markov)").sum()
    print(f"  Gamma preferred  : {n_gamma} states")
    print(f"  Exp preferred    : {n_exp} states")
    comparison.to_csv(RESULTS_DIR / "weinreb_model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Queueing network — build graph, propagate arrival rates, compute ρ
    #
    #    Inputs:
    #       service_rates : dict[str, float] — μ for each state
    #       routing_probs : dict[tuple[str, str], float] — P(transition → target | source)
    #
    #    Returns:
    #       qn  : QueueingNetwork object (networkx.DiGraph in .graph)
    #       summary : pd.DataFrame, columns:
    #                    state             : str
    #                    service_rate      : float
    #                    servers           : int   (default 1)
    #                    arrival_rate      : float  (λ, propagated downstream)
    #                    traffic_intensity  : float  (ρ = λ / (c·μ))
    #
    #    TODO: implemented in src/queuediff/queueing_network.py
    # ------------------------------------------------------------------
    print("[7/9] Building queueing network …")
    try:
        from queuediff.queueing_network import build_from_data

        service_rates = dict(zip(fit_results["state"], fit_results["exp_rate"]))
        routing_probs = {
            ("HSC", "MPP"): 1.0,
            ("MPP", "CMP"): 0.5,
            ("MPP", "LMPP"): 0.5,
            ("CMP", "MEP"): 0.5,
            ("CMP", "GMP"): 0.5,
        }
        qn = build_from_data(service_rates, routing_probs, name="Weinreb")
        summary = qn.summary()
    except Exception as exc:
        print(f"  FAILED to build queueing network: {exc}", file=sys.stderr)
        sys.exit(1)

    summary.to_csv(RESULTS_DIR / "weinreb_queueing_summary.csv", index=False)
    print(f"  States in network : {len(summary)}")

    # ------------------------------------------------------------------
    # 8. Bottleneck diagnostics — flag and rank states by traffic intensity
    #
    #    Input:
    #       summary         : pd.DataFrame (output of step 7).
    #       rho_threshold   : float  (default 0.8; ρ > threshold → bottleneck)
    #
    #    Returns:
    #       flagged         : pd.DataFrame, same as summary plus:
    #                            is_bottleneck  : bool
    #                            severity       : category (low/moderate/high/critical/overloaded)
    #       stats           : dict with keys:
    #                            n_bottlenecks, n_states, max_rho, worst_state, …
    #
    #    TODO: implemented in src/queuediff/bottleneck_diagnostics.py
    # ------------------------------------------------------------------
    print("[8/9] Computing bottleneck diagnostics …")
    try:
        from queuediff.bottleneck_diagnostics import (
            flag_bottlenecks,
            summarize_network_bottlenecks,
        )

        flagged = flag_bottlenecks(summary, rho_threshold=0.8)
        stats = summarize_network_bottlenecks(flagged)
    except Exception as exc:
        print(f"  FAILED bottleneck diagnostics: {exc}", file=sys.stderr)
        sys.exit(1)

    flagged.to_csv(RESULTS_DIR / "weinreb_bottlenecks.csv", index=False)
    print(f"  Bottlenecks (ρ > 0.8) : {stats['n_bottlenecks']}")
    print(f"  Worst state           : {stats['worst_bottleneck_state']}  "
          f"(ρ={stats['max_traffic_intensity']:.3f})")

    # ------------------------------------------------------------------
    # 9. Branch-point validation — test for elevated congestion at the
    #    monocyte branch point (CMP → MEP vs. CMP → GMP).
    #
    #    Input:
    #       summary         : pd.DataFrame with .traffic_intensity and .state.
    #       branch_states   : list[str] — the two states that diverge at the
    #                          branch point (e.g., ['MEP', 'GMP']).
    #
    #    Returns:
    #       bp_result       : dict with keys:
    #                            branch_mean_rho       : float
    #                            non_branch_mean_rho   : float
    #                            statistic             : float  (test statistic)
    #                            pvalue                : float
    #                            test                  : str
    #
    #    TODO: implemented in src/queuediff/branch_point_validation.py
    # ------------------------------------------------------------------
    print("[9/9] Validating branch-point congestion …")
    try:
        from queuediff.branch_point_validation import branch_point_analysis

        bp_result = branch_point_analysis(
            summary,
            branch_states=["MEP", "GMP"],
        )
    except Exception as exc:
        print(f"  FAILED branch-point validation: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Branch-point ρ (MEP/GMP)       : {bp_result['branch_mean_rho']:.3f}")
    print(f"  Non-branch mean ρ              : {bp_result['non_branch_mean_rho']:.3f}")
    print(f"  Mann-Whitney U p-value         : {bp_result['pvalue']:.4f}")

    import json

    with open(RESULTS_DIR / "weinreb_branch_point_validation.json", "w") as f:
        json.dump(bp_result, f, indent=2)

    print()
    print("=" * 60)
    print("  Weinreb pipeline complete — results in results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
