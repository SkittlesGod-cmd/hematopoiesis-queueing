"""
Nestorowa et al. 2016 — structural cross-check only.

Per the research plan, this dataset has only a single timepoint, so no
service-rate estimation or queueing analysis is possible.  The script:

  1. Loads and preprocesses the Nestorowa data.
  2. Assigns marker-based states and runs Leiden clustering.
  3. Maps native states to the shared meta-hierarchy tiers.
  4. Computes concordance between marker-based and Leiden-based tier labels.
  5. Saves the concordance table and a JSON summary.

Save paths are set to ``results/nestorowa_*.csv`` / ``*.json``.

Usage::

    python scripts/run_pipeline_nestorowa.py
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/raw/nestorowa")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw Nestorowa data into AnnData
    #
    #    Inputs (from download_nestorowa.py):
    #       data/raw/nestorowa/coordinates_gene_counts_flow_cytometry.txt.gz
    #       data/raw/nestorowa/all_cell_types.txt
    #
    #    Returns:
    #       adata  : AnnData
    #                .X             = normalised expression (cells × genes)
    #                .var_names     = Ensembl gene IDs
    #                .obs           = cell-level metadata
    #                .obsm['flow_cytometry'] = protein marker panel
    #                .obsm['diffmap']        = diffusion map coordinates
    #
    #    TODO: implemented in src/queuediff/data_loading.py
    # ------------------------------------------------------------------
    print("[1/4] Loading Nestorowa raw data …")
    try:
        from queuediff.data_loading import load_nestorowa

        adata = load_nestorowa(DATA_DIR)
    except Exception as exc:
        print(f"  FAILED to load Nestorowa data: {exc}", file=sys.stderr)
        print(f"  Ensure files exist under {DATA_DIR} (run scripts/download_nestorowa.py first)")
        sys.exit(1)

    print(f"  Cells × Genes  : {adata.n_obs} × {adata.n_vars}")

    # ------------------------------------------------------------------
    # 2. State discretisation — marker-gene scoring + Leiden cross-check
    #
    #    Input:
    #       adata  : AnnData (expression counts in .X).
    #
    #    Adds to adata:
    #       adata.obs['marker_state']    : categorical
    #       adata.obs['leiden_cluster']  : categorical
    #       adata.uns['state_contingency'] : cross-tabulation
    #
    #    NOTE: Nestorowa data is already normalised.  The run() function
    #          re-runs the standard preprocessing pipeline (filter, log1p,
    #          HVG, PCA, Leiden), which is appropriate because the raw
    #          matrix in the downloaded file is normalised but not filtered.
    #
    #    TODO: implemented in src/queuediff/state_discretization.py
    #          (already written — loads, filters, normalises, log1p, HVG,
    #           PCA, clusters via Leiden, and assigns marker states)
    # ------------------------------------------------------------------
    print("[2/4] Assigning states …")
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
    # 3. Structural cross-check — map states to meta-hierarchy tiers and
    #    compute concordance between marker-based and Leiden-based labels.
    #
    #    Input:
    #       adata        : AnnData with .obs['marker_state'] and .obs['leiden_cluster'].
    #       dataset_label: str  (used in the printed report header).
    #
    #    Returns:
    #       result  : dict with keys:
    #                    concordance   : pd.DataFrame  (per-tier metrics)
    #                    n_cells       : int
    #                    overall_rate  : float  (fraction of cells with consistent tier labels)
    #
    #    Side-effect: prints a formatted report to stdout.
    #
    #    TODO: implemented in src/queuediff/structural_crosscheck.py
    #          (already written — maps, compares, prints report)
    # ------------------------------------------------------------------
    print("[3/4] Running structural cross-check …")
    try:
        from queuediff.structural_crosscheck import run as run_crosscheck

        result = run_crosscheck(
            adata,
            dataset_label="Nestorowa 2016",
            marker_state_key="marker_state",
            leiden_key="leiden_cluster",
        )
    except Exception as exc:
        print(f"  FAILED structural cross-check: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Overall concordance : {result['overall_rate']:.3f}  "
          f"({result['n_cells']} cells)")

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    print("[4/4] Saving results …")

    concordance = result["concordance"]
    concordance.to_csv(RESULTS_DIR / "nestorowa_tier_concordance.csv")
    print(f"  Saved results/nestorowa_tier_concordance.csv")

    with open(RESULTS_DIR / "nestorowa_structural_summary.json", "w") as f:
        json.dump(
            {
                "overall_rate": result["overall_rate"],
                "n_cells": result["n_cells"],
                "n_tiers_with_cells": int(concordance["n_cells"].notna().sum()),
            },
            f,
            indent=2,
        )
    print(f"  Saved results/nestorowa_structural_summary.json")

    # Export the contingency table between marker tiers and Leiden clusters
    # for downstream figure generation or inspection.
    from pandas import crosstab

    tier_vs_cluster = crosstab(
        adata.obs["marker_tier"],
        adata.obs["leiden_cluster"],
    )
    tier_vs_cluster.to_csv(RESULTS_DIR / "nestorowa_tier_vs_leiden.csv")
    print(f"  Saved results/nestorowa_tier_vs_leiden.csv")

    print()
    print("=" * 60)
    print("  Nestorowa structural cross-check complete — results in results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
