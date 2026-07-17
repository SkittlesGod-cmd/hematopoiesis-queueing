"""Resolution sensitivity analysis for Leiden clustering vs marker-gene states.

Tests whether concordance between marker-gene-based state assignment and
Leiden clustering is stable across different Leiden resolution values.
Addresses the concern that arbitrary clustering-resolution choices
could produce arbitrary state boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import List, Dict, Optional
from anndata import AnnData

from queuediff.structural_crosscheck import (
    map_marker_to_meta,
    map_leiden_to_meta,
    compute_tier_concordance,
    MARKER_TO_META,
    META_TIERS,
)


def run_resolution_sweep(
    adata: sc.AnnData,
    resolutions: List[float],
    marker_state_col: str = "marker_state",
    leiden_base_key: str = "leiden_cluster",
    neighbors_key: Optional[str] = None,
) -> pd.DataFrame:
    """Run Leiden clustering at multiple resolutions and compute concordance.

    Reuses the existing neighbors graph in ``adata`` — only the resolution
    parameter is varied. This isolates the effect of resolution specifically.

    Parameters
    ----------
    adata
        AnnData with PCA computed, neighbors graph already built,
        and ``marker_state_col`` in ``.obs``.
    resolutions
        List of Leiden resolution values to test (e.g., [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]).
    marker_state_col
        Column in ``adata.obs`` containing marker-based state assignments.
    leiden_base_key
        Base name for Leiden cluster columns (resolution will be appended).
    neighbors_key
        Key in ``adata.uns`` for the neighbors results. If None, uses default.

    Returns
    -------
    pandas.DataFrame
        One row per resolution with columns:
        - resolution: the tested resolution value
        - n_clusters: number of Leiden clusters found
        - concordance_score: overall concordance between marker and Leiden tiers
    """
    if marker_state_col not in adata.obs.columns:
        raise ValueError(f"'{marker_state_col}' not found in adata.obs")

    # Ensure neighbors graph exists
    if "neighbors" not in adata.uns and neighbors_key is None:
        raise ValueError("Neighbors graph not found in adata.uns['neighbors']. "
                         "Run sc.pp.neighbors first or pass neighbors_key.")

    results = []

    for res in resolutions:
        key = f"{leiden_base_key}_res{str(res).replace('.', '_')}"

        # Run Leiden on the EXISTING neighbors graph (reuse graph, only vary resolution)
        sc.tl.leiden(adata, resolution=res, neighbors_key=neighbors_key, key_added=key)
        adata.obs[key] = adata.obs[key].astype("category")

        # Compute concordance using structural_crosscheck logic
        adata = map_marker_to_meta(adata, MARKER_TO_META, marker_state_col)
        adata = map_leiden_to_meta(
            adata,
            marker_tier_key="marker_tier",
            leiden_key=key,
            tier_key="leiden_tier",
        )
        concordance = compute_tier_concordance(adata, "marker_tier", "leiden_tier")

        # Overall concordance
        total_cells = concordance["n_cells"].sum()
        total_ok = concordance["n_consistent"].sum()
        overall = total_ok / total_cells if total_cells > 0 else np.nan

        results.append({
            "resolution": res,
            "n_clusters": int(adata.obs[key].nunique()),
            "concordance_score": float(overall),
        })

    return pd.DataFrame(results)


def summarize_stability(sweep_df: pd.DataFrame, stability_threshold: float = 0.15) -> Dict:
    """Summarize concordance stability across resolutions.

    Parameters
    ----------
    sweep_df
        DataFrame returned by :py:func:`run_resolution_sweep` with columns
        'resolution', 'n_clusters', 'concordance_score'.
    stability_threshold
        Maximum allowed range (max - min) of concordance scores for the
        result to be considered "stable". Default 0.15 (15 percentage points).
        **This is a stated, adjustable choice — not a hard scientific fact.**

    Returns
    -------
    dict
        Keys: 'concordance_mean', 'concordance_std', 'concordance_range',
        'is_stable' (bool), 'stability_threshold_used'.
    """
    if "concordance_score" not in sweep_df.columns:
        raise ValueError("sweep_df must have 'concordance_score' column")

    scores = sweep_df["concordance_score"].dropna()

    if len(scores) == 0:
        return {
            "concordance_mean": np.nan,
            "concordance_std": np.nan,
            "concordance_range": np.nan,
            "is_stable": False,
            "stability_threshold_used": stability_threshold,
        }

    c_range = float(scores.max() - scores.min())
    is_stable = c_range < stability_threshold

    return {
        "concordance_mean": float(scores.mean()),
        "concordance_std": float(scores.std()),
        "concordance_range": c_range,
        "is_stable": is_stable,
        "stability_threshold_used": stability_threshold,
    }


if __name__ == "__main__":
    """Sanity check: run resolution sweep on subsampled Weinreb data."""
    print("=" * 70)
    print("RESOLUTION SENSITIVITY SANITY CHECK (Subsampled)")
    print("=" * 70)
    print("\nNOTE: This uses a random subsample of 15,000 cells for speed.")
    print("The full dataset (130,887 cells) should be used for final results.\n")

    # Reuse loading/preprocessing from state_discretization
    from queuediff.state_discretization import (
        preprocess_standard,
        score_marker_states,
        assign_marker_states,
        MARKER_GENES,
    )
    from scipy.io import mmread
    import gzip

    # Data paths (reuse what exists in scripts/data/)
    base_path = Path("scripts/data/raw/weinreb")
    counts_path = base_path / "stateFate_inVitro_normed_counts.mtx.gz"
    genes_path = base_path / "stateFate_inVitro_gene_names.txt.gz"
    meta_path = base_path / "stateFate_inVitro_metadata.txt.gz"

    for p in [counts_path, genes_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}")

    print(f"Loading Weinreb data from {base_path}...")

    # Load normalized counts matrix (appears to be cells × genes already)
    X = mmread(gzip.open(counts_path, "rt")).tocsr()  # NO transpose - already cells × genes
    with gzip.open(genes_path, "rt") as f:
        gene_names = [line.strip() for line in f]
    meta = pd.read_csv(gzip.open(meta_path, "rt"), sep="\t", index_col=0)

    # Build AnnData: X is (n_cells, n_genes), var_names are genes
    adata = sc.AnnData(X=X.astype(np.float32))
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs = meta
    adata.obs_names_make_unique()
    print(f"  Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    # SUBSAMPLE FIRST (before expensive preprocessing)
    SUBSAMPLE_N = 15000
    rng = np.random.default_rng(42)
    if adata.n_obs > SUBSAMPLE_N:
        idx = rng.choice(adata.n_obs, SUBSAMPLE_N, replace=False)
        adata = adata[idx].copy()
        print(f"  Subsampled to {adata.n_obs} cells (seed=42)")

    # Preprocess
    print("Running standard preprocessing (filter, normalize, log1p, HVG, scale, PCA)...")
    adata = preprocess_standard(adata, min_genes=200, min_cells=3, n_top_genes=2000, n_pcs=30)
    print(f"  After filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    # Marker state assignment
    print("Scoring marker genes and assigning states...")
    adata = assign_marker_states(adata, MARKER_GENES)
    print(f"  States assigned: {adata.obs['marker_state'].value_counts().to_dict()}")

    # Build neighbors graph ONCE (reused across all resolutions)
    print("Building kNN graph (n_neighbors=30)...")
    sc.pp.neighbors(adata, n_neighbors=30)

    # Resolution sweep
    resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    print(f"\nRunning Leiden at resolutions: {resolutions}")
    sweep_df = run_resolution_sweep(adata, resolutions)

    # Print sweep table
    print("\n" + "=" * 70)
    print("RESOLUTION SWEEP RESULTS")
    print("=" * 70)
    print(sweep_df.to_string(index=False))

    # Stability summary
    print("\n" + "=" * 70)
    print("STABILITY SUMMARY")
    print("=" * 70)
    summary = summarize_stability(sweep_df)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if summary['is_stable'] else 'FAIL'} "
          f"(range {summary['concordance_range']:.3f} "
          f"{'< 0.15' if summary['is_stable'] else '>= 0.15'})")
    print("=" * 70)