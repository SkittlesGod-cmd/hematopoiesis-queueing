from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread

from .data_loading import load_weinreb_from_files as load_weinreb, load_from_mtx


# --- Known hematopoietic marker genes ---
#
# Sources:
#   Laurenti & Göttgens (2020) Nature Reviews Mol. Cell. Biol. 21:502–520
#   Weinreb et al. (2020) Science 367:eaaw3381
#   Nestorowa et al. (2016) Blood 128:e20–e31
#
MARKER_GENES: dict[str, list[str]] = {
    "HSC": [
        "Meis1", "Hlf", "Procr", "Mllt3", "Pbx1",   # Laurenti & Göttgens, Table 1
        "Hoxb5", "Gata2",
    ],
    "MPP": [
        "Cd34", "Kit", "Flt3",                         # Weinreb et al. Fig 2 markers
    ],
    "LMPP": [
        "Flt3", "Il7r", "Dntt", "Fcer1g", "Cd2",      # Nestorowa et al. supplement
    ],
    "CMP": [
        "Csf1r", "Mpo", "Cebpa", "Cebpb",              # Laurenti & Göttgens, Fig 3
        "Cebpe", "Csf2rb",                             # Laurenti & Göttgens, Fig 3 (early myeloid commitment)
    ],
    "MEP": [
        "Gata1", "Klf1", "EpoR", "Tal1", "Gypa",       # Laurenti & Göttgens, Fig 3
        "Itga2b", "Vwf",                               # Laurenti & Göttgens, Fig 3 (megakaryocyte-erythroid)
    ],
    "GMP": [
        "Elane", "Mpo", "Ctsg", "Prtn3", "Csf3r",      # Weinreb et al. Fig 2 markers
    ],
}


def preprocess_standard(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    already_normalized: bool = False,
    already_log_transformed: bool = False,
) -> sc.AnnData:
    """Run standard single-cell preprocessing.

    The input data can be in one of three states:
    - Raw counts: already_normalized=False, already_log_transformed=False
    - Library-size-normalized (CPM/TPM): already_normalized=True, already_log_transformed=False
    - Library-size-normalized AND log-transformed: already_normalized=True, already_log_transformed=True

    Steps:
        1. Filter cells with fewer than *min_genes* detected.
        2. Filter genes detected in fewer than *min_cells*.
        3. If already_normalized=False: total-count normalize to 10,000 per cell.
        4. If already_log_transformed=False: Log1p transform.
        5. Store a copy of the log-normalised FULL gene expression matrix
           in adata.layers['lognorm_full'] BEFORE HVG selection, so that
           cell cycle, apoptosis, and other marker genes not in the HVG
           subset remain available for scoring.
        6. Compute cell cycle scores (S-phase + G2M-phase) on the FULL
           log-normalized data BEFORE HVG selection, so that cell cycle
           marker genes are not filtered out.
        7. Select *n_top_genes* highly variable genes using flavor='seurat'
           on the log-normalised data.
        8. Subset to HVGs.
        9. Store a copy of the log-normalised HVG matrix in
           adata.layers['lognorm'] for marker-gene scoring.
        10. Convert to dense array (HVG subset is small: ~2000 genes).
        11. Scale to unit variance (clip at 10).
        12. Run PCA (*n_pcs* components).

    The scaled/clipped adata.X is used only for PCA and downstream
    Leiden clustering. Marker-gene scoring (sc.tl.score_genes) for HVG
    genes should use the log-normalised values in adata.layers['lognorm'].
    Cell-cycle and apoptosis scoring should use adata.layers['lognorm_full']
    which contains all genes.

    Parameters
    ----------
    adata
        AnnData with count data (raw or library-size-normalized).
    min_genes
        Minimum number of genes that must be detected in a cell.
    min_cells
        Minimum number of cells a gene must be detected in.
    n_top_genes
        Number of highly variable genes to select.
    n_pcs
        Number of PCA components to compute.
    already_normalized
        If True, skip sc.pp.normalize_total (data is library-size-normalized).
        If False (default), run normalize_total (data is raw counts).
    already_log_transformed
        If True, skip sc.pp.log1p (data is already log-transformed).
        If False (default), run log1p (data is not log-transformed).

    Returns
    -------
    The same AnnData object, modified in place.
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if not already_normalized:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if not already_log_transformed:
        sc.pp.log1p(adata)

    # Store log-normalised FULL gene expression matrix BEFORE HVG selection
    # so that cell cycle, apoptosis, and other marker genes not in the HVG
    # subset remain available for scoring.
    if hasattr(adata.X, "toarray"):
        adata.layers["lognorm_full"] = adata.X.toarray()
    else:
        adata.layers["lognorm_full"] = adata.X.copy()

    # Compute cell cycle scores on FULL log-normalized data BEFORE HVG selection
    # so that cell cycle marker genes are not filtered out.
    s_genes = ['Pcna', 'Mcm2']
    g2m_genes = ['Top2a', 'Ccnb1']
    present_s = [g for g in s_genes if g in adata.var_names]
    present_g2m = [g for g in g2m_genes if g in adata.var_names]
    if present_s:
        sc.tl.score_genes(adata, gene_list=present_s, score_name='_cycle_score_S', random_state=0)
    else:
        adata.obs['_cycle_score_S'] = 0.0
    if present_g2m:
        sc.tl.score_genes(adata, gene_list=present_g2m, score_name='_cycle_score_G2M', random_state=0)
    else:
        adata.obs['_cycle_score_G2M'] = 0.0
    adata.obs['cycling_score'] = adata.obs['_cycle_score_S'] + adata.obs['_cycle_score_G2M']
    # Clean up temporary columns
    del adata.obs['_cycle_score_S']
    del adata.obs['_cycle_score_G2M']

    # HVG selection on log-normalised data (flavor='seurat')
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()
    # Store log-normalised HVG matrix for marker-gene scoring
    if hasattr(adata.X, "toarray"):
        adata.layers["lognorm"] = adata.X.toarray()
    else:
        adata.layers["lognorm"] = adata.X.copy()
    # Dense conversion for HVG subset (~2000 genes) before scaling
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    return adata


def cluster_leiden(
    adata: sc.AnnData,
    n_neighbors: int = 30,
    resolution: float = 1.0,
    neighbors_key: str | None = None,
    key_added: str = "leiden_cluster",
) -> sc.AnnData:
    """Run Leiden clustering on PCA-reduced data.

    Requires that PCA has been computed and stored in ``adata.obsm['X_pca']``.

    Parameters
    ----------
    adata
        Preprocessed AnnData with PCA.
    n_neighbors
        Number of neighbours for the kNN graph.
    resolution
        Leiden resolution parameter (higher → more clusters).
    neighbors_key
        Optional key for the neighbours results in ``adata.uns``.
    key_added
        Column name in ``adata.obs`` for the cluster labels.

    Returns
    -------
    The same AnnData with ``adata.obs[key_added]`` added.
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, key_added=neighbors_key)
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                 flavor="igraph", n_iterations=2, directed=False)
    adata.obs[key_added] = adata.obs[key_added].astype("category")
    return adata


def score_marker_states(
    adata: sc.AnnData,
    marker_genes: dict[str, list[str]] | None = None,
    score_prefix: str = "score_",
    layer: str = "lognorm",
) -> pd.DataFrame:
    """Score every cell against every marker-gene state panel.

    Uses ``sc.tl.score_genes`` internally for each state.  The score is
    the average normalised expression of the gene set minus the average
    expression of a randomly sampled reference set of the same size.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in the specified layer.
    marker_genes
        Dict mapping state name → list of marker gene symbols.
        Defaults to :py:data:`MARKER_GENES`.
    score_prefix
        Prefix for the temporary score column inserted into ``adata.obs``.
    layer
        Layer in ``adata.layers`` containing log-normalised expression
        (e.g., "lognorm"). Must exist; raises ValueError if missing.

    Returns
    -------
    DataFrame (cells × states) of gene-set scores.
    """
    if marker_genes is None:
        marker_genes = MARKER_GENES

    # Validate that the requested layer exists
    if layer not in adata.layers:
        raise ValueError(
            f"Layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}. "
            f"Run preprocess_standard first to create the 'lognorm' layer."
        )

    scores = {}
    for state, genes in marker_genes.items():
        col = score_prefix + state
        present = [g for g in genes if g in adata.var_names]
        if not present:
            scores[state] = np.zeros(adata.n_obs)
            continue
        # Use the specified layer directly via scanpy's layer parameter
        sc.tl.score_genes(adata, gene_list=present, score_name=col,
                          random_state=0, layer=layer)
        scores[state] = adata.obs[col].values.copy()
        del adata.obs[col]

    return pd.DataFrame(scores, index=adata.obs_names)


def assign_marker_states(
    adata: sc.AnnData,
    marker_genes: dict[str, list[str]] | None = None,
    key_added: str = "marker_state",
    layer: str = "lognorm",
) -> sc.AnnData:
    """Assign each cell to its highest-scoring marker-defined state.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in the specified layer.
    marker_genes
        Dict mapping state name → marker gene list.
        Defaults to :py:data:`MARKER_GENES`.
    key_added
        Column name in ``adata.obs`` for the assigned state.
    layer
        Layer in ``adata.layers`` containing log-normalised expression.
        Defaults to "lognorm".

    Returns
    -------
    The same AnnData with ``adata.obs[key_added]`` added.
    """
    scores = score_marker_states(adata, marker_genes, layer=layer)
    adata.obs[key_added] = scores.idxmax(axis=1).astype("category")
    return adata


def run_from_adata(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    leiden_resolution: float = 1.0,
    marker_genes: dict[str, list[str]] | None = None,
    already_normalized: bool = False,
) -> sc.AnnData:
    """Preprocess, cluster, and assign marker states to an existing AnnData.

    Parameters
    ----------
    adata
        AnnData with count data in ``.X`` (raw or library-size-normalized).
    min_genes
        Minimum genes per cell for filtering.
    min_cells
        Minimum cells per gene for filtering.
    n_top_genes
        Number of highly variable genes to retain.
    n_pcs
        Number of PCA components.
    leiden_resolution
        Resolution parameter for Leiden clustering.
    marker_genes
        Marker gene dict.  Defaults to :py:data:`MARKER_GENES`.
    already_normalized
        If True, skip sc.pp.normalize_total and only run sc.pp.log1p
        (data is library-size-normalized but not log-transformed).
        If False (default), run both normalize_total and log1p
        (data is raw counts).

    Returns
    -------
    AnnData with ``.obs['leiden_cluster']`` and ``.obs['marker_state']``.
    """
    adata = preprocess_standard(
        adata, min_genes=min_genes, min_cells=min_cells,
        n_top_genes=n_top_genes, n_pcs=n_pcs,
        already_normalized=already_normalized,
    )

    adata = cluster_leiden(adata, resolution=leiden_resolution)
    adata = assign_marker_states(adata, marker_genes=marker_genes, layer="lognorm")

    return adata


def run(
    counts_mtx_path: str | Path,
    gene_names_path: str | Path,
    metadata_path: str | Path | None = None,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    leiden_resolution: float = 1.0,
    marker_genes: dict[str, list[str]] | None = None,
    already_normalized: bool = False,
) -> sc.AnnData:
    """Load raw data, preprocess, cluster, and assign marker-based states.

    Pipeline
    --------
    1. Load count matrix, gene names, and (optional) metadata into AnnData.
    2. Filter, normalise (if not already), log1p, select HVGs, scale, PCA.
    3. Leiden clustering on PCA components (unsupervised cross-check).
    4. Marker-gene scoring per state and hard assignment by maximum score.

    Parameters
    ----------
    counts_mtx_path
        Path to the Market Exchange Format count matrix (*.mtx or .gz).
    gene_names_path
        Path to the gene names file, one per line.
    metadata_path
        Optional path to cell metadata TSV (index must match cell order).
    min_genes
        Minimum genes per cell for filtering.
    min_cells
        Minimum cells per gene for filtering.
    n_top_genes
        Number of highly variable genes to retain.
    n_pcs
        Number of PCA components.
    leiden_resolution
        Resolution parameter for Leiden clustering.
    marker_genes
        Marker gene dict.  Defaults to :py:data:`MARKER_GENES`.
    already_normalized
        If True, skip sc.pp.normalize_total and only run sc.pp.log1p
        (data is library-size-normalized but not log-transformed).
        If False (default), run both normalize_total and log1p
        (data is raw counts).

    Returns
    -------
    AnnData with ``.obs['leiden_cluster']`` and ``.obs['marker_state']``.
    """
    if metadata_path is not None:
        adata = load_weinreb(counts_mtx_path, gene_names_path, metadata_path)
    else:
        adata = load_from_mtx(counts_mtx_path, gene_names_path)

    adata = preprocess_standard(
        adata, min_genes=min_genes, min_cells=min_cells,
        n_top_genes=n_top_genes, n_pcs=n_pcs,
        already_normalized=already_normalized,
    )

    adata = cluster_leiden(adata, resolution=leiden_resolution)
    adata = assign_marker_states(adata, marker_genes=marker_genes, layer="lognorm")

    return adata


if __name__ == "__main__":
    """Audit MARKER_GENES for overlapping genes and panel sizes."""
    print("=" * 70)
    print("MARKER_GENES OVERLAP AUDIT")
    print("=" * 70)

    # Print marker count per state
    print("\nMARKER COUNT PER STATE:")
    for state, genes in sorted(MARKER_GENES.items()):
        print(f"  {state:5s}: {len(genes)} markers")
    print()

    # Build reverse mapping: gene -> list of states
    gene_to_states: dict[str, list[str]] = {}
    for state, genes in MARKER_GENES.items():
        for gene in genes:
            gene_to_states.setdefault(gene, []).append(state)

    # Find overlaps
    overlaps = {gene: states for gene, states in gene_to_states.items() if len(states) > 1}

    if not overlaps:
        print("No overlapping genes found.")
    else:
        print(f"Found {len(overlaps)} gene(s) appearing in multiple states:\n")
        for gene, states in sorted(overlaps.items()):
            print(f"  Gene: {gene}")
            print(f"    States: {', '.join(states)}")
            print(f"    Count:  {len(states)}")
            print()

        # Biological interpretation notes
        print("INTERPRETATION NOTES:")
        print("  - Flt3 in MPP + LMPP: Flt3 is a known early progenitor marker; "
              "MPP and LMPP are sequential/adjacent states where Flt3 "
              "expression transitions. Biologically expected overlap for "
              "a transitional marker.")
        print("  - Mpo in CMP + GMP: Myeloperoxidase is a myeloid-lineage enzyme; "
              "CMP is the common myeloid progenitor, GMP is the "
              "granulocyte-monocyte progenitor (downstream of CMP). "
              "Mpo expression begins in CMP and increases in GMP. "
              "Biologically expected nested expression.")
        print("  - Review all overlaps above before modifying panels.")
    print("=" * 70)


def audit_hsc_discrimination() -> None:
    """Audit HSC marker gene discrimination power using log-normalized expression."""
    print("\n" + "=" * 70)
    print("HSC MARKER DISCRIMINATION AUDIT")
    print("=" * 70)
    print("\nLoading and preprocessing Weinreb data (15k subsample, seed=42)...")

    # Reuse same data loading pipeline as resolution_sensitivity.py
    from .data_loading import load_weinreb

    base_path = Path("scripts/data/raw/weinreb")
    counts_path = base_path / "stateFate_inVitro_normed_counts.mtx.gz"
    genes_path = base_path / "stateFate_inVitro_gene_names.txt.gz"
    meta_path = base_path / "stateFate_inVitro_metadata.txt.gz"

    for p in [counts_path, genes_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}")

    adata = load_weinreb(counts_path, genes_path, meta_path)

    # Subsample first
    SUBSAMPLE_N = 15000
    rng = np.random.default_rng(42)
    if adata.n_obs > SUBSAMPLE_N:
        idx = rng.choice(adata.n_obs, SUBSAMPLE_N, replace=False)
        adata = adata[idx].copy()

    # Preprocess with already_normalized=True
    adata = preprocess_standard(adata, min_genes=200, min_cells=3, n_top_genes=2000, n_pcs=30, already_normalized=True)

    # Assign marker states (uses lognorm layer)
    adata = assign_marker_states(adata, MARKER_GENES, layer="lognorm")

    # Get log-normalized expression matrix for HSC's marker genes
    hsc_genes = MARKER_GENES["HSC"]
    lognorm = adata.layers["lognorm"]
    marker_states = adata.obs["marker_state"].values

    # Check which HSC genes are present in the HVG subset
    present_genes = [g for g in hsc_genes if g in adata.var_names]
    missing_genes = [g for g in hsc_genes if g not in adata.var_names]
    if missing_genes:
        print(f"WARNING: Missing genes from HVG subset: {missing_genes}")

    # Compute per-gene discrimination
    states = sorted(set(marker_states))
    print(f"\n{'Gene':<12s} {'HSC_mean':>10s} {'Other_means (state:value)':>50s} {'Disc_ratio':>12s} {'Flag':>15s}")
    print("-" * 120)

    for gene in present_genes:
        gene_idx = list(adata.var_names).index(gene)
        gene_expr = lognorm[:, gene_idx]

        # Mean expression in HSC
        hsc_mask = (marker_states == "HSC")
        hsc_mean = float(gene_expr[hsc_mask].mean()) if hsc_mask.any() else 0.0

        # Mean expression in each non-HSC state
        other_means = {}
        for state in states:
            if state == "HSC":
                continue
            mask = (marker_states == state)
            if mask.any():
                other_means[state] = float(gene_expr[mask].mean())
            else:
                other_means[state] = 0.0

        # Find next-highest non-HSC state
        if other_means:
            next_highest_state = max(other_means, key=other_means.get)
            next_highest_mean = other_means[next_highest_state]
        else:
            next_highest_state = "N/A"
            next_highest_mean = 0.0

        # Discrimination ratio
        disc_ratio = hsc_mean / next_highest_mean if next_highest_mean > 0 else float('inf')

        # Format other means
        other_str = ", ".join(f"{s}:{v:.2f}" for s, v in sorted(other_means.items(), key=lambda x: -x[1]))

        # Flag
        flag = "LOW DISCRIMINATION" if disc_ratio < 1.3 and disc_ratio != float('inf') else ""

        print(f"{gene:<12s} {hsc_mean:>10.3f} {other_str:<50s} {disc_ratio:>10.2f}  {flag}")

    print("=" * 70)
    print("Threshold: discrimination ratio < 1.3 flagged as LOW DISCRIMINATION")
    print("=" * 70)


if __name__ == "__main__":
    """Audit MARKER_GENES for overlapping genes and panel sizes."""
    print("=" * 70)
    print("MARKER_GENES OVERLAP AUDIT")
    print("=" * 70)

    # Print marker count per state
    print("\nMARKER COUNT PER STATE:")
    for state, genes in sorted(MARKER_GENES.items()):
        print(f"  {state:5s}: {len(genes)} markers")
    print()

    # Build reverse mapping: gene -> list of states
    gene_to_states: dict[str, list[str]] = {}
    for state, genes in MARKER_GENES.items():
        for gene in genes:
            gene_to_states.setdefault(gene, []).append(state)

    # Find overlaps
    overlaps = {gene: states for gene, states in gene_to_states.items() if len(states) > 1}

    if not overlaps:
        print("No overlapping genes found.")
    else:
        print(f"Found {len(overlaps)} gene(s) appearing in multiple states:\n")
        for gene, states in sorted(overlaps.items()):
            print(f"  Gene: {gene}")
            print(f"    States: {', '.join(states)}")
            print(f"    Count:  {len(states)}")
            print()

        # Biological interpretation notes
        print("INTERPRETATION NOTES:")
        print("  - Flt3 in MPP + LMPP: Flt3 is a known early progenitor marker; "
              "MPP and LMPP are sequential/adjacent states where Flt3 "
              "expression transitions. Biologically expected overlap for "
              "a transitional marker.")
        print("  - Mpo in CMP + GMP: Myeloperoxidase is a myeloid-lineage enzyme; "
              "CMP is the common myeloid progenitor, GMP is the "
              "granulocyte-monocyte progenitor (downstream of CMP). "
              "Mpo expression begins in CMP and increases in GMP. "
              "Biologically expected nested expression.")
        print("  - Review all overlaps above before modifying panels.")
    print("=" * 70)

    # Run HSC discrimination audit
    audit_hsc_discrimination()
