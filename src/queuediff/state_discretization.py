from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread


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
    ],
    "MEP": [
        "Gata1", "Klf1", "EpoR", "Tal1", "Gypa",       # Laurenti & Göttgens, Fig 3
    ],
    "GMP": [
        "Elane", "Mpo", "Ctsg", "Prtn3", "Csf3r",      # Weinreb et al. Fig 2 markers
    ],
}


def load_weinreb_counts(
    counts_mtx_path: str | Path,
    gene_names_path: str | Path,
    metadata_path: str | Path,
) -> sc.AnnData:
    """Load Weinreb-format raw counts (mtx + gene list + metadata) into AnnData.

    Parameters
    ----------
    counts_mtx_path
        Path to the Market Exchange Format count matrix (*.mtx or *.mtx.gz).
    gene_names_path
        Path to the gene names file, one name per line.
    metadata_path
        Path to the cell metadata TSV.

    Returns
    -------
    AnnData with counts in ``.X`` (cells × genes), gene names in ``.var_names``,
    and cell metadata in ``.obs``.
    """
    counts_mtx_path, gene_names_path, metadata_path = (
        Path(counts_mtx_path), Path(gene_names_path), Path(metadata_path)
    )

    X = mmread(_open_text(counts_mtx_path)).T.tocsr()
    with _open_text(gene_names_path) as f:
        gene_names = [line.strip() for line in f]
    meta = pd.read_csv(_open_text(metadata_path), sep="\t", index_col=0)

    adata = sc.AnnData(X=X, dtype=np.float32)
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs = meta
    adata.obs_names_make_unique()
    return adata


def load_from_mtx(
    counts_mtx_path: str | Path,
    gene_names_path: str | Path,
) -> sc.AnnData:
    """Load a sparse count matrix and gene names into an AnnData object.

    Parameters
    ----------
    counts_mtx_path
        Path to the .mtx file (optionally gzipped).
    gene_names_path
        Path to the gene names file, one name per line.

    Returns
    -------
    AnnData with ``.X`` containing the sparse count matrix.
    """
    counts_mtx_path, gene_names_path = Path(counts_mtx_path), Path(gene_names_path)
    X = mmread(_open_text(counts_mtx_path)).T.tocsr()
    with _open_text(gene_names_path) as f:
        gene_names = [line.strip() for line in f]
    adata = sc.AnnData(X=X, dtype=np.float32)
    adata.var_names = gene_names
    adata.var_names_make_unique()
    return adata


def preprocess_standard(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
) -> sc.AnnData:
    """Run standard single-cell preprocessing for log-normalized input data.

    The Weinreb stateFate_inVitro dataset ships as normed_counts (already
    library-size normalised, but unlogged). This function assumes the input
    is raw counts OR already normalised counts. The pipeline below matches
    Scanpy's recommended order for flavor='seurat' (designed for
    log-normalised data):

    Steps:
        1. Filter cells with fewer than *min_genes* detected.
        2. Filter genes detected in fewer than *min_cells*.
        3. Total-count normalise to 10,000 per cell.
        4. Log1p transform.
        5. Select *n_top_genes* highly variable genes using flavor='seurat'
           on the log-normalised data.
        6. Subset to HVGs.
        7. Store a copy of the log-normalised HVG matrix in
           adata.layers['lognorm'] for marker-gene scoring.
        8. Convert to dense array (HVG subset is small: ~2000 genes).
        9. Scale to unit variance (clip at 10).
       10. Run PCA (*n_pcs* components).

    The scaled/clipped adata.X is used only for PCA and downstream
    Leiden clustering. Marker-gene scoring (sc.tl.score_genes) must use
    the log-normalised values in adata.layers['lognorm'].

    Parameters
    ----------
    adata
        AnnData with count data (raw or pre-normalised).
    min_genes
        Minimum number of genes that must be detected in a cell.
    min_cells
        Minimum number of cells a gene must be detected in.
    n_top_genes
        Number of highly variable genes to select.
    n_pcs
        Number of PCA components to compute.

    Returns
    -------
    The same AnnData object, modified in place.
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
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
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, key=neighbors_key)
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                 flavor="igraph", n_iterations=2, directed=False)
    adata.obs[key_added] = adata.obs[key_added].astype("category")
    return adata


def score_marker_states(
    adata: sc.AnnData,
    marker_genes: dict[str, list[str]] | None = None,
    score_prefix: str = "score_",
) -> pd.DataFrame:
    """Score every cell against every marker-gene state panel.

    Uses ``sc.tl.score_genes`` internally for each state.  The score is
    the average normalised expression of the gene set minus the average
    expression of a randomly sampled reference set of the same size.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in ``.layers['lognorm']``
        (or ``.X`` if the layer is missing).
    marker_genes
        Dict mapping state name → list of marker gene symbols.
        Defaults to :py:data:`MARKER_GENES`.
    score_prefix
        Prefix for the temporary score column inserted into ``adata.obs``.

    Returns
    -------
    DataFrame (cells × states) of gene-set scores.
    """
    if marker_genes is None:
        marker_genes = MARKER_GENES

    # Use lognorm layer for scoring; preserve original X
    orig_X = adata.X
    if "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"]

    scores = {}
    for state, genes in marker_genes.items():
        col = score_prefix + state
        present = [g for g in genes if g in adata.var_names]
        if not present:
            scores[state] = np.zeros(adata.n_obs)
            continue
        sc.tl.score_genes(adata, gene_list=present, score_name=col, random_state=0)
        scores[state] = adata.obs[col].values.copy()
        del adata.obs[col]

    # Restore original X
    adata.X = orig_X

    return pd.DataFrame(scores, index=adata.obs_names)


def assign_marker_states(
    adata: sc.AnnData,
    marker_genes: dict[str, list[str]] | None = None,
    key_added: str = "marker_state",
) -> sc.AnnData:
    """Assign each cell to its highest-scoring marker-defined state.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in ``.X``.
    marker_genes
        Dict mapping state name → marker gene list.
        Defaults to :py:data:`MARKER_GENES`.
    key_added
        Column name in ``adata.obs`` for the assigned state.

    Returns
    -------
    The same AnnData with ``adata.obs[key_added]`` added.
    """
    scores = score_marker_states(adata, marker_genes)
    adata.obs[key_added] = scores.idxmax(axis=1).astype("category")
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
) -> sc.AnnData:
    """Load raw data, preprocess, cluster, and assign marker-based states.

    Pipeline
    --------
    1. Load count matrix, gene names, and (optional) metadata into AnnData.
    2. Filter, normalise, log1p, select HVGs, scale, PCA.
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

    Returns
    -------
    AnnData with ``.obs['leiden_cluster']`` and ``.obs['marker_state']``.
    """
    if metadata_path is not None:
        adata = load_weinreb_counts(counts_mtx_path, gene_names_path, metadata_path)
    else:
        adata = load_from_mtx(counts_mtx_path, gene_names_path)

    adata = preprocess_standard(
        adata, min_genes=min_genes, min_cells=min_cells,
        n_top_genes=n_top_genes, n_pcs=n_pcs,
    )

    adata = cluster_leiden(adata, resolution=leiden_resolution)
    adata = assign_marker_states(adata, marker_genes=marker_genes)

    return adata


def _open_text(path: str | Path):
    """Open a (possibly gzipped) text file and return a file-like object."""
    path = Path(path)
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")
