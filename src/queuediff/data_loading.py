"""Data loading module -- single source of truth for all data ingestion.

No other module should load data directly. All loading goes through
the functions in this module to ensure consistent preprocessing and
clone matrix alignment.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix


def load_weinreb(
    data_dir: str | Path,
    include_clones: bool = False,
) -> ad.AnnData:
    """Load Weinreb et al. 2020 hematopoietic differentiation data.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the four stateFate_inVitro_* files.
    include_clones : bool, default False
        If True, attach LARRY clone matrix to adata.obsm['clone_matrix'].
        Stored in obsm so it survives adata[subset] slicing automatically,
        avoiding position-based alignment bugs after filtering.

    Returns
    -------
    AnnData
        Cells x genes matrix (already normalized, float32).
        adata.obs contains 'Library', 'Time_point', and optionally
        cell-type annotations from the metadata file.

    Notes
    -----
    - Data is ALREADY normalized by the authors. We never call
      sc.pp.normalize_total on this data.
    - Matrix is cells x genes natively. No transpose needed.
    - Timepoints are days 2, 4, 6 only.
    """
    data_dir = Path(data_dir)

    # Load gene names
    gene_names_file = data_dir / "stateFate_inVitro_gene_names.txt.gz"
    with gzip.open(gene_names_file, "rt") as f:
        # Strip trailing blank lines
        gene_names = [line.strip() for line in f if line.strip()]

    # Load count matrix (cells x genes, already normalized)
    mtx_file = data_dir / "stateFate_inVitro_normed_counts.mtx.gz"
    X = mmread(mtx_file).T  # mmread gives genes x cells for this file, transpose
    X = csr_matrix(X, dtype=np.float32)

    # Load metadata
    metadata_file = data_dir / "stateFate_inVitro_metadata.txt.gz"
    metadata = pd.read_csv(metadata_file, sep="\t", index_col=0)

    # Construct AnnData
    adata = ad.AnnData(
        X=X,
        obs=metadata,
        var=pd.DataFrame(index=gene_names),
    )
    # Enforce float32
    adata.X = csr_matrix(adata.X, dtype=np.float32)

    if include_clones:
        # Clone matrix rows align to metadata rows by ORDER at load time.
        # By storing in obsm before any filtering, adata[subset] automatically
        # gives the correctly-aligned subset -- no position-based alignment needed.
        clone_file = data_dir / "stateFate_inVitro_clone_matrix.mtx.gz"
        clone_mtx = mmread(clone_file).T  # same orientation convention
        clone_mtx = csr_matrix(clone_mtx, dtype=np.float32)
        adata.obsm["clone_matrix"] = clone_mtx

    return adata


def preprocess_standard(
    adata: ad.AnnData,
    already_normalized: bool = True,
    n_top_genes: int = 2000,
    min_genes: int = 200,
    min_cells: int = 3,
    n_pcs: int = 50,
) -> ad.AnnData:
    """Standard preprocessing pipeline for scRNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Raw or normalized cells x genes matrix.
    already_normalized : bool, default True
        If True, skip normalize_total (Weinreb data is pre-normalized).
        If False, run normalize_total before log1p.
    n_top_genes : int, default 2000
        Number of highly variable genes to select.
    min_genes : int, default 200
        Minimum genes per cell for filtering.
    min_cells : int, default 3
        Minimum cells per gene for filtering.
    n_pcs : int, default 50
        Number of principal components to compute.

    Returns
    -------
    AnnData
        Processed data with:
        - adata.obsm['lognorm_full']: log-normalized full gene set (dense ndarray)
          (for cell-cycle/apoptosis scoring -- those genes not in HVGs).
          Stored in obsm (not layers) because it has all genes, not just HVGs.
        - adata.uns['lognorm_full_genes']: gene names for lognorm_full columns
        - adata.layers['lognorm']: log-normalized HVG subset
          (for marker gene state scoring)
        - adata.X: scaled PCA-ready matrix
        - adata.obsm['X_pca']: PCA coordinates

    Notes
    -----
    Preprocessing order (do not reorder):
      filter_cells -> filter_genes -> [normalize_total if needed] ->
      log1p -> save lognorm_full -> HVG selection -> subset to HVGs ->
      save lognorm -> dense conversion -> scale -> PCA

    lognorm_full is required because cell-cycle genes (Pcna, Mcm2, Top2a,
    Ccnb1) and apoptosis genes are NOT in the top 2000 HVGs. Using lognorm
    for these scores produces exactly zero -- silent corruption.
    """
    import scanpy as sc

    adata = adata.copy()

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalize if needed (Weinreb data is already normalized)
    if not already_normalized:
        sc.pp.normalize_total(adata, target_sum=1e4)

    # Log-transform
    sc.pp.log1p(adata)

    # Save lognorm_full BEFORE HVG subsetting (full gene set).
    # Required: cell-cycle genes not in HVG subset, would silently score zero on lognorm layer.
    # Stored as obsm (not layers) because layers are gene-aligned and get subset when
    # we subset to HVGs. obsm allows a different column count (all genes, not just HVGs).
    lognorm_full_matrix = adata.X.copy()
    lognorm_full_gene_names = list(adata.var_names)

    # HVG selection -- seurat flavor (NOT seurat_v3, no raw counts available)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat"
    )

    # Subset to HVGs
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Re-attach lognorm_full after subsetting (full gene set, different column count from adata.X)
    adata.obsm["lognorm_full"] = (
        lognorm_full_matrix.toarray() if hasattr(lognorm_full_matrix, "toarray")
        else np.array(lognorm_full_matrix)
    )
    adata.uns["lognorm_full_genes"] = lognorm_full_gene_names

    # Save lognorm AFTER HVG subsetting (HVG subset only)
    adata.layers["lognorm"] = adata.X.copy()

    # Dense conversion (suppresses sparse densification warnings in scale/PCA)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()

    # Scale
    sc.pp.scale(adata, max_value=10)

    # PCA
    sc.pp.pca(adata, n_comps=n_pcs)

    return adata


def load_nestorowa(data_dir: str | Path) -> ad.AnnData:
    """Load Nestorowa et al. 2016 hematopoietic data.

    Used for structural cross-check only (single timepoint, not for
    quantitative bottleneck analysis).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing Nestorowa data files.

    Returns
    -------
    AnnData
        Loaded dataset.
    """
    data_dir = Path(data_dir)

    # Nestorowa provides an expression matrix and metadata
    mtx_file = data_dir / "nestorowa_corrected_log2_transformed_counts.txt.gz"
    expr = pd.read_csv(mtx_file, sep="\t", index_col=0)

    adata = ad.AnnData(
        X=expr.values.astype(np.float32),
        obs=pd.DataFrame(index=expr.index),
        var=pd.DataFrame(index=expr.columns),
    )

    # Load cell type annotations if available
    anno_file = data_dir / "nestorowa_cell_annotations.txt.gz"
    if anno_file.exists():
        annotations = pd.read_csv(anno_file, sep="\t", index_col=0)
        adata.obs = adata.obs.join(annotations, how="left")

    return adata
