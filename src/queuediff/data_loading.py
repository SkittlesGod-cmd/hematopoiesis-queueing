"""Data loading utilities for single-cell RNA-seq datasets.

Consolidates Weinreb-format and generic MTX loading functions.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread


PathLike = Union[str, Path]


def _open_text(path: PathLike):
    """Open a (possibly gzipped) text file and return a file-like object."""
    path = Path(path)
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_weinreb(data_dir: PathLike) -> sc.AnnData:
    """Load Weinreb-format data from a directory.

    Expects the following files in ``data_dir``:
        - stateFate_inVitro_normed_counts.mtx.gz (cells × genes sparse matrix)
        - stateFate_inVitro_gene_names.txt.gz (one gene symbol per line)
        - stateFate_inVitro_metadata.txt.gz (cell metadata TSV, index=cell barcode)

    The count matrix is already in cells × genes orientation (130,887 × 25,289),
    so NO transpose is applied.

    Parameters
    ----------
    data_dir
        Directory containing the three Weinreb data files.

    Returns
    -------
    AnnData with counts in ``.X`` (cells × genes), gene names in ``.var_names``,
    and cell metadata in ``.obs``.
    """
    data_dir = Path(data_dir)

    counts_path = data_dir / "stateFate_inVitro_normed_counts.mtx.gz"
    gene_names_path = data_dir / "stateFate_inVitro_gene_names.txt.gz"
    metadata_path = data_dir / "stateFate_inVitro_metadata.txt.gz"

    for p in [counts_path, gene_names_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}")

    # NO TRANSPOSE: the Weinreb mtx is already cells × genes
    X = mmread(_open_text(counts_path)).tocsr()
    with _open_text(gene_names_path) as f:
        gene_names = [line.strip() for line in f]
    meta = pd.read_csv(_open_text(metadata_path), sep="\t", index_col=0)

    adata = sc.AnnData(X=X, dtype=np.float32)
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs = meta
    adata.obs_names_make_unique()
    return adata


def load_weinreb_from_files(
    counts_mtx_path: PathLike,
    gene_names_path: PathLike,
    metadata_path: PathLike,
) -> sc.AnnData:
    """Load Weinreb-format raw counts from explicit file paths.

    DEPRECATED: Use ``load_weinreb(data_dir)`` instead, which takes a directory
    and auto-detects the standard file names.

    This function is kept for backward compatibility with code that calls it
    with explicit file paths. The count matrix is expected to be in cells × genes
    orientation (NO transpose applied).

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

    # NO TRANSPOSE: assume input is already cells × genes
    X = mmread(_open_text(counts_mtx_path)).tocsr()
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
    counts_mtx_path: PathLike,
    gene_names_path: PathLike,
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
    AnnData with ``.X`` containing the sparse count matrix (cells × genes).
    """
    counts_mtx_path, gene_names_path = Path(counts_mtx_path), Path(gene_names_path)
    # NO TRANSPOSE: assume input is already cells × genes
    X = mmread(_open_text(counts_mtx_path)).tocsr()
    with _open_text(gene_names_path) as f:
        gene_names = [line.strip() for line in f]
    adata = sc.AnnData(X=X, dtype=np.float32)
    adata.var_names = gene_names
    adata.var_names_make_unique()
    return adata