"""Shared test fixtures for the queuediff test suite.

Tests use synthetic fixtures, not real data, so they run fast and
don't require data downloads.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def synthetic_adata() -> ad.AnnData:
    """Create a small synthetic AnnData mimicking Weinreb structure.

    200 cells x 100 genes, float32, sparse, with metadata columns
    matching Weinreb format.
    """
    rng = np.random.default_rng(42)
    n_cells, n_genes = 200, 100

    X = rng.poisson(lam=2, size=(n_cells, n_genes)).astype(np.float32)
    X = csr_matrix(X)

    gene_names = [f"Gene{i}" for i in range(n_genes)]
    cell_ids = [f"Cell{i}" for i in range(n_cells)]

    timepoints = np.repeat([2, 4, 6], [80, 60, 60])
    obs = pd.DataFrame(
        {
            "Library": [f"Lib{t}" for t in timepoints],
            "Time_point": timepoints.astype(float),
        },
        index=cell_ids,
    )

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=gene_names))
    return adata


@pytest.fixture
def synthetic_adata_with_clones(synthetic_adata) -> ad.AnnData:
    """Synthetic AnnData with a clone matrix in obsm."""
    rng = np.random.default_rng(123)
    n_cells = synthetic_adata.n_obs
    n_clones = 20

    # Sparse binary clone assignment (each cell in at most one clone)
    clone_data = np.zeros((n_cells, n_clones), dtype=np.float32)
    # Assign 60% of cells to a clone
    cloned_cells = rng.choice(n_cells, size=int(n_cells * 0.6), replace=False)
    for cell_idx in cloned_cells:
        clone_idx = rng.integers(0, n_clones)
        clone_data[cell_idx, clone_idx] = 1.0

    synthetic_adata.obsm["clone_matrix"] = csr_matrix(clone_data)
    return synthetic_adata


@pytest.fixture
def preprocessed_adata(synthetic_adata) -> ad.AnnData:
    """Synthetic AnnData that has been through preprocessing.

    Has lognorm_full, lognorm layers, and PCA.
    Mimics the output of preprocess_standard.
    """
    rng = np.random.default_rng(99)
    n_cells = synthetic_adata.n_obs
    n_hvgs = 50
    n_pcs = 20

    # Subset to "HVGs"
    adata = synthetic_adata[:, :n_hvgs].copy()

    # Create obsm lognorm_full (full gene set, different column count)
    lognorm_full = rng.random((n_cells, synthetic_adata.n_vars)).astype(np.float32)
    adata.obsm["lognorm_full"] = lognorm_full
    adata.uns["lognorm_full_genes"] = list(synthetic_adata.var_names)

    lognorm = rng.random((n_cells, n_hvgs)).astype(np.float32)
    adata.layers["lognorm"] = lognorm

    # Scaled dense X
    adata.X = rng.standard_normal((n_cells, n_hvgs)).astype(np.float32)

    # PCA
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, n_pcs)).astype(np.float32)

    return adata
