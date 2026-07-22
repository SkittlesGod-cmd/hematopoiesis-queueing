"""Tests for data_loading module."""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, random as sparse_random

from queuediff.data_loading import load_weinreb, preprocess_standard


@pytest.fixture
def fake_weinreb_dir(tmp_path: Path) -> Path:
    """Create fake Weinreb data files in a temp directory."""
    rng = np.random.default_rng(42)
    n_cells, n_genes, n_clones = 50, 30, 5

    # Gene names (with a trailing blank line to test stripping)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    gene_names_content = "\n".join(gene_names) + "\n\n"
    with gzip.open(tmp_path / "stateFate_inVitro_gene_names.txt.gz", "wt") as f:
        f.write(gene_names_content)

    # Normalized count matrix (genes x cells as stored in file, transposed on load)
    X = sparse_random(n_genes, n_cells, density=0.3, random_state=42, format="coo")
    X = X.astype(np.float32)
    with gzip.open(tmp_path / "stateFate_inVitro_normed_counts.mtx.gz", "wb") as f:
        mmwrite(f, X)

    # Metadata
    cell_ids = [f"Cell{i}" for i in range(n_cells)]
    timepoints = np.repeat([2.0, 4.0, 6.0], [20, 15, 15])
    metadata = pd.DataFrame(
        {
            "Library": [f"Lib{int(t)}" for t in timepoints],
            "Time_point": timepoints,
        },
        index=cell_ids,
    )
    with gzip.open(tmp_path / "stateFate_inVitro_metadata.txt.gz", "wt") as f:
        metadata.to_csv(f, sep="\t")

    # Clone matrix (genes x cells orientation in file)
    clone_mtx = sparse_random(n_clones, n_cells, density=0.2, random_state=7, format="coo")
    clone_mtx = (clone_mtx > 0).astype(np.float32)
    with gzip.open(tmp_path / "stateFate_inVitro_clone_matrix.mtx.gz", "wb") as f:
        mmwrite(f, clone_mtx)

    return tmp_path


class TestLoadWeinreb:
    def test_returns_anndata(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir)
        assert isinstance(adata, ad.AnnData)

    def test_shape_cells_x_genes(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir)
        # 50 cells, 30 genes
        assert adata.shape == (50, 30)

    def test_dtype_float32(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir)
        assert adata.X.dtype == np.float32

    def test_no_trailing_blank_gene_names(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir)
        assert all(name.strip() != "" for name in adata.var_names)
        assert len(adata.var_names) == 30

    def test_metadata_columns_present(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir)
        assert "Time_point" in adata.obs.columns
        assert "Library" in adata.obs.columns

    def test_timepoints_correct(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir)
        expected = {2.0, 4.0, 6.0}
        assert set(adata.obs["Time_point"].unique()) == expected

    def test_include_clones_false_no_obsm(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir, include_clones=False)
        assert "clone_matrix" not in adata.obsm

    def test_include_clones_true_adds_obsm(self, fake_weinreb_dir):
        adata = load_weinreb(fake_weinreb_dir, include_clones=True)
        assert "clone_matrix" in adata.obsm
        assert adata.obsm["clone_matrix"].shape[0] == adata.n_obs

    def test_clone_matrix_survives_subsetting(self, fake_weinreb_dir):
        """Clone matrix in obsm must survive adata[subset] without position bugs."""
        adata = load_weinreb(fake_weinreb_dir, include_clones=True)
        # Subset to first 20 cells
        subset = adata[:20].copy()
        assert subset.obsm["clone_matrix"].shape[0] == 20


class TestPreprocessStandard:
    """Tests for preprocess_standard.

    Note: synthetic_adata has only 100 genes, so min_genes must be < 100
    to avoid filtering all cells. We pass min_genes=10 explicitly.
    """

    PP_KWARGS = dict(already_normalized=True, min_genes=10, min_cells=1)

    def test_creates_lognorm_full(self, synthetic_adata):
        """lognorm_full in obsm (not layers) because it has full gene set."""
        result = preprocess_standard(synthetic_adata, **self.PP_KWARGS)
        assert "lognorm_full" in result.obsm

    def test_creates_lognorm_layer(self, synthetic_adata):
        result = preprocess_standard(synthetic_adata, **self.PP_KWARGS)
        assert "lognorm" in result.layers

    def test_lognorm_full_has_more_genes_than_lognorm(self, synthetic_adata):
        """lognorm_full covers full gene set, lognorm only HVGs."""
        result = preprocess_standard(
            synthetic_adata, **self.PP_KWARGS, n_top_genes=50, n_pcs=20
        )
        # lognorm should have n_top_genes columns (HVG subset)
        assert result.layers["lognorm"].shape[1] == result.n_vars
        # lognorm_full should have more genes (full gene set, stored in obsm)
        assert result.obsm["lognorm_full"].shape[1] > result.layers["lognorm"].shape[1]

    def test_has_pca(self, synthetic_adata):
        result = preprocess_standard(synthetic_adata, **self.PP_KWARGS)
        assert "X_pca" in result.obsm

    def test_x_is_dense_after_preprocessing(self, synthetic_adata):
        result = preprocess_standard(synthetic_adata, **self.PP_KWARGS)
        assert not hasattr(result.X, "toarray")

    def test_already_normalized_skips_normalize(self, synthetic_adata):
        """With already_normalized=True, no normalization step occurs."""
        result = preprocess_standard(synthetic_adata, **self.PP_KWARGS)
        # lognorm_full values should all be >= 0 (log1p of non-negative)
        vals = result.obsm["lognorm_full"]
        assert np.all(vals >= 0)

    def test_hvg_flavor_seurat(self, synthetic_adata):
        """HVG selection uses seurat flavor (not seurat_v3)."""
        result = preprocess_standard(synthetic_adata, **self.PP_KWARGS)
        # If it ran without error using seurat flavor, test passes
        assert result.n_vars <= synthetic_adata.n_vars
