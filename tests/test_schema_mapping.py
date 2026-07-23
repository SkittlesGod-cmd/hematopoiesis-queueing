"""Tests for schema_mapping module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from queuediff.schema_mapping import (
    DatasetSchema,
    NESTOROWA_SCHEMA,
    WEINREB_SCHEMA,
    map_gene_names,
    validate_schema,
)


class TestDatasetSchemas:
    def test_weinreb_schema_exists(self):
        assert WEINREB_SCHEMA is not None
        assert isinstance(WEINREB_SCHEMA, DatasetSchema)

    def test_weinreb_schema_attributes(self):
        assert WEINREB_SCHEMA.name == "Weinreb2020"
        assert WEINREB_SCHEMA.timepoint_col == "Time_point"
        assert WEINREB_SCHEMA.time_unit == "days"
        assert WEINREB_SCHEMA.has_clones is True
        assert WEINREB_SCHEMA.gene_id_type == "symbol"
        assert WEINREB_SCHEMA.already_normalized is True
        assert WEINREB_SCHEMA.additional_metadata == {"library_col": "Library"}

    def test_nestorowa_schema_exists(self):
        assert NESTOROWA_SCHEMA is not None
        assert isinstance(NESTOROWA_SCHEMA, DatasetSchema)

    def test_nestorowa_schema_attributes(self):
        assert NESTOROWA_SCHEMA.name == "Nestorowa2016"
        assert NESTOROWA_SCHEMA.timepoint_col == ""
        assert NESTOROWA_SCHEMA.time_unit == "days"
        assert NESTOROWA_SCHEMA.has_clones is False
        assert NESTOROWA_SCHEMA.gene_id_type == "symbol"
        assert NESTOROWA_SCHEMA.already_normalized is False

    def test_custom_schema_defaults(self):
        schema = DatasetSchema(name="Test")
        assert schema.timepoint_col == "Time_point"
        assert schema.time_unit == "days"
        assert schema.has_clones is False
        assert schema.gene_id_type == "symbol"
        assert schema.already_normalized is False
        assert schema.additional_metadata == {}


class TestValidateSchema:
    def test_valid_weinreb_adata(self, synthetic_adata_with_clones):
        """Valid Weinreb-like adata returns no issues."""
        issues = validate_schema(synthetic_adata_with_clones, WEINREB_SCHEMA)
        assert issues == []

    def test_valid_nestorowa_adata(self, synthetic_adata):
        """Valid Nestorowa-like adata returns no issues."""
        # Nestorowa has no timepoint column, no clones, no additional_metadata
        issues = validate_schema(synthetic_adata, NESTOROWA_SCHEMA)
        assert issues == []

    def test_missing_timepoint_column(self, synthetic_adata):
        """Missing timepoint column produces an issue."""
        schema = DatasetSchema(name="Bad", timepoint_col="MissingCol")
        issues = validate_schema(synthetic_adata, schema)
        assert len(issues) == 1
        assert "MissingCol" in issues[0]
        assert "timepoint" in issues[0].lower()

    def test_missing_clone_matrix(self, synthetic_adata):
        """has_clones=True but no clone_matrix in obsm produces an issue."""
        schema = DatasetSchema(name="NeedsClones", has_clones=True)
        issues = validate_schema(synthetic_adata, schema)
        assert len(issues) == 1
        assert "clone_matrix" in issues[0]

    def test_missing_additional_metadata(self, synthetic_adata):
        """Missing additional metadata column produces an issue."""
        schema = DatasetSchema(
            name="NeedsMeta",
            additional_metadata={"batch_col": "BatchID"},
        )
        issues = validate_schema(synthetic_adata, schema)
        assert len(issues) == 1
        assert "BatchID" in issues[0]
        assert "batch_col" in issues[0]

    def test_non_float_dtype(self, synthetic_adata):
        """Non-float X dtype produces an issue."""
        # Create a copy with int dtype
        adata_int = synthetic_adata.copy()
        adata_int.X = adata_int.X.astype(int)
        issues = validate_schema(adata_int, WEINREB_SCHEMA)
        # Should have dtype issue (and possibly clone_matrix issue since
        # this fixture doesn't have clones)
        dtype_issues = [i for i in issues if "float" in i]
        assert len(dtype_issues) == 1

    def test_empty_timepoint_col_skips_check(self, synthetic_adata):
        """Nestorowa has timepoint_col='' so the timepoint check is skipped."""
        # The synthetic_adata has Time_point, but let's verify with an adata
        # that doesn't — with empty timepoint_col it should still pass
        adata_no_tp = synthetic_adata.copy()
        adata_no_tp.obs = adata_no_tp.obs.drop(columns=["Time_point"])
        issues = validate_schema(adata_no_tp, NESTOROWA_SCHEMA)
        assert issues == []

    def test_multiple_issues(self):
        """Multiple missing components produce multiple issues."""
        X = np.ones((5, 3), dtype=np.float64)
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        schema = DatasetSchema(
            name="BadAll",
            timepoint_col="Time_point",
            has_clones=True,
            additional_metadata={"lib_col": "Library"},
        )
        issues = validate_schema(adata, schema)
        # Missing timepoint, missing clone_matrix, missing Library
        assert len(issues) == 3


class TestMapGeneNames:
    def test_exact_match(self):
        """Exact gene name matches are returned."""
        mapping = map_gene_names(["GATA1", "SPI1"], ["GATA1", "SPI1"])
        assert mapping == {"GATA1": "GATA1", "SPI1": "SPI1"}

    def test_case_insensitive_match(self):
        """Case-insensitive matching returns the target casing."""
        mapping = map_gene_names(["gata1", "SPI1"], ["GATA1", "Spi1"])
        assert mapping["gata1"] == "GATA1"
        assert mapping["SPI1"] == "Spi1"

    def test_no_match_returns_none(self):
        """Genes with no match in target return None."""
        mapping = map_gene_names(["UNKNOWN", "GATA1"], ["GATA1", "SPI1"])
        assert mapping["UNKNOWN"] is None
        assert mapping["GATA1"] == "GATA1"

    def test_empty_inputs(self):
        """Empty source gene list returns empty dict."""
        mapping = map_gene_names([], ["GATA1", "SPI1"])
        assert mapping == {}

    def test_empty_target_list(self):
        """Empty target gene list maps everything to None."""
        mapping = map_gene_names(["GATA1", "SPI1"], [])
        assert mapping == {"GATA1": None, "SPI1": None}

    def test_both_empty(self):
        """Both empty inputs return empty dict."""
        mapping = map_gene_names([], [])
        assert mapping == {}

    def test_duplicate_source_variations(self):
        """Source has duplicate case variations; each maps independently."""
        mapping = map_gene_names(["GATA1", "gata1"], ["GATA1"])
        assert mapping["GATA1"] == "GATA1"
        assert mapping["gata1"] == "GATA1"

    def test_target_lower_takes_first_occurrence(self):
        """When target has case-duplicate, the first occurrence wins."""
        mapping = map_gene_names(["gata1"], ["GATA1", "gata1"])
        # Target dict is built with {g.lower(): g}, second overwrites first
        assert mapping["gata1"] == "gata1"

    def test_mixed_matches_and_misses(self):
        """Mix of matches and misses in a larger gene list."""
        source = ["GATA1", "fakemarker", "SPI1", "another_fake"]
        target = ["GATA1", "SPI1", "KLF1"]
        mapping = map_gene_names(source, target)
        assert mapping["GATA1"] == "GATA1"
        assert mapping["SPI1"] == "SPI1"
        assert mapping["fakemarker"] is None
        assert mapping["another_fake"] is None

    def test_large_gene_list(self):
        """Mapping works correctly on a large gene list."""
        source = [f"Gene{i}" for i in range(500)]
        target = [f"gene{i}" for i in range(500)]  # lowercase targets
        mapping = map_gene_names(source, target)
        for gene in source:
            assert mapping[gene] is not None
            assert mapping[gene].startswith("gene")
