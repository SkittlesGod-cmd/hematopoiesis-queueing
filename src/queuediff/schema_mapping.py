"""Schema mapping between datasets.

Maps column names and metadata schemas between different single-cell
datasets to enable cross-dataset validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import pandas as pd


@dataclass
class DatasetSchema:
    """Schema for a single-cell dataset.

    Attributes
    ----------
    name : str
        Dataset name (e.g., 'Weinreb2020', 'Nestorowa2016').
    timepoint_col : str
        Column in obs containing timepoint information.
    time_unit : str
        Unit of timepoint values ('days' or 'hours').
    has_clones : bool
        Whether the dataset has lineage barcodes.
    gene_id_type : str
        Gene identifier type ('symbol' or 'ensembl').
    already_normalized : bool
        Whether the data is pre-normalized.
    additional_metadata : dict[str, str]
        Additional column mappings.
    """

    name: str
    timepoint_col: str = "Time_point"
    time_unit: str = "days"
    has_clones: bool = False
    gene_id_type: str = "symbol"
    already_normalized: bool = False
    additional_metadata: dict[str, str] = field(default_factory=dict)


# Pre-defined schemas for known datasets
WEINREB_SCHEMA = DatasetSchema(
    name="Weinreb2020",
    timepoint_col="Time_point",
    time_unit="days",
    has_clones=True,
    gene_id_type="symbol",
    already_normalized=True,
    additional_metadata={"library_col": "Library"},
)

NESTOROWA_SCHEMA = DatasetSchema(
    name="Nestorowa2016",
    timepoint_col="",  # Single timepoint, no column needed
    time_unit="days",
    has_clones=False,
    gene_id_type="symbol",
    already_normalized=False,
)


def validate_schema(
    adata: ad.AnnData,
    schema: DatasetSchema,
) -> list[str]:
    """Validate that an AnnData object conforms to a schema.

    Parameters
    ----------
    adata : AnnData
        Dataset to validate.
    schema : DatasetSchema
        Expected schema.

    Returns
    -------
    list[str]
        List of validation issues (empty if all good).
    """
    issues = []

    # Check timepoint column
    if schema.timepoint_col and schema.timepoint_col not in adata.obs.columns:
        issues.append(f"Missing timepoint column: '{schema.timepoint_col}'")

    # Check clone matrix
    if schema.has_clones and "clone_matrix" not in adata.obsm:
        issues.append("Schema expects clone_matrix in obsm but it's absent")

    # Check data type
    if adata.X.dtype not in (float, "float32", "float64"):
        issues.append(f"Expected float dtype, got {adata.X.dtype}")

    # Check additional metadata
    for key, col_name in schema.additional_metadata.items():
        if col_name not in adata.obs.columns:
            issues.append(f"Missing metadata column '{col_name}' (key: {key})")

    return issues


def map_gene_names(
    source_genes: list[str],
    target_genes: list[str],
) -> dict[str, str | None]:
    """Map gene names between datasets.

    Simple case-insensitive matching (both datasets use gene symbols).

    Parameters
    ----------
    source_genes : list[str]
        Gene names from source dataset.
    target_genes : list[str]
        Gene names from target dataset.

    Returns
    -------
    dict[str, str | None]
        Source gene -> matched target gene (or None if no match).
    """
    target_lower = {g.lower(): g for g in target_genes}
    return {
        gene: target_lower.get(gene.lower())
        for gene in source_genes
    }
