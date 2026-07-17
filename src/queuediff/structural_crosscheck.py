from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------------
# Shared coarse meta-hierarchy  (from docs/research_plan.md)
#
#   Stem/Multipotent → Myeloid-primed Progenitor → Lymphoid-primed
#   Progenitor → Committed Progenitor → Mature
#
# These tiers are broad enough that the native states of both the Weinreb and
# the Nestorowa dataset can be nested underneath them.
# ---------------------------------------------------------------------------

META_TIERS: list[str] = [
    "Stem_Multipotent",
    "Myeloid_Progenitor",
    "Lymphoid_Progenitor",
    "Committed_Progenitor",
    "Mature",
]

# Native (marker-gene) state → meta-tier mapping.
# Covers all states returned by state_discretization.assign_marker_states().
MARKER_TO_META: dict[str, str] = {
    "HSC":  "Stem_Multipotent",
    "MPP":  "Stem_Multipotent",

    "CMP":  "Myeloid_Progenitor",

    "LMPP": "Lymphoid_Progenitor",

    "MEP":  "Committed_Progenitor",
    "GMP":  "Committed_Progenitor",
}


def map_marker_to_meta(
    adata: sc.AnnData,
    mapping: Optional[dict[str, str]] = None,
    marker_key: str = "marker_state",
    tier_key: str = "marker_tier",
) -> sc.AnnData:
    """Map each cell's marker-based state to its meta-hierarchy tier.

    Parameters
    ----------
    adata
        AnnData with a categorical column *marker_key*.
    mapping
        Dict mapping native state → tier name.
        Defaults to :py:data:`MARKER_TO_META`.
    marker_key
        Column in ``adata.obs`` holding the native marker-based state.
    tier_key
        Column name to add in ``adata.obs``.

    Returns
    -------
    The same AnnData with ``adata.obs[tier_key]`` added.
    """
    if mapping is None:
        mapping = MARKER_TO_META
    result = adata.obs[marker_key].map(mapping).astype("category")
    if "Unmapped" not in result.cat.categories:
        result = result.cat.add_categories("Unmapped")
    result = result.fillna("Unmapped")
    adata.obs[tier_key] = result
    return adata


def map_leiden_to_meta(
    adata: sc.AnnData,
    mapping: Optional[dict[str, str]] = None,
    marker_tier_key: str = "marker_tier",
    leiden_key: str = "leiden_cluster",
    tier_key: str = "leiden_tier",
) -> sc.AnnData:
    """Assign each Leiden cluster to the meta-tier that best describes it.

    For every Leiden cluster the majority marker-based tier among its cells
    is used as the cluster-level tier label.  Every cell in that cluster
    then receives that same tier assignment, producing a Leiden-derived
    tier annotation that can be compared directly with the marker-derived one.

    Parameters
    ----------
    adata
        AnnData with a marker tier column and a Leiden cluster column.
    mapping
        Unused; kept for API consistency.  Tier mapping is inferred from data.
    marker_tier_key
        Column in ``adata.obs`` with the marker-based tier.
    leiden_key
        Column in ``adata.obs`` with the Leiden cluster label.
    tier_key
        Column name to add in ``adata.obs``.

    Returns
    -------
    The same AnnData with ``adata.obs[tier_key]`` added.
    """
    tier_by_cluster = (
        adata.obs.groupby(leiden_key, observed=True)[marker_tier_key]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unmapped")
    )
    tier_map = dict(zip(tier_by_cluster.index, tier_by_cluster.values))
    result = adata.obs[leiden_key].map(tier_map).astype("category")
    if "Unmapped" not in result.cat.categories:
        result = result.cat.add_categories("Unmapped")
    result = result.fillna("Unmapped")
    adata.obs[tier_key] = result
    return adata


def compute_tier_concordance(
    adata: sc.AnnData,
    marker_tier_key: str = "marker_tier",
    leiden_tier_key: str = "leiden_tier",
) -> pd.DataFrame:
    """Compute per-tier concordance between marker-based and Leiden-based tiers.

    For each meta-tier the table reports:

    - **n_cells** — total cells assigned to that tier by the marker method.
    - **n_consistent** — cells whose Leiden-derived tier matches the marker tier.
    - **concordance** — ``n_consistent / n_cells``.
    - **n_clusters** — number of distinct Leiden clusters that map to this tier.

    Parameters
    ----------
    adata
        AnnData with both tier columns present.
    marker_tier_key
        Column name for the marker-derived tier.
    leiden_tier_key
        Column name for the Leiden-derived tier.

    Returns
    -------
    DataFrame with one row per meta-tier, columns as described above.
    """
    m = adata.obs[marker_tier_key].astype(str)
    l = adata.obs[leiden_tier_key].astype(str)
    consistent = m == l

    cluster_map = (
        adata.obs.groupby(leiden_tier_key, observed=True)[marker_tier_key]
        .nunique()
    )

    rows = []
    for tier in META_TIERS:
        mask = adata.obs[marker_tier_key] == tier
        n = mask.sum()
        n_ok = consistent[mask].sum()
        rows.append({
            "meta_tier":       tier,
            "n_cells":         n,
            "n_consistent":    int(n_ok),
            "concordance":     float(n_ok / n) if n > 0 else float("nan"),
            "n_clusters":      int(cluster_map.get(tier, 0)),
        })

    return pd.DataFrame(rows).set_index("meta_tier")


def print_report(concordance: pd.DataFrame, dataset_label: str = "") -> None:
    """Print a human-readable cross-check report to stdout.

    Parameters
    ----------
    concordance
        DataFrame returned by :py:func:`compute_tier_concordance`.
    dataset_label
        Optional name for the dataset (printed in the header).
    """
    header = " Structural Cross-Check Report "
    if dataset_label:
        header += f"– {dataset_label} "
    width = max(len(header) + 4, 60)
    print()
    print("=" * width)
    print(f"  {header}")
    print("=" * width)

    total_cells = concordance["n_cells"].sum()
    total_ok = concordance["n_consistent"].sum()
    overall = total_ok / total_cells if total_cells > 0 else float("nan")

    print(f"\n  Overall concordance:  {overall:.3f}  ({total_ok}/{total_cells} cells)")
    print(f"\n  Per-tier breakdown:")
    print(f"  {'Tier':<28s} {'Cells':>8s}  {'Consistent':>10s}  {'Concordance':>12s}  {'Clusters':>9s}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*9}")
    for tier in concordance.index:
        row = concordance.loc[tier]
        print(
            f"  {tier:<28s} {int(row['n_cells']):>8d}  "
            f"{int(row['n_consistent']):>10d}  "
            f"{row['concordance']:>11.3f}   "
            f"{int(row['n_clusters']):>8d}"
        )

    print()
    print(f"  Tiers represented:   {concordance['n_cells'].notna().sum()} / {len(META_TIERS)}")
    print(f"  Leiden clusters:     {concordance['n_clusters'].sum():.0f}")
    print("=" * width)
    print()


def run(
    adata: sc.AnnData,
    dataset_label: str = "",
    marker_state_key: str = "marker_state",
    leiden_key: str = "leiden_cluster",
    marker_to_meta: Optional[dict[str, str]] = None,
) -> dict:
    """Run the full structural cross-check on a single AnnData object.

    Steps
    -----
    1. Map marker-based states to meta-hierarchy tiers.
    2. Assign each Leiden cluster a tier by majority vote.
    3. Compute per-tier concordance.
    4. Print a formatted report.

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[marker_state_key]`` and
        ``adata.obs[leiden_key]`` (categorical).
    dataset_label
        Name printed in the report header.
    marker_state_key
        Column with marker-based state labels.
    leiden_key
        Column with Leiden cluster labels.
    marker_to_meta
        Dict mapping native state → meta-tier.
        Defaults to :py:data:`MARKER_TO_META`.

    Returns
    -------
    Dict with keys:
        ``"concordance"``  — DataFrame of per-tier metrics,
        ``"n_cells"``      — total cells processed,
        ``"overall_rate"`` — fraction of cells with consistent tier labels.
    """
    if marker_to_meta is None:
        marker_to_meta = MARKER_TO_META

    adata = map_marker_to_meta(adata, marker_to_meta, marker_state_key)
    adata = map_leiden_to_meta(adata, marker_tier_key="marker_tier", leiden_key=leiden_key)
    concordance = compute_tier_concordance(adata)
    print_report(concordance, dataset_label)

    total_cells = concordance["n_cells"].sum()
    total_ok = concordance["n_consistent"].sum()
    overall = total_ok / total_cells if total_cells > 0 else float("nan")

    return {
        "concordance":   concordance,
        "n_cells":       int(total_cells),
        "overall_rate":  float(overall),
    }
