"""State discretization: assign cells to hematopoietic states using marker genes.

Assigns each cell to one of {HSC, MPP, LMPP, CMP, MEP, GMP} based on
marker gene expression scores. Also computes cell-cycle and apoptosis
scores using the full gene set (lognorm_full), and derives division/death
rates via population-dynamics calibration.
"""

from __future__ import annotations

import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


# ── Validated marker gene panels ──────────────────────────────────────────────
# Audited against the Weinreb dataset; do not modify without re-validation.
# Known intentional overlaps:
#   Flt3: MPP + LMPP (transitional marker, biologically expected)
#   Mpo:  CMP + GMP (nested myeloid expression, biologically expected)
MARKER_PANELS: dict[str, list[str]] = {
    "HSC":  ["Meis1", "Hlf", "Procr", "Mllt3", "Pbx1", "Hoxb5", "Gata2"],
    "MPP":  ["Cd34", "Kit", "Flt3"],
    "LMPP": ["Flt3", "Il7r", "Dntt", "Fcer1g", "Cd2"],
    "CMP":  ["Csf1r", "Mpo", "Cebpa", "Cebpb", "Cebpe", "Csf2rb"],
    "MEP":  ["Gata1", "Klf1", "EpoR", "Tal1", "Gypa", "Itga2b", "Vwf"],
    "GMP":  ["Elane", "Mpo", "Ctsg", "Prtn3", "Csf3r"],
}

# Cell-cycle gene signature (reduced, confirmed present in Weinreb dataset).
# Mki67 is ABSENT from the dataset.
CELL_CYCLE_S_GENES: list[str] = ["Pcna", "Mcm2"]
CELL_CYCLE_G2M_GENES: list[str] = ["Top2a", "Ccnb1"]

# Apoptosis gene signature (confirmed present in Weinreb dataset).
# Bad is PRO-apoptotic (inhibits Bcl2/Bcl-xL). Tp53 is absent.
APOPTOSIS_PRO_GENES: list[str] = ["Casp3", "Casp8", "Casp9", "Bax", "Bak1", "Cycs", "Bad"]
APOPTOSIS_ANTI_GENES: list[str] = ["Bcl2", "Bcl2l1"]


def score_marker_states(
    adata: ad.AnnData,
    panels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Score each cell for each hematopoietic state using marker gene panels.

    Uses adata.layers['lognorm'] (HVG subset) for marker scoring.
    Each cell's score for a state is the mean expression of that state's
    marker genes present in the data.

    Parameters
    ----------
    adata : AnnData
        Must have adata.layers['lognorm'] with HVG-subset gene expression.
    panels : dict, optional
        State -> gene list mapping. Defaults to MARKER_PANELS.

    Returns
    -------
    pd.DataFrame
        Cells x states score matrix. Index = cell barcodes, columns = state names.

    Notes
    -----
    Genes not found in adata.var_names are silently skipped (with a warning).
    HSC completeness ~0.36 against Leiden clusters is expected (HSC
    transcriptionally close to MPP). This is real biology, not a bug.
    """
    if panels is None:
        panels = MARKER_PANELS

    # Use lognorm layer (HVG subset) for marker scoring
    if "lognorm" not in adata.layers:
        raise ValueError(
            "adata.layers['lognorm'] required for marker scoring. "
            "Run preprocess_standard first."
        )

    # Get lognorm data as dense array
    lognorm = adata.layers["lognorm"]
    if hasattr(lognorm, "toarray"):
        lognorm = lognorm.toarray()
    lognorm = np.asarray(lognorm, dtype=np.float32)

    var_names = list(adata.var_names)
    var_to_idx = {g: i for i, g in enumerate(var_names)}  # O(1) lookups
    scores = {}

    for state, genes in panels.items():
        found = [g for g in genes if g in var_to_idx]
        missing = [g for g in genes if g not in var_to_idx]
        if missing:
            warnings.warn(
                f"State '{state}': genes {missing} not in var_names (HVG subset). "
                f"Using {len(found)}/{len(genes)} genes for scoring.",
                stacklevel=2,
            )
        if not found:
            scores[state] = np.zeros(adata.n_obs, dtype=np.float32)
            continue

        # Mean expression of found marker genes — O(1) per gene via dict
        gene_indices = [var_to_idx[g] for g in found]
        scores[state] = lognorm[:, gene_indices].mean(axis=1)

    return pd.DataFrame(scores, index=adata.obs_names)


def assign_states(score_df: pd.DataFrame) -> pd.Series:
    """Assign each cell to the state with the highest marker score.

    Parameters
    ----------
    score_df : pd.DataFrame
        Cells x states score matrix from score_marker_states.

    Returns
    -------
    pd.Series
        State assignment per cell. Ties broken by column order (first state wins).
    """
    return score_df.idxmax(axis=1).rename("state")


def score_cell_cycle(
    adata: ad.AnnData,
    s_genes: list[str] | None = None,
    g2m_genes: list[str] | None = None,
) -> pd.DataFrame:
    """Score cell-cycle activity per cell.

    MUST use adata.obsm['lognorm_full'] (full gene set), NOT layers['lognorm'].
    Cell-cycle genes (Pcna, Mcm2, Top2a, Ccnb1) are NOT in the top 2000 HVGs.
    Using lognorm would produce exactly zero scores -- silent corruption.

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['lognorm_full'] and adata.uns['lognorm_full_genes'].
    s_genes : list[str], optional
        S-phase genes. Defaults to CELL_CYCLE_S_GENES.
    g2m_genes : list[str], optional
        G2M-phase genes. Defaults to CELL_CYCLE_G2M_GENES.

    Returns
    -------
    pd.DataFrame
        Columns: s_score, g2m_score, cycling_score (sum of s + g2m).
        Index = cell barcodes.
    """
    if s_genes is None:
        s_genes = CELL_CYCLE_S_GENES
    if g2m_genes is None:
        g2m_genes = CELL_CYCLE_G2M_GENES

    lognorm_full, gene_names = _get_lognorm_full(adata)

    s_score = _mean_gene_score(lognorm_full, gene_names, s_genes, "S-phase")
    g2m_score = _mean_gene_score(lognorm_full, gene_names, g2m_genes, "G2M-phase")

    return pd.DataFrame(
        {
            "s_score": s_score,
            "g2m_score": g2m_score,
            "cycling_score": s_score + g2m_score,
        },
        index=adata.obs_names,
    )


def score_apoptosis(
    adata: ad.AnnData,
    pro_genes: list[str] | None = None,
    anti_genes: list[str] | None = None,
) -> pd.DataFrame:
    """Score apoptosis activity per cell.

    MUST use adata.obsm['lognorm_full'] (full gene set), NOT layers['lognorm'].

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['lognorm_full'] and adata.uns['lognorm_full_genes'].
    pro_genes : list[str], optional
        Pro-apoptotic genes. Defaults to APOPTOSIS_PRO_GENES.
        Bad is PRO-apoptotic (inhibits Bcl2/Bcl-xL).
    anti_genes : list[str], optional
        Anti-apoptotic genes. Defaults to APOPTOSIS_ANTI_GENES.

    Returns
    -------
    pd.DataFrame
        Columns: pro_score, anti_score, net_apoptotic_score (pro - anti).
        Index = cell barcodes.
    """
    if pro_genes is None:
        pro_genes = APOPTOSIS_PRO_GENES
    if anti_genes is None:
        anti_genes = APOPTOSIS_ANTI_GENES

    lognorm_full, gene_names = _get_lognorm_full(adata)

    pro_score = _mean_gene_score(lognorm_full, gene_names, pro_genes, "pro-apoptotic")
    anti_score = _mean_gene_score(lognorm_full, gene_names, anti_genes, "anti-apoptotic")

    return pd.DataFrame(
        {
            "pro_score": pro_score,
            "anti_score": anti_score,
            "net_apoptotic_score": pro_score - anti_score,
        },
        index=adata.obs_names,
    )


def calibrate_division_death_rates(
    adata: ad.AnnData,
    state_assignments: pd.Series,
    cycling_scores: pd.Series,
    apoptotic_scores: pd.Series,
    timepoint_col: str = "Time_point",
    time_unit_hours: float = 24.0,
) -> pd.DataFrame:
    """Calibrate division and death rates using population-dynamics approach.

    DO NOT use fraction-above-threshold × (1/cycle_length_hours). That approach
    produces death >> division for all states, making the ODE system ill-posed.

    Uses population-dynamics calibration:
      net_growth_rate = log(N_{t+1}/N_t) / delta_t_hours  (observed)
      signature_ratio = mean_cycling / mean_apoptotic      (relative signal)
      death_rate = net_growth / (signature_ratio - 1)
      division_rate = signature_ratio × death_rate

    Parameters
    ----------
    adata : AnnData
        Must have timepoint column in obs.
    state_assignments : pd.Series
        State per cell (from assign_states).
    cycling_scores : pd.Series
        Per-cell cycling score (from score_cell_cycle, 'cycling_score' column).
    apoptotic_scores : pd.Series
        Per-cell net apoptotic score (from score_apoptosis, 'net_apoptotic_score').
    timepoint_col : str
        Column in adata.obs with timepoint values (in days).
    time_unit_hours : float
        Hours per timepoint unit. Default 24.0 (timepoints are in days).

    Returns
    -------
    pd.DataFrame
        One row per state with columns: state, net_growth_rate, signature_ratio,
        division_rate, death_rate, net_shrinking.
    """
    timepoints = sorted(adata.obs[timepoint_col].unique())
    states = sorted(state_assignments.unique())

    results = []
    for state in states:
        state_mask = state_assignments == state

        # Population counts per timepoint
        counts = []
        for tp in timepoints:
            tp_mask = adata.obs[timepoint_col] == tp
            n = (state_mask & tp_mask).sum()
            counts.append(n)

        # Net growth rate: mean of log(N_{t+1}/N_t) / delta_t over intervals
        growth_rates = []
        for i in range(len(timepoints) - 1):
            n_t = max(counts[i], 1)  # Avoid log(0)
            n_t1 = max(counts[i + 1], 1)
            delta_t_hours = (timepoints[i + 1] - timepoints[i]) * time_unit_hours
            rate = np.log(n_t1 / n_t) / delta_t_hours
            growth_rates.append(rate)

        net_growth_rate = float(np.mean(growth_rates)) if growth_rates else 0.0

        # Signature ratio: mean cycling / mean apoptotic for cells in this state
        mean_cycling = float(cycling_scores[state_mask].mean())
        mean_apoptotic = float(apoptotic_scores[state_mask].mean())

        # Apoptotic score can be negative (anti > pro). Use absolute value for ratio.
        # ratio = cycling / |apoptotic| as relative signal strength.
        abs_apoptotic = abs(mean_apoptotic) if mean_apoptotic != 0 else 1e-10

        signature_ratio = mean_cycling / abs_apoptotic

        # Floor signature_ratio at 1.01 to avoid division by zero or negative death rate
        if signature_ratio <= 1.0:
            warnings.warn(
                f"State '{state}': signature_ratio={signature_ratio:.4f} <= 1.0, "
                f"flooring at 1.01. This means cycling ≤ apoptotic signal.",
                stacklevel=2,
            )
            signature_ratio = 1.01

        # Handle near-zero net growth
        if abs(net_growth_rate) < 1e-6:
            epsilon = 1e-6 * (1.0 if net_growth_rate >= 0 else -1.0)
            net_growth_rate = epsilon

        # Population-dynamics calibration
        death_rate = net_growth_rate / (signature_ratio - 1.0)

        # Clip negative death rate to 0
        net_shrinking = net_growth_rate < 0
        if death_rate < 0:
            warnings.warn(
                f"State '{state}': death_rate={death_rate:.6f} < 0, clipping to 0.",
                stacklevel=2,
            )
            death_rate = 0.0

        division_rate = signature_ratio * death_rate

        results.append({
            "state": state,
            "net_growth_rate": net_growth_rate,
            "signature_ratio": signature_ratio,
            "division_rate": division_rate,
            "death_rate": death_rate,
            "net_shrinking": net_shrinking,
        })

    return pd.DataFrame(results)


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_lognorm_full(adata: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    """Extract lognorm_full matrix and gene names from adata.

    Returns
    -------
    tuple
        (matrix as dense ndarray, list of gene names)

    Raises
    ------
    ValueError
        If lognorm_full is not present in adata.obsm.
    """
    if "lognorm_full" not in adata.obsm:
        raise ValueError(
            "adata.obsm['lognorm_full'] required for cell-cycle/apoptosis scoring. "
            "This contains the full gene set (not just HVGs). "
            "Run preprocess_standard first."
        )
    if "lognorm_full_genes" not in adata.uns:
        raise ValueError(
            "adata.uns['lognorm_full_genes'] required (gene names for lognorm_full)."
        )

    matrix = adata.obsm["lognorm_full"]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)

    gene_names = list(adata.uns["lognorm_full_genes"])
    return matrix, gene_names


def _mean_gene_score(
    matrix: np.ndarray,
    gene_names: list[str],
    query_genes: list[str],
    label: str,
) -> np.ndarray:
    """Compute mean expression of query genes across cells.

    Parameters
    ----------
    matrix : ndarray
        Cells x genes expression matrix.
    gene_names : list[str]
        Gene names corresponding to matrix columns.
    query_genes : list[str]
        Genes to score.
    label : str
        Label for warning messages.

    Returns
    -------
    ndarray
        Per-cell mean expression score.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}  # O(1) lookups
    found = [g for g in query_genes if g in gene_to_idx]
    missing = [g for g in query_genes if g not in gene_to_idx]
    if missing:
        warnings.warn(
            f"{label}: genes {missing} not in lognorm_full gene set. "
            f"Using {len(found)}/{len(query_genes)} genes.",
            stacklevel=2,
        )
    if not found:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    indices = [gene_to_idx[g] for g in found]
    return matrix[:, indices].mean(axis=1).astype(np.float32)
