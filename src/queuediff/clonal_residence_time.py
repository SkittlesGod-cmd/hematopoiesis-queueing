"""Three-rate decomposition model for estimating state residence times 
from Weinreb LARRY clonal barcode data.

Model: for each marker-defined state, net clone size change between 
timepoints decomposes into division rate, death rate, and net 
transition rate. Division and death are estimated independently via 
gene-signature scoring; transition rate is solved as the residual 
using observed clone trajectories as the constraint.

All functions document their explicit assumptions so they can be 
cited as stated limitations in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread, mmwrite
import gzip
from pathlib import Path
from typing import Dict, List, Optional

from queuediff.state_discretization import preprocess_standard


def _open_text(path: str | Path):
    """Open a (possibly gzipped) text file and return a file-like object."""
    path = Path(path)
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def compute_cell_cycle_score(
    adata: sc.AnnData,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    layer: Optional[str] = "lognorm",
) -> pd.Series:
    """Compute a per-cell cycling score using S-phase and G2M-phase gene sets.

    Uses the 4 cell-cycle marker genes confirmed present in this dataset:
    S-phase: Pcna, Mcm2
    G2M-phase: Top2a, Ccnb1

    The cycling score is the SUM of the S-phase score and G2M-phase score.
    Sum is used rather than max because a cell actively in EITHER phase
    is proliferating, and the additive combination preserves magnitude
    information from both phases (a cell with high S AND high G2M signal
    should score higher than one with only one high score).

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in the specified layer.
    s_genes
        S-phase marker genes. Defaults to ['Pcna', 'Mcm2'].
    g2m_genes
        G2M-phase marker genes. Defaults to ['Top2a', 'Ccnb1'].
    layer
        Layer in ``adata.layers`` containing log-normalised expression.
        If None, uses ``adata.X`` directly.

    Returns
    -------
    pandas.Series
        Cycling score per cell (indexed by cell barcode).
    """
    if s_genes is None:
        s_genes = ['Pcna', 'Mcm2']
    if g2m_genes is None:
        g2m_genes = ['Top2a', 'Ccnb1']

    # If layer is None, use adata.X directly; otherwise use the specified layer
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X_backup = adata.X
        adata.X = adata.layers[layer]
        restore = True
    else:
        restore = False

    try:
        # Score S-phase genes
        s_scores = {}
        for state, genes in [('S', s_genes), ('G2M', g2m_genes)]:
            col = f"_cycle_score_{state}"
            present = [g for g in genes if g in adata.var_names]
            if not present:
                # No genes present, score is 0
                s_scores[state] = np.zeros(adata.n_obs)
                continue
            sc.tl.score_genes(adata, gene_list=present, score_name=col, random_state=0, ctrl_as_ref=False)
            s_scores[state] = adata.obs[col].values.copy()
            del adata.obs[col]

        # Sum of S and G2M scores
        cycling_score = s_scores['S'] + s_scores['G2M']

    finally:
        if restore:
            adata.X = X_backup

    return pd.Series(cycling_score, index=adata.obs_names, name="cycling_score")


def compute_apoptotic_score(
    adata: sc.AnnData,
    pro_apoptotic: Optional[List[str]] = None,
    anti_apoptotic: Optional[List[str]] = None,
    layer: Optional[str] = "lognorm",
    n_bins: int = 25,
) -> pd.Series:
    """Compute a net apoptotic score per cell.

    Net score = pro-apoptotic gene-set score minus anti-apoptotic 
    gene-set score, both computed via sc.tl.score_genes on the 
    specified layer.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in the specified layer.
    pro_apoptotic
        Pro-apoptotic marker genes. Defaults to effectors + upstream 
        regulators: ['Casp3','Casp8','Casp9','Bax','Bak1','Cycs','Bad'].
    anti_apoptotic
        Anti-apoptotic marker genes. Defaults to ['Bcl2','Bcl2l1'].
    layer
        Layer in ``adata.layers`` containing log-normalised expression.
        If None, uses ``adata.X`` directly.
    n_bins
        Number of expression level bins for control gene sampling in 
        sc.tl.score_genes. Default 25 (scanpy default). Use smaller 
        values (e.g., 5) for small synthetic gene pools.

    Returns
    -------
    pandas.Series
        Net apoptotic score per cell (indexed by cell barcode).
    """
    if pro_apoptotic is None:
        pro_apoptotic = ['Casp3', 'Casp8', 'Casp9', 'Bax', 'Bak1', 'Cycs', 'Bad']
    if anti_apoptotic is None:
        anti_apoptotic = ['Bcl2', 'Bcl2l1']

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        use_layer = layer
    else:
        use_layer = None

    scores = {}
    for label, genes in [('pro', pro_apoptotic), ('anti', anti_apoptotic)]:
        col = f"_apoptosis_score_{label}"
        present = [g for g in genes if g in adata.var_names]
        if not present:
            scores[label] = np.zeros(adata.n_obs)
            continue
        # Use layer parameter directly in score_genes to avoid swapping adata.X
        sc.tl.score_genes(adata, gene_list=present, score_name=col, random_state=0,
                          ctrl_as_ref=False, layer=layer, use_raw=False, n_bins=n_bins)
        scores[label] = adata.obs[col].values.copy()
        del adata.obs[col]

    apoptosis_score = scores['pro'] - scores['anti']

    return pd.Series(apoptosis_score, index=adata.obs_names, name="apoptosis_score")


def estimate_division_rate_per_state(
    adata: sc.AnnData,
    state_col: str = "marker_state",
    cycling_score_col: str = "cycling_score",
    cycle_length_hours: float = 20.0,
    threshold: float = 0.0,
) -> pd.Series:
    """Estimate per-state division rate from cycling score.

    Assumptions (explicit for paper):
    - cycle_length_hours: time for one complete cell cycle (default 20h, 
      typical for mammalian hematopoietic progenitors)
    - threshold: cycling_score > threshold is considered "cycling" 
      (default 0.0, i.e. positive score = cycling)
    - Fraction of cycling cells = cells in cycle / cycle_length 
      gives instantaneous division rate (per hour)

    Parameters
    ----------
    adata
        AnnData with state assignments in ``adata.obs[state_col]`` and
        cycling scores in ``adata.obs[cycling_score_col]``.
    state_col
        Column in ``adata.obs`` with marker-based state assignments.
    cycling_score_col
        Column in ``adata.obs`` with cycling scores.
    cycle_length_hours
        Assumed cell cycle duration in hours. Default 20.0 
        (explicit assumption for paper).
    threshold
        Cycling score threshold for calling a cell "cycling". 
        Default 0.0 (positive score = cycling).

    Returns
    -------
    pandas.Series
        Division rate (per hour) indexed by state name.
    """
    if state_col not in adata.obs.columns:
        raise ValueError(f"State column '{state_col}' not found in adata.obs")
    if cycling_score_col not in adata.obs.columns:
        raise ValueError(f"Cycling score column '{cycling_score_col}' not found in adata.obs")

    cycling_scores = adata.obs[cycling_score_col].values
    states = adata.obs[state_col].values

    division_rates = {}
    for state in sorted(set(states)):
        mask = (states == state)
        if not mask.any():
            division_rates[state] = 0.0
            continue
        n_cycling = (cycling_scores[mask] > threshold).sum()
        n_total = mask.sum()
        frac_cycling = n_cycling / n_total
        division_rate = frac_cycling / cycle_length_hours
        division_rates[state] = division_rate

    return pd.Series(division_rates, name="division_rate_per_hour")


def estimate_death_rate_per_state(
    adata: sc.AnnData,
    state_col: str = "marker_state",
    apoptotic_score_col: str = "apoptosis_score",
    death_commitment_hours: float = 6.0,
    threshold: float = 0.0,
) -> pd.Series:
    """Estimate per-state death rate from apoptotic score.

    Assumptions (explicit for paper):
    - death_commitment_hours: time from apoptotic commitment to cell 
      clearance (default 6h, typical for caspase activation to 
      phagocytic clearance)
    - threshold: apoptotic_score > threshold is considered 
      "apoptotically committed" (default 0.0, positive net score = 
      pro-apoptotic dominance)
    - Fraction of apoptotic cells = cells committed / commitment_time 
      gives instantaneous death rate (per hour)

    Parameters
    ----------
    adata
        AnnData with state assignments and apoptotic scores.
    state_col
        Column in ``adata.obs`` with marker-based state assignments.
    apoptotic_score_col
        Column in ``adata.obs`` with net apoptotic scores.
    death_commitment_hours
        Assumed time from apoptotic commitment to clearance (hours).
        Default 6.0 (explicit assumption for paper).
    threshold
        Apoptotic score threshold. Default 0.0 (positive = pro-apoptotic).

    Returns
    -------
    pandas.Series
        Death rate (per hour) indexed by state name.
    """
    if state_col not in adata.obs.columns:
        raise ValueError(f"State column '{state_col}' not found in adata.obs")
    if apoptotic_score_col not in adata.obs.columns:
        raise ValueError(f"Apoptotic score column '{apoptotic_score_col}' not found in adata.obs")

    apoptotic_scores = adata.obs[apoptotic_score_col].values
    states = adata.obs[state_col].values

    death_rates = {}
    for state in sorted(set(states)):
        mask = (states == state)
        if not mask.any():
            death_rates[state] = 0.0
            continue
        n_apoptotic = (apoptotic_scores[mask] > threshold).sum()
        n_total = mask.sum()
        frac_apoptotic = n_apoptotic / n_total
        death_rate = frac_apoptotic / death_commitment_hours
        death_rates[state] = death_rate

    return pd.Series(death_rates, name="death_rate_per_hour")


def extract_clone_trajectories(
    adata: sc.AnnData,
    clone_matrix_path: str | Path,
    metadata_path: str | Path,
    state_col: str = "marker_state",
    time_col: str = "Time point",
) -> pd.DataFrame:
    """Load clone matrix and build per-clone, per-timepoint, per-state cell counts.

    The Weinreb clone matrix is cells x clones (130887 cells x 5864 clones).
    Each cell belongs to at most one clone (entry = 1 if cell belongs to clone).

    The row order of the clone matrix matches the row order of the metadata file
    (stateFate_inVitro_metadata.txt.gz). This function loads the metadata to
    get the cell barcodes, then matches them to adata.obs_names (which may be
    a filtered subset) by barcode.

    IMPORTANT: The Weinreb data has duplicate cell barcodes in the metadata.
    When AnnData is created, scanpy makes obs_names unique by appending '-1',
    '-2', etc. to duplicates. The clone matrix rows correspond to the ORIGINAL
    (pre-make_unique) barcodes. This function handles this by stripping the
    '-N' suffix from adata barcodes before matching.

    Parameters
    ----------
    adata
        AnnData with state and timepoint in .obs. Cell barcodes in adata.obs_names
        must match a subset of the metadata index (after stripping '-N' suffixes).
    clone_matrix_path
        Path to stateFate_inVitro_clone_matrix.mtx.gz (cells x clones).
    metadata_path
        Path to stateFate_inVitro_metadata.txt.gz. The index (cell barcodes)
        gives the row order of the clone matrix.
    state_col
        Column in ``adata.obs`` containing marker-based state assignments.
    time_col
        Column in ``adata.obs`` containing timepoint values.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns: clone_id, timepoint, state, n_cells.
    """
    # Load clone matrix (cells x clones)
    X = mmread(gzip.open(clone_matrix_path, "rt")).tocsr()

    # X is cells x clones (rows = cells, cols = clones)
    n_cells, n_clones = X.shape

    # Load metadata to get cell barcodes in clone matrix row order
    meta = pd.read_csv(_open_text(metadata_path), sep="\t", index_col=0)
    # metadata index is the cell barcodes in clone matrix row order
    clone_matrix_barcodes = meta.index.values

    # Match adata cells to clone matrix rows by barcode
    # adata.obs_names may have been made unique (e.g., 'd6_2_2-1'), so strip suffix
    adata_barcodes = adata.obs_names.values
    
    # Helper: strip trailing '-N' where N is a number (scanpy make_unique suffix)
    def _base_barcode(bc: str) -> str:
        # Find last '-' followed by digits at end of string
        import re
        match = re.match(r'^(.+?)-\d+$', bc)
        return match.group(1) if match else bc
    
    adata_base_barcodes = np.array([_base_barcode(bc) for bc in adata_barcodes])

    # Build lookup: base barcode -> list of indices in clone matrix (handles duplicates)
    base_bc_to_indices = {}
    for i, bc in enumerate(clone_matrix_barcodes):
        base_bc_to_indices.setdefault(bc, []).append(i)

    # For each adata cell, find its corresponding clone matrix row
    matched_indices = []
    for i, base_bc in enumerate(adata_base_barcodes):
        if base_bc in base_bc_to_indices and base_bc_to_indices[base_bc]:
            # Pop the first available index for this base barcode
            clone_idx = base_bc_to_indices[base_bc].pop(0)
            matched_indices.append(clone_idx)
        else:
            raise ValueError(
                f"Cell {adata_barcodes[i]} (base barcode '{base_bc}') not found "
                f"in clone matrix metadata, or all matching rows already used"
            )

    # Get state and timepoint per matched cell (row order now matches adata)
    states = adata.obs[state_col].values
    timepoints = adata.obs[time_col].values

    # For each cell in adata, find its clone using the matched clone matrix row
    rows = []
    for adata_idx, clone_idx in enumerate(matched_indices):
        clone_indices = X[clone_idx].nonzero()[1]  # non-zero columns = clone IDs
        if len(clone_indices) == 1:
            clone_id = clone_indices[0]
            rows.append({
                'clone_id': clone_id,
                'timepoint': timepoints[adata_idx],
                'state': states[adata_idx],
                'n_cells': 1
            })
        elif len(clone_indices) > 1:
            # Should not happen per data structure, but handle gracefully
            for clone_id in clone_indices:
                rows.append({
                    'clone_id': clone_id,
                    'timepoint': timepoints[adata_idx],
                    'state': states[adata_idx],
                    'n_cells': 1
                })
        # cells with 0 clones are ignored

    df = pd.DataFrame(rows)
    # Aggregate to counts
    df = df.groupby(['clone_id', 'timepoint', 'state']).size().reset_index(name='n_cells')

    return df


def solve_transition_rate(
    clone_trajectories: pd.DataFrame,
    division_rates: pd.Series,
    death_rates: pd.Series,
    state_col: str = "state",
) -> pd.DataFrame:
    """Solve for net transition rate as the residual of observed growth.

    For each state and consecutive timepoint pair:
    observed_growth_rate = division_rate - death_rate - transition_rate
    => transition_rate = division_rate - death_rate - observed_growth_rate

    The observed growth rate is computed from clone trajectories:
    log(n_cells[t+1] / n_cells[t]) / delta_t

    LIMITATION (explicit for paper):
    This measures NET transition (outflow - inflow). It accounts for 
    inflow from upstream states only implicitly through the residual. 
    A fully general model would need the full state-transition graph 
    solved jointly, not state-by-state. This simplification means 
    transition rates can be negative (net inflow > outflow) and should 
    be interpreted as net rates, not pure outflows. Revisiting this 
    with a full graph-based model is a stated future improvement.

    Parameters
    ----------
    clone_trajectories
        DataFrame from ``extract_clone_trajectories`` with columns
        clone_id, timepoint, state, n_cells.
    division_rates
        Series from ``estimate_division_rate_per_state`` (per hour).
    death_rates
        Series from ``estimate_death_rate_per_state`` (per hour).
    state_col
        Column name for state in clone_trajectories.

    Returns
    -------
    pandas.DataFrame
        Columns: state, timepoint_start, timepoint_end, delta_t_hours, 
        division_rate, death_rate, observed_growth_rate, transition_rate.
    """
    # Timepoints in hours (assuming days -> hours * 24)
    timepoints = sorted(clone_trajectories['timepoint'].unique())
    if len(timepoints) < 2:
        raise ValueError("Need at least 2 timepoints to compute growth rates")

    results = []

    for state in sorted(clone_trajectories[state_col].unique()):
        state_data = clone_trajectories[clone_trajectories[state_col] == state]

        for i in range(len(timepoints) - 1):
            t1 = timepoints[i]
            t2 = timepoints[i + 1]
            delta_t_days = t2 - t1
            delta_t_hours = delta_t_days * 24.0

            # Aggregate clone counts per timepoint
            n_t1 = state_data[state_data['timepoint'] == t1]['n_cells'].sum()
            n_t2 = state_data[state_data['timepoint'] == t2]['n_cells'].sum()

            if n_t1 == 0:
                observed_growth_rate = np.nan
            else:
                observed_growth_rate = np.log(n_t2 / n_t1) / delta_t_hours

            div_rate = division_rates.get(state, 0.0)
            death_rate = death_rates.get(state, 0.0)
            transition_rate = div_rate - death_rate - observed_growth_rate

            results.append({
                'state': state,
                'timepoint_start': t1,
                'timepoint_end': t2,
                'delta_t_hours': delta_t_hours,
                'division_rate': div_rate,
                'death_rate': death_rate,
                'observed_growth_rate': observed_growth_rate,
                'transition_rate': transition_rate,
            })

    return pd.DataFrame(results)


def summarize_transition_rates(transition_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize transition rates per state by taking mean absolute value."""
    summary = transition_df.groupby('state')['transition_rate'].apply(
        lambda x: np.nanmean(np.abs(x)) if not x.isna().all() else np.nan
    ).reset_index()
    summary.columns = ['state', 'mean_abs_transition_rate_per_hour']
    summary['mean_residence_time_hours'] = 1.0 / summary['mean_abs_transition_rate_per_hour'].abs()
    return summary


def estimate_residence_times(
    adata: sc.AnnData,
    clone_matrix_path: str | Path,
    metadata_path: str | Path,
    state_col: str = "marker_state",
    time_col: str = "Time point",
    cycle_length_hours: float = 20.0,
    death_commitment_hours: float = 6.0,
    cycling_threshold: float = 0.0,
    apoptotic_threshold: float = 0.0,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    already_preprocessed: bool = False,
) -> pd.DataFrame:
    """Top-level orchestrator: runs the full three-rate decomposition.

    Runs steps 1-6 in sequence:
    1. Compute cycling scores (S + G2M) on RAW data (before HVG filtering)
    2. Compute apoptotic scores (pro - anti) on log-normalized data
    3. Estimate division rates per state
    4. Estimate death rates per state
    5. Extract clone trajectories
    6. Solve transition rates as residuals

    Returns a per-state DataFrame with mean residence time.

    Parameters
    ----------
    adata
        Raw AnnData with count data (NOT yet preprocessed), OR already
        preprocessed AnnData with cycling_score in adata.obs and lognorm layer.
    clone_matrix_path
        Path to clone matrix (.mtx.gz).
    metadata_path
        Path to metadata TSV (used to align clone matrix rows to cell barcodes).
    state_col
        State column name in adata.obs.
    time_col
        Timepoint column name in adata.obs.
    cycle_length_hours
        Cell cycle duration assumption (hours).
    death_commitment_hours
        Apoptotic commitment to clearance time (hours).
    cycling_threshold
        Cycling score threshold.
    apoptotic_threshold
        Apoptotic score threshold.
    n_top_genes
        Number of HVGs to select (if not already preprocessed).
    n_pcs
        Number of PCA components (if not already preprocessed).
    already_preprocessed
        If True, skip preprocessing (assumes cycling_score in adata.obs
        and lognorm layer exist). If False, run full preprocessing.

    Returns
    -------
    pandas.DataFrame
        Columns: state, division_rate_per_hour, death_rate_per_hour,
        transition_rate_per_hour (mean across intervals),
        mean_residence_time_hours (= 1 / |transition_rate|).
    """
    # Step 1: Preprocessing (includes cell cycle scoring)
    # The cell cycle genes (Pcna, Mcm2, Top2a, Ccnb1) are often filtered out
    # by HVG selection, so we must score them before HVG filtering.
    if not already_preprocessed:
        print("Running standard preprocessing (with cell cycle scoring)...")
        adata = preprocess_standard(
            adata,
            min_genes=200,
            min_cells=3,
            n_top_genes=2000,
            n_pcs=30,
            already_normalized=True,  # data is already library-size normalized (normed_counts)
        )
    else:
        print("Data already preprocessed, skipping preprocessing step...")

    # The cycling_score is now pre-computed in adata.obs by preprocess_standard
    # Step 2: Compute apoptotic scores using the lognorm layer
    print("Computing apoptotic scores...")
    adata.obs["apoptosis_score"] = compute_apoptotic_score(adata, layer="lognorm")

    # Step 3: Division rates
    print("Estimating division rates...")
    division_rates = estimate_division_rate_per_state(
        adata, state_col=state_col, cycling_score_col="cycling_score",
        cycle_length_hours=cycle_length_hours, threshold=cycling_threshold
    )

    # Step 4: Death rates
    print("Estimating death rates...")
    death_rates = estimate_death_rate_per_state(
        adata, state_col=state_col, apoptotic_score_col="apoptosis_score",
        death_commitment_hours=death_commitment_hours, threshold=apoptotic_threshold
    )

    # Step 5: Clone trajectories
    print("Extracting clone trajectories...")
    clone_trajectories = extract_clone_trajectories(
        adata, clone_matrix_path, metadata_path, state_col=state_col
    )

    # Step 6: Transition rates
    print("Solving transition rates...")
    transition_df = solve_transition_rate(
        clone_trajectories, division_rates, death_rates, state_col="state"
    )

    # Aggregate to per-state summary
    # Mean absolute transition rate per state (across intervals)
    mean_transition = transition_df.groupby('state')['transition_rate'].apply(
        lambda x: np.nanmean(np.abs(x)) if not x.isna().all() else np.nan
    )

    result = pd.DataFrame({
        'division_rate_per_hour': division_rates,
        'death_rate_per_hour': death_rates,
        'transition_rate_per_hour': mean_transition,
    })

    # Mean residence time = 1 / |transition_rate| (queueing theory)
    result['mean_residence_time_hours'] = 1.0 / result['transition_rate_per_hour'].abs()
    result = result.fillna(np.inf)

    return result