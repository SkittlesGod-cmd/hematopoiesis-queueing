"""Run structural cross-check on Nestorowa et al. 2016 data.

Single-timepoint dataset used ONLY for structural validation,
NOT for quantitative bottleneck analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_nestorowa_crosscheck(
    nestorowa_dir: str | Path,
    weinreb_assignments: pd.Series,
    output_dir: str | Path,
    verbose: bool = True,
) -> dict:
    """Run structural cross-check using Nestorowa data.

    Parameters
    ----------
    nestorowa_dir : str or Path
        Directory containing Nestorowa data files.
    weinreb_assignments : pd.Series
        State assignments from the Weinreb pipeline.
    output_dir : str or Path
        Directory for output files.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Cross-check results.
    """
    from queuediff.data_loading import load_nestorowa, preprocess_standard
    from queuediff.schema_mapping import NESTOROWA_SCHEMA, validate_schema
    from queuediff.structural_crosscheck import (
        crosscheck_state_structure,
        format_crosscheck_report,
    )

    nestorowa_dir = Path(nestorowa_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Loading Nestorowa data...")

    try:
        adata = load_nestorowa(nestorowa_dir)
        if verbose:
            print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Validate schema before preprocessing
        schema_issues = validate_schema(adata, NESTOROWA_SCHEMA)
        if schema_issues and verbose:
            print(f"  Schema warnings: {schema_issues}")

        # Preprocess (not already normalized)
        adata = preprocess_standard(adata, already_normalized=False)
        if verbose:
            print(f"  After preprocessing: {adata.shape[0]} cells × {adata.shape[1]} HVGs")

        # Cross-check
        result = crosscheck_state_structure(
            weinreb_assignments, adata,
            primary_name="Weinreb", secondary_name="Nestorowa",
        )

        report = format_crosscheck_report(result)
        if verbose:
            print(report)

        with open(output_dir / "nestorowa_crosscheck.txt", "w") as f:
            f.write(report)

        return result

    except FileNotFoundError:
        if verbose:
            print("  Nestorowa data not available. Skipping cross-check.")
            print("  (This is expected if download_nestorowa.py has not been run.)")
        return {"error": "Data not available"}


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    nestorowa_dir = script_dir / "data" / "raw" / "nestorowa"
    output_dir = script_dir.parent / "results"

    # Load state assignments from Weinreb pipeline output
    assignments_file = output_dir / "state_assignments.csv"
    if not assignments_file.exists():
        print(f"State assignments not found at {assignments_file}")
        print("Run 'python3 scripts/run_pipeline_weinreb.py' first.")
        sys.exit(1)

    print("Loading Weinreb state assignments...")
    state_assignments_df = pd.read_csv(assignments_file, index_col=0)
    state_assignments = state_assignments_df["state"]
    print(f"  Loaded {len(state_assignments)} cell assignments")

    run_nestorowa_crosscheck(nestorowa_dir, state_assignments, output_dir)
