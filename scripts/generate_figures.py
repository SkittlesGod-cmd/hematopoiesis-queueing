"""
Generate all publication figures from saved pipeline results.

Each ``plot_*`` function reads pre-computed CSV(s) from ``results/`` and
writes both a PNG (for rapid iteration) and a PDF (for the manuscript) to
``results/figures/``.

Run from the project root::

    python scripts/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def _save(fig: plt.Figure, name: str) -> None:
    """Write *fig* as both PNG and PDF to *FIGURES_DIR*."""
    for ext in (".png", ".pdf"):
        path = FIGURES_DIR / f"{name}{ext}"
        fig.savefig(path)
        print(f"  Saved {path}")


# ------------------------------------------------------------------
# Individual figure functions
# ------------------------------------------------------------------


def plot_residence_time_distributions() -> None:
    """
    Figure 1 — Per-state fitted gamma vs. exponential residence-time overlay.

    Data source
    -----------
    - ``results/weinreb_clonal_residence_times.csv``
        Columns: ``state``, ``residence_time``.

    - ``results/weinreb_distribution_fits.csv``
        Columns: ``state``, ``gamma_shape``, ``gamma_scale``, ``exp_rate``.

    What to show
    ------------
    One subplot per differentiation state (HSC, MPP, CMP, LMPP, MEP, GMP).

    Each subplot:
    - Histogram (density-scaled) of observed clonal residence times.
    - Overlaid PDF of the fitted gamma distribution (dashed, coloured).
    - Overlaid PDF of the fitted exponential distribution (dotted, grey).

    Axes:
    - x = residence time (days), y = density.
    - Legend identifying the two fitted curves.
    - Title = state name, with ``n = <n_obs>`` in the corner.

    Layout
    ------
    2 rows × 3 columns.  Shared y-axis label on the left column, shared
    x-axis label below the bottom row.

    Output
    ------
    ``results/figures/fig1_residence_time_distributions.{png,pdf}``
    """
    raise NotImplementedError(
        "Implement this function when distribution_fitting is ready.\n"
        "Expected inputs:\n"
        "  results/weinreb_clonal_residence_times.csv\n"
        "  results/weinreb_distribution_fits.csv\n"
        "See docstring for layout details."
    )


def plot_aic_bic_comparison() -> None:
    """
    Figure 2 — Bar chart comparing AIC / BIC for gamma vs. exponential per state.

    Data source
    -----------
    - ``results/weinreb_model_comparison.csv``
        Columns: ``state``, ``gamma_aic``, ``exp_aic``, ``gamma_bic``,
                 ``exp_bic``, ``delta_aic``, ``delta_bic``, ``lr_pvalue``,
                 ``q_value_bh``, ``preferred_model``.

    What to show
    ------------
    A grouped bar chart with two panels side-by-side:

    **Left panel (ΔAIC)**
    - x-axis: state (categorical).
    - y-axis: ΔAIC = gamma_AIC − exp_AIC.
    - Horizontal dashed line at zero.
    - Bars coloured by sign (green if ΔAIC < 0 → gamma preferred;
      red if ΔAIC > 0 → exponential preferred).
    - Annotation: number of asterisks for BH-significant states
      (``*`` if ``q_value_bh < 0.05``, ``**`` if ``< 0.01``).

    **Right panel (ΔBIC)** — same layout.

    Layout
    ------
    1 row × 2 columns.  Shared x-axis label ``"Differentiation state"``.
    Suptitle or figure caption: ``"Semi-Markov vs. Markov model comparison"``.

    Output
    ------
    ``results/figures/fig2_aic_bic_comparison.{png,pdf}``
    """
    raise NotImplementedError(
        "Implement this function when model_comparison is ready.\n"
        "Expected input:\n"
        "  results/weinreb_model_comparison.csv\n"
        "See docstring for layout details."
    )


def plot_bottleneck_ranking() -> None:
    """
    Figure 3 — Ranked bar chart of traffic intensity (ρ) per differentiation stage.

    Data source
    -----------
    - ``results/weinreb_queueing_summary.csv``
        Columns: ``state``, ``service_rate``, ``servers``, ``arrival_rate``,
                 ``traffic_intensity``.

    - ``results/weinreb_bottlenecks.csv``
        Columns: same as queueing_summary plus ``is_bottleneck``, ``severity``.

    What to show
    ------------
    A single horizontal bar chart.

    - y-axis: states sorted by traffic_intensity descending (highest ρ at top).
    - x-axis: traffic intensity (ρ).
    - Each bar coloured by ``severity`` category (low = green, moderate =
      yellow, high = orange, critical = dark orange, overloaded = red).
    - Vertical dashed line at ρ = 0.8 (the bottleneck threshold), with a
      ``"bottleneck threshold"`` label.
    - Bar labels on the right: ``"λ = {arrival_rate:.2f}, μ = {service_rate:.2f}"``.

    Overload critical states (ρ ≥ 1.0) with a hatch pattern for emphasis.

    Output
    ------
    ``results/figures/fig3_bottleneck_ranking.{png,pdf}``
    """
    raise NotImplementedError(
        "Implement this function when queueing_network and "
        "bottleneck_diagnostics are ready.\n"
        "Expected inputs:\n"
        "  results/weinreb_queueing_summary.csv\n"
        "  results/weinreb_bottlenecks.csv\n"
        "See docstring for layout details."
    )


def plot_synthetic_recovery_accuracy() -> None:
    """
    Figure 4 — Recovery accuracy vs. bottleneck severity.

    Data source
    -----------
    - ``results/synthetic_sweep/sweep_results.csv``
        Columns: ``true_bottleneck``, ``true_severity``, ``replicate``,
                 ``inferred_bottleneck``, ``inferred_rank_of_true``.

    What to show
    ------------
    A two-panel figure:

    **Left panel: Top-1 recovery rate vs. severity**
    - x-axis: bottleneck severity factor (log scale, 1–10).
    - y-axis: top-1 recovery rate (proportion, 0–1).
    - Main line: mean recovery rate across all bottleneck states at each
      severity, with error bars (±1 SD) across replicates.
    - Overlay faint individual lines per bottleneck state (same data,
      grouped by state) to show state-to-state variability.

    **Right panel: Confusion matrix at the highest severity (e.g., 10×)**
    - Rows = true bottleneck state.
    - Columns = inferred bottleneck state.
    - Annotated heatmap with values.

    Layout
    ------
    1 row × 2 columns, shared figure width ≈ 10 in.

    Output
    ------
    ``results/figures/fig4_synthetic_recovery_accuracy.{png,pdf}``
    """
    raise NotImplementedError(
        "Implement this function when recovery_validation is ready.\n"
        "Expected input:\n"
        "  results/synthetic_sweep/sweep_results.csv\n"
        "See docstring for layout details."
    )


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------


def main() -> None:
    """Generate all figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    functions = [
        ("fig1_residence_time_distributions", plot_residence_time_distributions),
        ("fig2_aic_bic_comparison", plot_aic_bic_comparison),
        ("fig3_bottleneck_ranking", plot_bottleneck_ranking),
        ("fig4_synthetic_recovery_accuracy", plot_synthetic_recovery_accuracy),
    ]

    for name, func in functions:
        print(f"[{name}]")
        try:
            func()
        except NotImplementedError as e:
            print(f"  SKIPPED — {e}")
        except FileNotFoundError as e:
            print(f"  SKIPPED — missing input data: {e}")
        except Exception as e:
            print(f"  ERROR  — {type(e).__name__}: {e}")
        print()

    print("All figures processed.")


if __name__ == "__main__":
    main()
