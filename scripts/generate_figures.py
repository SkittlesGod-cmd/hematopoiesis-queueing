"""Generate publication-quality figures for the queuediff paper.

All figure functions fully implemented -- no NotImplementedError stubs.
Produces figures for:
  1. State discretization (UMAP colored by state)
  2. Residence time distributions with fitted gamma/exponential
  3. Model comparison summary (AIC differences)
  4. Traffic intensity ranking (bottleneck bar chart)
  5. Queueing network topology
  6. Synthetic recovery validation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gamma as gamma_dist, expon as expon_dist


# Consistent color palette for hematopoietic states
STATE_COLORS = {
    "HSC": "#1f77b4",
    "MPP": "#ff7f0e",
    "LMPP": "#2ca02c",
    "CMP": "#d62728",
    "MEP": "#9467bd",
    "GMP": "#8c564b",
}


def setup_style():
    """Set publication figure style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_state_distribution(
    state_assignments: pd.Series,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of cell counts per state.

    Parameters
    ----------
    state_assignments : pd.Series
        State per cell.
    output_path : Path, optional
        If provided, saves figure to this path.

    Returns
    -------
    Figure
    """
    setup_style()
    counts = state_assignments.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [STATE_COLORS.get(s, "#666666") for s in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Hematopoietic State")
    ax.set_ylabel("Number of Cells")
    ax.set_title("State Assignment Distribution")

    for i, (state, count) in enumerate(counts.items()):
        ax.text(i, count + counts.max() * 0.02, str(count),
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_residence_time_distributions(
    residence_times: dict[str, np.ndarray],
    model_comparison: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Histogram of residence times with fitted gamma/exponential overlays.

    Parameters
    ----------
    residence_times : dict[str, ndarray]
        State -> residence time arrays (hours).
    model_comparison : pd.DataFrame
        From apply_fdr_correction (has gamma_shape, gamma_scale, etc.).
    output_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    setup_style()
    states = sorted(residence_times.keys())
    n_states = len(states)
    n_cols = min(3, n_states)
    n_rows = (n_states + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_states == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, state in enumerate(states):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        times = residence_times[state]

        # Histogram
        ax.hist(times, bins=30, density=True, alpha=0.6,
                color=STATE_COLORS.get(state, "#666666"), label="Data")

        # Fitted gamma overlay
        state_row = model_comparison[model_comparison["state"] == state]
        if not state_row.empty:
            row_data = state_row.iloc[0]
            x = np.linspace(0, times.max(), 200)

            # Gamma fit
            shape = row_data.get("gamma_shape", 1.0)
            scale = row_data.get("gamma_scale", times.mean())
            y_gamma = gamma_dist.pdf(x, a=shape, scale=scale)
            ax.plot(x, y_gamma, "r-", linewidth=2, label=f"Gamma (k={shape:.1f})")

            # Exponential fit
            exp_scale = row_data.get("exp_scale", times.mean())
            y_exp = expon_dist.pdf(x, scale=exp_scale)
            ax.plot(x, y_exp, "b--", linewidth=1.5, label="Exponential")

        ax.set_xlabel("Residence Time (hours)")
        ax.set_ylabel("Density")
        ax.set_title(state)
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_states, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle("Service Time Distributions by State", fontsize=13, y=1.02)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_model_comparison(
    model_comparison: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of ΔAIC values with significance markers.

    Parameters
    ----------
    model_comparison : pd.DataFrame
        From apply_fdr_correction.
    output_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    setup_style()
    df = model_comparison.sort_values("delta_aic", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [STATE_COLORS.get(s, "#666666") for s in df["state"]]
    bars = ax.bar(df["state"], df["delta_aic"], color=colors,
                  edgecolor="black", linewidth=0.5)

    # Significance threshold line
    ax.axhline(y=2, color="gray", linestyle="--", linewidth=1, label="ΔAIC = 2 threshold")

    # Mark gamma-preferred
    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("gamma_preferred", False):
            ax.text(i, row["delta_aic"] + df["delta_aic"].max() * 0.02, "★",
                    ha="center", fontsize=12, color="gold")

    ax.set_xlabel("Hematopoietic State")
    ax.set_ylabel("ΔAIC (Exponential − Gamma)")
    ax.set_title("Model Comparison: Gamma vs Exponential")
    ax.legend()
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_traffic_intensity(
    ranking: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of traffic intensity ranking.

    Parameters
    ----------
    ranking : pd.DataFrame
        From compute_bottleneck_ranking.
    output_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    setup_style()
    df = ranking.sort_values("traffic_intensity", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [STATE_COLORS.get(s, "#666666") for s in df["state"]]
    bars = ax.barh(df["state"], df["traffic_intensity"], color=colors,
                   edgecolor="black", linewidth=0.5)

    # Highlight primary bottleneck
    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("is_primary_bottleneck", False):
            ax.barh(row["state"], row["traffic_intensity"],
                    color="none", edgecolor="red", linewidth=2.5)

    ax.set_xlabel("Traffic Intensity (ρ)")
    ax.set_title("Bottleneck Ranking by Traffic Intensity")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_network_topology(
    routing_probs: dict[str, dict[str, float]],
    service_rates: dict[str, float] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Network diagram showing routing topology.

    Parameters
    ----------
    routing_probs : dict
        Source -> {target: probability}.
    service_rates : dict, optional
        State -> service rate for node labels.
    output_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    setup_style()
    import networkx as nx

    G = nx.DiGraph()
    all_states = set()
    for src, targets in routing_probs.items():
        all_states.add(src)
        for tgt, prob in targets.items():
            all_states.add(tgt)
            G.add_edge(src, tgt, weight=prob)

    for s in all_states:
        if s not in G:
            G.add_node(s)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)

    # Draw nodes
    node_colors = [STATE_COLORS.get(n, "#666666") for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800,
                          node_color=node_colors, alpha=0.8)

    # Draw edges with probability labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=20,
                          edge_color="gray", width=2, alpha=0.7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    # Node labels
    labels = {}
    for n in G.nodes:
        if service_rates and n in service_rates:
            labels[n] = f"{n}\nμ={service_rates[n]:.3f}"
        else:
            labels[n] = n
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9)

    ax.set_title("Queueing Network Topology")
    ax.axis("off")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_recovery_validation(
    recovery_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of true vs fitted gamma shape parameters.

    Parameters
    ----------
    recovery_df : pd.DataFrame
        From validate_parameter_recovery.
    output_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Shape recovery
    colors = [STATE_COLORS.get(s, "#666666") for s in recovery_df["state"]]
    ax1.scatter(recovery_df["true_shape"], recovery_df["fitted_shape"],
               c=colors, s=80, edgecolors="black", linewidth=0.5)
    lim = max(recovery_df["true_shape"].max(), recovery_df["fitted_shape"].max()) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("True Gamma Shape (k)")
    ax1.set_ylabel("Fitted Gamma Shape (k)")
    ax1.set_title("Shape Parameter Recovery")

    for _, row in recovery_df.iterrows():
        ax1.annotate(row["state"], (row["true_shape"], row["fitted_shape"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Mean recovery
    ax2.scatter(recovery_df["true_mean"], recovery_df["fitted_mean"],
               c=colors, s=80, edgecolors="black", linewidth=0.5)
    lim = max(recovery_df["true_mean"].max(), recovery_df["fitted_mean"].max()) * 1.1
    ax2.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("True Mean Residence Time (h)")
    ax2.set_ylabel("Fitted Mean Residence Time (h)")
    ax2.set_title("Mean Residence Time Recovery")

    for _, row in recovery_df.iterrows():
        ax2.annotate(row["state"], (row["true_mean"], row["fitted_mean"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def generate_all_figures(
    results: dict,
    output_dir: str | Path,
) -> list[Path]:
    """Generate all publication figures from pipeline results.

    Parameters
    ----------
    results : dict
        Output from run_pipeline_weinreb.run_pipeline.
    output_dir : str or Path
        Directory to save figures.

    Returns
    -------
    list[Path]
        Paths to generated figures.
    """
    setup_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Figure 1: State distribution
    if "state_assignments" in results:
        path = output_dir / "fig1_state_distribution.png"
        plot_state_distribution(results["state_assignments"], path)
        generated.append(path)
        plt.close()

    # Figure 2: Residence time distributions
    if "residence_times" in results and "model_comparison" in results:
        path = output_dir / "fig2_residence_distributions.png"
        plot_residence_time_distributions(
            results["residence_times"], results["model_comparison"], path
        )
        generated.append(path)
        plt.close()

    # Figure 3: Model comparison
    if "model_comparison" in results:
        path = output_dir / "fig3_model_comparison.png"
        plot_model_comparison(results["model_comparison"], path)
        generated.append(path)
        plt.close()

    # Figure 4: Traffic intensity
    if "bottleneck_ranking" in results:
        path = output_dir / "fig4_traffic_intensity.png"
        plot_traffic_intensity(results["bottleneck_ranking"], path)
        generated.append(path)
        plt.close()

    # Figure 5: Network topology
    if "routing_probabilities" in results:
        service_rates = None
        if "residence_summary" in results:
            service_rates = {
                row["state"]: 1.0 / row["mean_hours"]
                for _, row in results["residence_summary"].iterrows()
            }
        path = output_dir / "fig5_network_topology.png"
        plot_network_topology(results["routing_probabilities"], service_rates, path)
        generated.append(path)
        plt.close()

    # Figure 6: Recovery validation (from synthetic sweep)
    from queuediff.synthetic_generator import default_hematopoiesis_params
    from queuediff.recovery_validation import validate_parameter_recovery

    params = default_hematopoiesis_params()
    recovery = validate_parameter_recovery(params, n_samples=1000,
                                           rng=np.random.default_rng(42))
    path = output_dir / "fig6_recovery_validation.png"
    plot_recovery_validation(recovery, path)
    generated.append(path)
    plt.close()

    print(f"Generated {len(generated)} figures in {output_dir}")
    return generated


if __name__ == "__main__":
    import json

    # Generate figures from saved results if available
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    figures_dir = results_dir / "figures"

    # Load saved results
    print("Generating figures from saved results...")

    results = {}
    # Load CSVs back if available
    csv_files = {
        "residence_summary": "residence_times.csv",
        "model_comparison": "model_comparison.csv",
        "bottleneck_ranking": "bottleneck_ranking.csv",
    }
    for key, filename in csv_files.items():
        path = results_dir / filename
        if path.exists():
            results[key] = pd.read_csv(path)

    # Load state assignments (for fig 1)
    state_assignments_path = results_dir / "state_assignments.csv"
    if state_assignments_path.exists():
        state_df = pd.read_csv(state_assignments_path, index_col=0)
        results["state_assignments"] = state_df["state"]

    # Load residence times as dict of arrays (for fig 2)
    residence_times_path = results_dir / "residence_times.json"
    if residence_times_path.exists():
        with open(residence_times_path) as f:
            rt_json = json.load(f)
        results["residence_times"] = {k: np.array(v) for k, v in rt_json.items()}

    # Load routing probabilities (for fig 5)
    routing_path = results_dir / "routing_probabilities.json"
    if routing_path.exists():
        with open(routing_path) as f:
            results["routing_probabilities"] = json.load(f)

    figures_dir.mkdir(parents=True, exist_ok=True)
    generate_all_figures(results, figures_dir)
    print("Done.")
