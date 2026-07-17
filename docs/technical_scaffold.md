# Project Technical Scaffold

## Language
Python 3.11+

## Core Libraries / Tools

| Library | Purpose |
|---|---|
| `scanpy` | Single-cell data handling (`AnnData` objects), preprocessing, clustering |
| `anndata` | Underlying data structure `scanpy` is built on |
| `leidenalg` + `python-igraph` | Unsupervised clustering (Leiden algorithm), used for the state-schema cross-check |
| `lifelines` | Survival/hazard analysis — Kaplan-Meier, Cox proportional hazards, residence-time estimation |
| `scipy` (`scipy.stats`) | Gamma and exponential distribution fitting via MLE |
| `statsmodels` | AIC/BIC computation, likelihood-ratio tests, Benjamini-Hochberg (FDR) correction |
| `numpy` / `pandas` | General numerical and tabular data handling |
| `networkx` | Building and analyzing the queueing network graph (topology, routing) |
| `matplotlib` / `seaborn` | Static figures for the paper |
| `plotly` | Interactive exploration during development (optional) |
| `pytest` | Unit testing the pipeline before trusting it on real data |
| `jupyter` | Exploratory notebooks |

Install via `pip install scanpy anndata leidenalg python-igraph lifelines scipy statsmodels numpy pandas networkx matplotlib seaborn plotly pytest jupyter` (a pinned `requirements.txt` should replace this once versions are locked).

---

## Scripts Needed (by pipeline stage)

1. **Data acquisition**
   - `download_weinreb.py` — pulls the count matrix, gene names, clone matrix, and metadata from the Klein Lab server
   - `download_nestorowa.py` — pulls the Nestorowa dataset

2. **Preprocessing / state discretization**
   - `state_discretization.py` — marker-gene-based state assignment + Leiden clustering cross-check
   - `schema_mapping.py` — maps each dataset's native states onto the shared coarse meta-hierarchy

3. **Residence-time estimation**
   - `clonal_residence_time.py` — direct estimation from LARRY barcode/clone data (Weinreb subset)
   - `flux_residence_time.py` — population-flux-based fallback estimation for unbarcoded cells

4. **Distribution fitting & model comparison**
   - `distribution_fitting.py` — MLE fitting of gamma and exponential service-time distributions per state
   - `model_comparison.py` — AIC/BIC comparison, Benjamini-Hochberg correction across states

5. **Queueing network**
   - `queueing_network.py` — builds the network object (states, routing, arrival/service rates)
   - `bottleneck_diagnostics.py` — computes traffic intensity per stage, flags bottlenecks

6. **Synthetic validation**
   - `synthetic_generator.py` — generates simulated hierarchies with known, swept bottleneck severities
   - `recovery_validation.py` — measures how accurately the pipeline recovers the true bottleneck location

7. **Real-data validation**
   - `branch_point_validation.py` — tests for elevated congestion at the monocyte branch-point (the corrected validation target)
   - `structural_crosscheck.py` — Nestorowa schema-generalization check

8. **Orchestration**
   - `run_pipeline_weinreb.py` — runs the full quantitative pipeline end-to-end on the primary dataset
   - `run_pipeline_nestorowa.py` — runs the structural-only check
   - `run_synthetic_sweep.py` — runs the full synthetic severity sweep

9. **Output**
   - `generate_figures.py` — produces all paper/poster figures from saved results

---

## Suggested Folder Directory

```
hematopoiesis-queueing/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── weinreb/
│   │   └── nestorowa/
│   ├── processed/
│   └── synthetic/
├── src/
│   └── queuediff/
│       ├── __init__.py
│       ├── data_loading.py
│       ├── state_discretization.py
│       ├── schema_mapping.py
│       ├── clonal_residence_time.py
│       ├── flux_residence_time.py
│       ├── distribution_fitting.py
│       ├── model_comparison.py
│       ├── queueing_network.py
│       ├── bottleneck_diagnostics.py
│       ├── synthetic_generator.py
│       ├── recovery_validation.py
│       ├── branch_point_validation.py
│       └── structural_crosscheck.py
├── scripts/
│   ├── download_weinreb.py
│   ├── download_nestorowa.py
│   ├── run_pipeline_weinreb.py
│   ├── run_pipeline_nestorowa.py
│   ├── run_synthetic_sweep.py
│   └── generate_figures.py
├── notebooks/
│   ├── 01_explore_weinreb.ipynb
│   ├── 02_state_discretization_check.ipynb
│   ├── 03_gamma_fit_diagnostics.ipynb
│   └── 04_bottleneck_results.ipynb
├── tests/
│   ├── test_distribution_fitting.py
│   ├── test_queueing_network.py
│   └── test_synthetic_generator.py
├── results/
│   ├── figures/
│   └── tables/
└── paper/
    ├── manuscript/
    └── isef_poster/
```

### Notes on structure
- `src/queuediff/` is set up as an installable package (`queuediff` — matches the deliverable tool name from the plan), so functions can be imported cleanly in notebooks and scripts alike rather than copy-pasted.
- `notebooks/` is for exploration and diagnostics only — nothing in the final pipeline should depend on notebook-only code; anything that becomes permanent moves into `src/queuediff/`.
- `tests/` should be written alongside `synthetic_generator.py` and `recovery_validation.py` first, since those are what prove the pipeline is trustworthy before it touches real data.
- `results/` and `paper/` stay separate from `src/` so regenerating figures never risks touching pipeline code.
