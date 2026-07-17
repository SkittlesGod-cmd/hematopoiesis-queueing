# Semi-Markovian Queueing Network Inference of Rate-Limiting Bottlenecks in Hematopoietic Stem Cell Differentiation from Single-Cell Transcriptomic Time-Series

Can hematopoietic differentiation be modeled as a semi-Markov queueing network — with empirically-fitted gamma-distributed service times rather than the classical memoryless exponential assumption — to identify which differentiation stage is the rate-limiting bottleneck, directly from single-cell transcriptomic time-series data? This project fits gamma and exponential residence-time distributions per differentiation stage (via MLE), compares them with AIC/BIC, and ranks traffic intensity to flag bottlenecks. Primary dataset: Weinreb et al. 2020 (LARRY barcoded, 130k cells, days 2/4/6). Secondary: Nestorowa et al. 2016 (structural cross-check). Synthetic data with known ground-truth bottlenecks validates the pipeline.

## Project structure

```
├── data/
│   ├── raw/weinreb/          # Weinreb et al. 2020 (GSE140802)
│   ├── raw/nestorowa/        # Nestorowa et al. 2016
│   ├── processed/             # Cleaned AnnData objects
│   └── synthetic/             # Simulated hierarchies with known bottlenecks
├── src/queuediff/             # Installable Python package (queuediff)
│   ├── data_loading.py
│   ├── state_discretization.py
│   ├── schema_mapping.py
│   ├── clonal_residence_time.py
│   ├── flux_residence_time.py
│   ├── distribution_fitting.py
│   ├── model_comparison.py
│   ├── queueing_network.py
│   ├── bottleneck_diagnostics.py
│   ├── synthetic_generator.py
│   ├── recovery_validation.py
│   ├── branch_point_validation.py
│   └── structural_crosscheck.py
├── scripts/                   # Pipeline orchestration
│   ├── download_weinreb.py
│   ├── download_nestorowa.py
│   ├── run_pipeline_weinreb.py
│   ├── run_pipeline_nestorowa.py
│   ├── run_synthetic_sweep.py
│   └── generate_figures.py
├── notebooks/                 # Exploration and diagnostics only
├── tests/                     # Unit tests (written alongside synthetic validation)
├── results/
│   ├── figures/
│   └── tables/
├── paper/
│   ├── manuscript/
│   └── isef_poster/
└── docs/
    ├── research_plan.md
    └── technical_scaffold.md
```

## Setup

```bash
git clone <repo-url>
cd hematopoiesis-queueing
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e src/queuediff
```

## Status

Active development. Target submission: December 1, 2026 (journal + ISEF via NEOSEF).
