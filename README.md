# queuediff — Semi-Markov Queueing Network for Hematopoietic Differentiation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-152_passing-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

**queuediff** models hematopoietic stem cell differentiation as a semi-Markov queueing network with gamma-distributed service times using scRNA-seq time-series data (Weinreb et al. 2020). It identifies rate-limiting bottlenecks in the differentiation process that may trigger clonal expansion and leukemic transformation.

---

## Biological Motivation

Hematopoiesis proceeds through a hierarchy of progenitor states:

```
HSC → MPP → CMP → MEP (erythroid)
           ↘       ↘ GMP (myeloid)
            ↘ LMPP (lymphoid)
```

Each state has a characteristic **residence time** — how long a cell typically stays before differentiating. If a state becomes a **bottleneck** (high traffic intensity + long residence time), cells accumulate there, increasing the risk of oncogenic mutations.

**Confirmed findings**:
| State | Residence Time | Role |
|-------|---------------|------|
| GMP | 19.3h | **Primary bottleneck** |
| MEP | 18.4h | Second longest |
| HSC | 16.6h | Root state |
| MPP | 13.0h | Branch point |
| CMP | 10.5h | Branch point |
| LMPP | 8.4h | Terminal |

All 6 states show gamma-distributed (not exponential) residence times — the key signature of a semi-Markov process.

---

## Pipeline Overview

```
Raw scRNA-seq data (Weinreb 2020)
        ↓
   1. Load & Preprocess (log1p → HVG → PCA)
        ↓
   2. State Discretization (marker gene scoring)
        ↓
   3. Cell Cycle & Apoptosis Scoring
        ↓
   4. Division/Death Rate Calibration
        ↓
   5. Clone Trajectory Extraction (LARRY barcodes)
        ↓
   6. Clonal Residence Time Estimation
        ↓
   7. Gamma vs Exponential Distribution Fitting
        ↓
   8. FDR-Corrected Model Comparison
        ↓
   9. ODE-Based Flux Estimation (matrix exponential)
        ↓
  10. Branch Point Validation (per-cell tracking)
        ↓
  11. Queueing Network Construction
        ↓
  12. Traffic Intensity & Bottleneck Detection
        ↓
  13. Structural Cross-Check (Nestorowa 2016)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/SkittlesGod-cmd/hematopoiesis-queueing.git
cd hematopoiesis-queueing

# Install dependencies
python3 -m pip install -e .
python3 -m pip install -e ".[dev]"
```

---

## Running Tests

```bash
# Full test suite (152 tests)
python3 -m pytest -q --tb=short

# Specific modules
python3 -m pytest tests/test_queueing_network.py -q --tb=short
python3 -m pytest tests/test_bottleneck_diagnostics.py -q --tb=short

# With coverage
python3 -m pytest --cov=queuediff
```

---

## Running the Pipeline

```bash
# Step 1: Download data
python3 scripts/download_weinreb.py    # ~200-300MB
python3 scripts/download_nestorowa.py  # structural cross-check

# Step 2: Run full pipeline
python3 scripts/run_pipeline_weinreb.py

# Step 3: Cross-dataset validation
python3 scripts/run_pipeline_nestorowa.py

# Step 4: Synthetic parameter recovery validation
python3 scripts/run_synthetic_sweep.py

# Step 5: Generate figures
python3 scripts/generate_figures.py
```

---

## Project Structure

```
src/queuediff/              ← Core package (14 modules)
│   ├── data_loading.py           Data ingestion & preprocessing
│   ├── state_discretization.py   Marker scoring & state assignment
│   ├── distribution_fitting.py   MLE gamma/exponential fitting
│   ├── model_comparison.py       AIC/BIC, FDR correction
│   ├── clonal_residence_time.py  LARRY barcode trajectory analysis
│   ├── flux_residence_time.py    ODE-based flux estimation
│   ├── queueing_network.py       NetworkX queueing model
│   ├── bottleneck_diagnostics.py ρ ranking & bottleneck detection
│   ├── branch_point_validation.py Routing probability estimation
│   ├── synthetic_generator.py    Ground-truth data for validation
│   ├── recovery_validation.py    Parameter recovery assessment
│   ├── structural_crosscheck.py  Cross-dataset validation
│   ├── schema_mapping.py         Dataset schema definition
│   └── __init__.py               Package marker (v0.1.0)
│
tests/                      ← 13 test files, 152 tests
└── scripts/                ← Pipeline orchestration
```

---

## Key Technical Improvements (2026-07-22)

| Area | Improvement | Impact |
|------|-------------|--------|
| **ODE solver** | Matrix exponential (`scipy.linalg.expm`) replaces RK45 | Exact solution, no numerical integration error |
| **Branch points** | Per-cell tracking replaces per-clone aggregation | Eliminates routing probability overcounting |
| **Arrival rates** | Groupby dict replaces O(n²) set operations | Linear-time, scales to 130k cells |
| **Gene scoring** | Dict lookup replaces list.index() per gene | O(1) vs O(n) per gene lookup |
| **Data persistence** | Pipeline saves CSVs/JSON for all figures | All 6 figures generate from saved state |
| **Test coverage** | 35 new tests across 3 new files | 152 total, all passing |

---

## Dependencies

- Python ≥ 3.10
- scanpy ≥ 1.9, anndata ≥ 0.10
- numpy ≥ 1.24, scipy ≥ 1.11, pandas ≥ 2.0
- networkx ≥ 3.1, matplotlib ≥ 3.7, seaborn ≥ 0.12
- statsmodels ≥ 0.14, scikit-learn ≥ 1.3
- leidenalg ≥ 0.10, igraph ≥ 0.11

---

## Publication

This work is being prepared for peer-reviewed journal publication and the ISEF 2026 science competition (submission deadline: December 1, 2026).

**Reference**: Weinreb et al. (2020) *Lineage tracing on transcriptional landscapes links state to fate during differentiation*. Science 367(6479):eaaw3381.
