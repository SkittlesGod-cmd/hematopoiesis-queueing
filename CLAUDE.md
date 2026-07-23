# CLAUDE.md — Operational Guide for queuediff Project

## Project Objective
We are building **queuediff** — a production-quality Python package for modeling hematopoietic differentiation as a semi-Markov queueing network with gamma-distributed service times using scRNA-seq time-series data (Weinreb et al. 2020). This is for a peer-reviewed journal paper and ISEF science competition (deadline December 1, 2026).

**Goal**: Reproduce confirmed biological findings:
- 6 hematopoietic states (HSC, MPP, LMPP, CMP, MEP, GMP)
- All 6 states gamma-preferred (not exponential)
- GMP is primary bottleneck (highest traffic intensity AND gamma-preferred)
- Specific residence times: GMP 19.3h, MEP 18.4h, HSC 16.6h, MPP 13.0h, CMP 10.5h, LMPP 8.4h

---

## MANDATORY Runtime Command Rules

**CRITICAL: ALWAYS use `python3`, NEVER use `python` or `pytest` directly.**

The system Python is Python 3.14 on this machine, but the pip-installed packages are under a different Python (3.13) via Homebrew. Direct `pytest` or `python` invocations fail with "command not found" or import errors.

### Test Commands (use these):
```bash
# Full test suite (152 tests)
python3 -m pytest -q --tb=short

# Run specific test file
python3 -m pytest tests/test_distribution_fitting.py -q --tb=short

# Run specific test class or method
python3 -m pytest tests/test_distribution_fitting.py::TestFitGamma -q --tb=short
```

### Script Execution:
```bash
# Always use python3 to run scripts
python3 scripts/download_weinreb.py
python3 scripts/run_pipeline_weinreb.py
python3 scripts/run_synthetic_sweep.py
python3 scripts/generate_figures.py
```

---

## Current Status

**All 8 phases of build order completed. 209 tests passing, 0 failures.**

### Completed Improvements (2026-07-23)

| Module | Improvement | Impact |
|--------|-------------|--------|
| `flux_residence_time.py` | Accepts `routing_probs` from branch validation | ODE system matrix uses estimated probabilities instead of equal-split |
| `run_pipeline_weinreb.py` | Flux ODE is PRIMARY residence time method | Clonal method now fallback only (48h resolution limitation) |
| `run_pipeline_weinreb.py` | Branch point validation → flux ODE order | Estimated routing probabilities feed into ODE solver |
| `run_pipeline_weinreb.py` | Uses `network.traffic_intensity()` | Consistent ρ = λ/(c·μ) including servers parameter |
| `state_discretization.py` | Shrinking state death rate fix | `|net_growth|/(ratio-1)` instead of clamping to 0 |
| `clonal_residence_time.py` | Sparse clone extraction | No dense `.toarray()` — avoids 1.6GB OOM on full data |
| `distribution_fitting.py` | Gamma min samples ≥3 | Matches docstring (2-param fit needs ≥3 points) |
| `generate_figures.py` | `__main__` calls `generate_all_figures()` | Eliminates ~100 lines of code duplication |
| `flux_residence_time.py` | Removed unused imports | Cleaner import section |
| `run_pipeline_weinreb.py` | Removed unused imports | Cleaner import section |
| `download_weinreb.py` | Removed unused imports | Cleaner import section |
| `bottleneck_diagnostics.py` | Report label updated | "flux ODE primary" vs "clonal method" |

### Fixed Bugs (2026-07-23)
1. Flux ODE used equal-split routing (50/50) instead of estimated probabilities
2. Pipeline computed `lam/mu` manually, bypassing QueueingNetwork and servers param
3. Division/death rate calibration set `division_rate=0` for shrinking states
4. Clone matrix dense conversion (`toarray()`) caused ~1.6GB OOM risk
5. `fit_gamma` validated ≥2 but docstring said ≥3 — now enforces ≥3
6. `generate_figures.py` duplicated all figure logic between `__main__` and `generate_all_figures()`
7. Bottleneck report labeled "clonal method" but uses flux ODE as primary
8. 5 unused imports across 4 files

### Test Coverage Added (2026-07-23)
- `test_structural_crosscheck.py` (20 tests)
- `test_schema_mapping.py` (23 tests)
- `test_pipeline_integration.py` (14 tests)

### Total Test Count
**209 tests** across 16 test files:
- 152 original tests + 57 new (20 + 23 + 14)

### Current Blocker
**Weinreb data download problematic.** File on server is ~1.97GB vs expected 136MB. May need alternative source (GEO accession GSE140802).

### What to do next
1. Investigate alternative sources for Weinreb data (GEO/ArrayExpress)
2. Once data is available, run: `python3 scripts/download_weinreb.py`
3. Run full pipeline: `python3 scripts/run_pipeline_weinreb.py`
4. Check output against expected biological findings
5. Run Nestorowa cross-check: `python3 scripts/run_pipeline_nestorowa.py`
6. Generate figures: `python3 scripts/generate_figures.py`

---

## Build Order Phases (COMPLETED)

### Phase 1: Data Loading & Preprocessing ✅
- `src/queuediff/data_loading.py` (225 lines)
- Key decisions:
  - No `normalize_total` on Weinreb data (already normalized)
  - log1p → HVG selection (seurat flavor) → subset → scale → PCA
  - Two-layer strategy: `lognorm_full` (obsm, full gene set) for cell-cycle/apoptosis scoring, `lognorm` (layer, HVG subset) for marker scoring
  - Clone matrix in obsm for position-safe alignment

### Phase 2: State Discretization ✅
- `src/queuediff/state_discretization.py` (407 lines)
- 6 validated marker panels, population-dynamics calibration for division/death rates
- Cell cycle and apoptosis scoring uses `obsm['lognorm_full']` (full gene set)
- **Optimization**: O(1) gene→index dict for marker scoring

### Phase 3: Distribution Fitting & Model Comparison ✅
- `src/queuediff/distribution_fitting.py` (213 lines)
- `src/queuediff/model_comparison.py` (205 lines)
- Gamma vs exponential MLE, AIC/BIC, LR test, Benjamini-Hochberg FDR

### Phase 4: Synthetic Data Generator & Recovery Validation ✅
- `src/queuediff/synthetic_generator.py` (298 lines)
- `src/queuediff/recovery_validation.py` (168 lines)
- Synthetic sweep: 96.7–100% recovery at n=100 to n=5000

### Phase 5: Clone Matrix & Clonal Residence Time ✅
- `src/queuediff/clonal_residence_time.py` (310 lines)
- Uses `obsm['clone_matrix']` (position-safe by obsm)
- **Optimization**: O(n) groupby-based arrival rates

### Phase 6: Flux ODE & Bottleneck Diagnostics ✅
- `src/queuediff/flux_residence_time.py` (273 lines)
- `src/queuediff/bottleneck_diagnostics.py` (175 lines)
- **CRITICAL**: ODE solved via matrix exponential (exact, not numerical RK45)
- Timepoints (days) × 24 → hours before solving

### Phase 7: Queueing Network & Structural Cross-check ✅
- `src/queuediff/queueing_network.py` (258 lines) — backed by `nx.DiGraph`
- `src/queuediff/structural_crosscheck.py` (130 lines)
- `src/queuediff/schema_mapping.py` (132 lines) — integrated into nestorowa pipeline
- `src/queuediff/branch_point_validation.py` (147 lines) — per-cell transition counting
- Traffic intensity: ρ = λ / (c × μ)

### Phase 8: Pipeline Scripts & Figures ✅
- `scripts/download_weinreb.py` (73 lines)
- `scripts/run_pipeline_weinreb.py` (13-step orchestration + persistence)
- `scripts/run_pipeline_nestorowa.py` (93 lines) — fixed __main__
- `scripts/run_synthetic_sweep.py` (102 lines)
- `scripts/generate_figures.py` (496 lines) — generates all 6 figures from persistence

---

## Statistical Framework

| Component | Method |
|-----------|--------|
| Distribution fitting | MLE via `scipy.stats.gamma.fit`, `scipy.stats.expon.fit` |
| Model comparison | AIC/BIC (lower = better) |
| Significance test | Likelihood ratio test (chi-squared) |
| Multiple testing correction | Benjamini-Hochberg FDR via `statsmodels.multipletests` |
| Decision rule | `gamma_preferred = (delta_aic > 2) & (fdr_p < 0.05)` |
| Bottleneck ranking | Traffic intensity ρ = λ/(c·μ), highest = bottleneck |
| ODE solution | Matrix exponential `scipy.linalg.expm` (exact, not RK45) |

---

## Biological Findings to Reproduce

| State | Expected Residence Time (h) | Notes |
|-------|---------------------------|-------|
| GMP | 19.3 | Primary bottleneck (highest ρ + gamma-preferred) |
| MEP | 18.4 | Second longest |
| HSC | 16.6 | Root of hierarchy |
| MPP | 13.0 | Intermediate |
| CMP | 10.5 | Branching point (→ MEP, GMP) |
| LMPP | 8.4 | Shortest (terminal) |

All 6 states must show gamma-preferred (not exponential).

---

## Key Architecture Decisions

1. **Two-layer preprocessing**: `lognorm_full` in obsm (not layers) because layers get gene-subset when slicing to HVGs → would zero out cell-cycle/apoptosis scores
2. **Population-dynamics calibration**: NOT fraction-above-threshold for division/death rates
3. **Clone matrix alignment**: obsm-based (position-safe after any filtering)
4. **Arrival rate normalization**: λ_normalized = inflow_cells_per_hour / total_cells_in_state
5. **ODE unit conversion**: timepoints (days) → hours (×24) before solving; matrix exponential for exact solution
6. **Two separate bottleneck findings**: Traffic intensity ranking + gamma vs exponential preference

---

## Source File Reference

### Core Package (`src/queuediff/`)
- `__init__.py` — Package marker, `__version__ = "0.1.0"`
- `data_loading.py` — Load Weinreb/Nestorowa data, preprocessing
- `state_discretization.py` — Marker scoring, state assignment, rate calibration
- `distribution_fitting.py` — MLE fitting (gamma, exponential), LR test
- `model_comparison.py` — Per-state AIC/BIC comparison, FDR correction
- `synthetic_generator.py` — Ground-truth synthetic data
- `recovery_validation.py` — Parameter recovery validation
- `clonal_residence_time.py` — LARRY barcode trajectory extraction
- `flux_residence_time.py` — ODE-based flux estimation (matrix exponential)
- `queueing_network.py` — NetworkX-backed queueing network
- `bottleneck_diagnostics.py` — Bottleneck ranking and reporting
- `structural_crosscheck.py` — Cross-dataset structural validation
- `schema_mapping.py` — Cross-dataset state mapping
- `branch_point_validation.py` — Branch point identification

### Test Files (`tests/`)
| File | Tests |
|------|-------|
| `conftest.py` | 3 fixtures |
| `test_data_loading.py` | 15 tests |
| `test_state_discretization.py` | 21 tests |
| `test_distribution_fitting.py` | 10 tests |
| `test_model_comparison.py` | 8 tests |
| `test_synthetic_generator.py` | 11 tests |
| `test_clonal_residence_time.py` | 12 tests |
| `test_flux_residence_time.py` | 8 tests |
| `test_queueing_network.py` | 18 tests |
| `test_bottleneck_diagnostics.py` | 12 tests |
| `test_branch_point_validation.py` | 10 tests |
| `test_recovery_validation.py` | 17 tests |
| `test_structural_crosscheck.py` | 20 tests |
| `test_schema_mapping.py` | 23 tests |
| `test_pipeline_integration.py` | 14 tests |

### Scripts (`scripts/`)
- `download_weinreb.py` — Download from kleintools.hms.harvard.edu
- `download_nestorowa.py` — Download Nestorowa 2016 data (SSL workaround added)
- `run_pipeline_weinreb.py` — 13-step Weinreb pipeline with data persistence
- `run_pipeline_nestorowa.py` — Structural cross-check (__main__ fixed)
- `run_synthetic_sweep.py` — Parameter recovery sweep
- `generate_figures.py` — 6 publication figures (all generated from __main__)

---

## Known Issues & Gotchas

1. **Weinreb download**: Server file is ~1.97GB vs expected 136MB — may need GEO source
2. **SSL Certificates**: Download scripts use `ssl._create_unverified_context()` due to Python 3.14 SSL cert issues
3. **scipy DeprecationWarning**: ~85 warnings about sparse matrix operations — expected and harmless
4. **Weinreb data is already normalized**: Do NOT apply `normalize_total` — would corrupt data
5. **Directory paths have spaces**: Working directory is `/Users/svanik/Documents/Coding/Research /9th /hematopoiesis-queueing`
6. **branch_point_validation**: Uses per-cell tracking (not per-clone) for accurate routing probabilities
7. **flux_residence_time**: Uses matrix exponential not RK45 — exact solution for linear ODE

---

## Test Execution Quick Reference

```bash
# Run all 209 tests
python3 -m pytest -q --tb=short

# Run specific module tests
python3 -m pytest tests/test_bottleneck_diagnostics.py -q --tb=short
python3 -m pytest tests/test_branch_point_validation.py -q --tb=short
python3 -m pytest tests/test_structural_crosscheck.py -q --tb=short
python3 -m pytest tests/test_schema_mapping.py -q --tb=short
python3 -m pytest tests/test_pipeline_integration.py -q --tb=short
python3 -m pytest tests/test_flux_residence_time.py -q --tb=short
python3 -m pytest tests/test_queueing_network.py -q --tb=short

# Generate synthetic sweep results
python3 scripts/run_synthetic_sweep.py
```
