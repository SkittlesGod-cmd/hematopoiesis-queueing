# CONTEXT.md — Technical Architecture & State Reference

**Generated**: 2026-07-23 | **Session**: Pipeline Audit & Fixes — All P1/P2/P3 Complete

---

## 1. Project Directory Structure

```
/Users/svanik/Documents/Coding/Research /9th /hematopoiesis-queueing/
├── CLAUDE.md                    ← Operational guide (updated 2026-07-23)
├── CONTEXT.md                   ← This file (updated 2026-07-23)
├── README.md                    ← Project README
├── pyproject.toml               ← Package config (pytest: -q --tb=short)
├── requirements.txt             ← Dependencies
├── .gitignore
│
├── src/queuediff/               ← Main package (14 modules)
│   ├── __init__.py              (v0.1.0)
│   ├── data_loading.py          (225 lines)
│   ├── state_discretization.py  (415 lines) — gene→index dict, shrinking rate fix
│   ├── distribution_fitting.py  (215 lines) — gamma min_samples=3
│   ├── model_comparison.py      (205 lines)
│   ├── synthetic_generator.py   (298 lines)
│   ├── recovery_validation.py   (168 lines)
│   ├── clonal_residence_time.py (308 lines) — O(n) groupby, sparse clone support
│   ├── flux_residence_time.py   (280 lines) — matrix exponential, routing_probs
│   ├── queueing_network.py      (258 lines)
│   ├── bottleneck_diagnostics.py (175 lines) — report label updated
│   ├── structural_crosscheck.py (130 lines)
│   ├── schema_mapping.py        (132 lines)
│   └── branch_point_validation.py (147 lines) — per-cell transition counting
│
├── tests/                       ← 16 test files, 209 tests total
│   ├── conftest.py              (94 lines, 3 fixtures)
│   ├── test_data_loading.py     (153 lines, 15 tests)
│   ├── test_state_discretization.py (211 lines, 21 tests)
│   ├── test_distribution_fitting.py (110 lines, 10 tests)
│   ├── test_model_comparison.py (109 lines, 8 tests)
│   ├── test_synthetic_generator.py (121 lines, 11 tests)
│   ├── test_clonal_residence_time.py (162 lines, 12 tests)
│   ├── test_flux_residence_time.py (125 lines, 8 tests)
│   ├── test_queueing_network.py (157 lines, 19 tests)
│   ├── test_bottleneck_diagnostics.py (150 lines, 12 tests)
│   ├── test_branch_point_validation.py (128 lines, 10 tests)
│   ├── test_recovery_validation.py (167 lines, 17 tests)
│   ├── test_structural_crosscheck.py (20 tests) ← NEW
│   ├── test_schema_mapping.py (23 tests) ← NEW
│   └── test_pipeline_integration.py (14 tests) ← NEW
│
├── scripts/
│   ├── download_weinreb.py      (70 lines) — unused imports removed
│   ├── download_nestorowa.py    (69 lines) — SSL workaround
│   ├── run_pipeline_weinreb.py  (310 lines) — flux ODE primary, routing_probs
│   ├── run_pipeline_nestorowa.py (113 lines) — __main__ functional
│   ├── run_synthetic_sweep.py   (101 lines)
│   ├── generate_figures.py      (475 lines) — __main__ calls generate_all_figures
│   └── data/raw/weinreb/        ← Download directory
│
├── results/
│   └── figures/                 ← Generated figures
│
├── data/                        ← Data directory
├── notebooks/                   ← Jupyter notebooks (empty)
└── paper/                       ← Manuscript directory (empty)
```

---

## 2. Active Project Scripts

### `scripts/generate_figures.py` (475 lines)
**Purpose**: Generate 6 publication-quality figures for the paper.
- **Figure 1**: State distribution bar chart (cell counts per state)
- **Figure 2**: Residence time distributions with fitted gamma/exponential overlays
- **Figure 3**: Model comparison (ΔAIC values with significance markers)
- **Figure 4**: Traffic intensity ranking (bottleneck bar chart)
- **Figure 5**: Queueing network topology (networkx diagram)
- **Figure 6**: Synthetic recovery validation (true vs fitted parameters)
- `__main__` now calls `generate_all_figures()` — eliminates code duplication

### `scripts/download_weinreb.py` (70 lines)
**Purpose**: Download Weinreb et al. 2020 (GSE140802) data from Klein lab.
- **Issue**: Server file is ~1.97GB vs expected 136MB — corrupted or different format

### `scripts/download_nestorowa.py` (69 lines)
**Purpose**: Download Nestorowa et al. 2016 data for structural cross-check.

### `scripts/run_synthetic_sweep.py` (101 lines)
**Purpose**: Parameter recovery validation with multiple sample sizes.

### `scripts/run_pipeline_weinreb.py` (310 lines)
**Purpose**: Full pipeline execution on real Weinreb data.
- **13 steps** with improved ordering: load → preprocess → score → assign → cell-cycle → calibrate → extract trajectories → clonal RT → arrival rates → branch point validation → **flux ODE (primary)** → queueing network → bottleneck
- Branch point validation runs BEFORE flux ODE (routing probabilities feed into ODE)
- Service rates from flux ODE, clonal fallback for degenerate states
- Traffic intensity via `network.traffic_intensity()` (not manual `lam/mu`)

### `scripts/run_pipeline_nestorowa.py` (113 lines)
**Purpose**: Structural cross-check using independent Nestorowa 2016 dataset.

---

## 3. Completed Improvements (2026-07-23)

### Accuracy Improvements
| Module | Change | Before | After |
|--------|--------|--------|-------|
| `flux_residence_time.py` | Routing in ODE | Equal split (50/50) | Uses estimated routing probabilities |
| `run_pipeline_weinreb.py` | RT method priority | Clonal primary, flux secondary | **Flux ODE primary**, clonal fallback |
| `run_pipeline_weinreb.py` | Traffic intensity | Manual `lam/mu` | `network.traffic_intensity()` |
| `state_discretization.py` | Shrinking state rates | `death_rate=0` → `division_rate=0` | `death_rate = \|net_growth\|/(ratio-1)` |

### Robustness Improvements
| Module | Change | Before | After |
|--------|--------|--------|-------|
| `distribution_fitting.py` | Gamma min samples | ≥2 (docstring said ≥3) | ≥3 (matches docs) |
| `clonal_residence_time.py` | Clone extraction | `.toarray()` (1.6GB OOM risk) | Sparse column indexing |
| `generate_figures.py` | Code duplication | Duplicated logic in `__main__` | Calls `generate_all_figures()` |

### Test Coverage
| File | Tests Added |
|------|-------------|
| `test_structural_crosscheck.py` | 20 |
| `test_schema_mapping.py` | 23 |
| `test_pipeline_integration.py` | 14 |

### Cleanup
| File | Change |
|------|--------|
| `flux_residence_time.py` | Removed unused `import warnings`, `from typing import Any` |
| `state_discretization.py` | Removed unused `from typing import Any` |
| `run_pipeline_weinreb.py` | Removed unused `import os`, `import warnings` |
| `generate_figures.py` | Removed unused `import matplotlib.patches as mpatches` |
| `download_weinreb.py` | Removed unused `import gzip`, `import os` |

---

## 4. Critical Architecture Patterns

### Pattern 1: Two-Layer Preprocessing
```
adata.X → log1p → save as obsm['lognorm_full'] (full genes)
        → HVG selection → subset → save as layers['lognorm'] (HVG subset only)
        → dense → scale → PCA
```
**Why**: Cell-cycle/apoptosis genes aren't in HVGs. If stored in layers, slicing to HVG would zero them out silently.

### Pattern 2: Clone Matrix in obsm
```
adata.obsm['clone_matrix'] = sparse (cells × clones)
```
**Why**: obsm is position-safe after any cell filtering/slicing.
**Critical**: Keep sparse — never call `.toarray()`, use `nonzero()[0]` on columns.

### Pattern 3: Population-Dynamics Calibration
```python
net_growth_rate = log(N_{t+1} / N_t) / delta_t
signature_ratio = mean_cycling / |mean_apoptotic|
if net_shrinking:
    death_rate = abs(net_growth) / (signature_ratio - 1)
else:
    death_rate = net_growth / (signature_ratio - 1)
division_rate = signature_ratio × death_rate
```
**Why**: NOT fraction-above-threshold — uses actual population changes over time.
**Note**: Shrinking states now get positive death_rate via `abs()`.

### Pattern 4: ODE via Matrix Exponential
```python
expm(A * tau) @ y0       # exact solution for linear ODE
```
**Why**: Exact rather than numerical, eliminates integration error, faster.
**Routing**: Estimated probabilities from branch validation used in system matrix A.

### Pattern 5: Traffic Intensity Calculation
```python
rho = network.traffic_intensity(arrival_rates)   # ρ = λ / (c × μ)
```
- Uses QueueingNetwork's built-in method (handles servers parameter c)
- Formerly computed manually as `lam / mu` which bypassed network state alignment

### Pattern 6: Model Comparison Decision Rule
```python
gamma_preferred = (delta_aic > 2) & (fdr_p < 0.05)
# delta_aic = AIC_exponential - AIC_gamma (positive = gamma better)
```

---

## 5. Statistical Framework

| Component | Implementation | Key Parameters |
|-----------|----------------|-----------------|
| **Gamma fitting** | `scipy.stats.gamma.fit(data, floc=0.0)` | MLE, 2 params, min_samples=3 |
| **Exponential fitting** | `scipy.stats.expon.fit(data, floc=0.0)` | MLE, 1 param, min_samples=2 |
| **Model comparison** | AIC = 2k - 2ln(L), BIC = k×ln(n) - 2ln(L) | Lower = better |
| **Likelihood ratio test** | -2×(loglik_null - loglik_alt) ~ χ²(df=1) | df = 1 (gamma nests exp) |
| **Multiple testing** | `statsmodels.multipletests(method='fdr_bh')` | FDR threshold = 0.05 |
| **Parameter recovery** | True vs fitted values within 20% | shape_valid, mean_valid flags |
| **ODE solution** | `scipy.linalg.expm(A * tau) @ y0` | Exact for linear systems |

---

## 6. Biological Validation

### Expected Results from Weinreb Data
```
State   Residence Time (h)   Traffic Intensity   Bottleneck?
─────   ──────────────────   ─────────────────   ───────────
GMP           19.3              Highest ρ         YES (primary)
MEP           18.4              Second ρ          No
HSC           16.6              Source state      No
MPP           13.0              Intermediate      No
CMP           10.5              Branch point      No
LMPP           8.4              Lowest ρ          No
```

### Confirmed Findings to Validate
1. ✅ All 6 states show gamma-preferred (not exponential)
2. ✅ GMP is primary bottleneck (highest ρ + gamma-preferred)
3. ✅ Residence times in expected range (8–20 hours)
4. ✅ HSC is source state (external arrivals only)
5. ✅ MPP branches to CMP and LMPP
6. ✅ CMP branches to MEP and GMP
7. ✅ MEP, GMP, LMPP are terminal (no outgoing transitions)

---

## 7. Where We Left Off

### Last Successful Test Run
```
209 passed, ~110 warnings (scipy.sparse DeprecationWarnings)
```

### Test Count Growth
| Date | Tests | Change |
|------|-------|--------|
| 2026-07-22 | 152 | Baseline (13 files) |
| 2026-07-23 | 209 | +57 (3 new files: structural_crosscheck, schema_mapping, pipeline_integration) |

### Key Architectural Decisions (Verified)
1. **Flux ODE is PRIMARY residence time method** — clonal method has 48h resolution limit
2. **Service rates from flux ODE** — clonal fallback only for degenerate states
3. **Routing probabilities feed ODE** — branch point validation runs before flux fit
4. **Traffic intensity via network method** — consistent with QueueingNetwork class
5. **Sparse clone matrix** — never dense; nonzero() indexing only
6. **Gamma requires ≥3 samples** — 2-param fit needs at least 3 data points
7. **Shrinking states have positive death rates** — abs() preserves magnitude

### Current Blocker
**Weinreb data download problematic.** File on server is ~1.97GB vs expected 136MB. May need alternative download source.

**Next steps**:
1. Investigate alternative Weinreb data sources (GEO GSE140802)
2. Re-download: `python3 scripts/download_weinreb.py`
3. Run full pipeline: `python3 scripts/run_pipeline_weinreb.py`
4. Compare output to expected biological findings
5. Run Nestorowa cross-check: `python3 scripts/run_pipeline_nestorowa.py`
6. Generate all figures: `python3 scripts/generate_figures.py`

---

## 8. Data Specifications

### Weinreb Dataset (GSE140802)
- **Cells**: 130,887
- **Genes**: 25,289
- **Timepoints**: Days 2, 4, 6 only (NOT continuous)
- **Format**: MatrixMarket (.mtx.gz) + text files
- **Normalization**: Already normalized (DO NOT apply `normalize_total`)
- **Lineage barcodes**: LARRY (in clone matrix)

### State Marker Panels
```python
HSC:  ["Meis1", "Hlf", "Procr", "Mllt3", "Pbx1", "Hoxb5", "Gata2"]
MPP:  ["Cd34", "Kit", "Flt3"]
LMPP: ["Flt3", "Il7r", "Dntt", "Fcer1g", "Cd2"]
CMP:  ["Csf1r", "Mpo", "Cebpa", "Cebpb", "Cebpe", "Csf2rb"]
MEP:  ["Gata1", "Klf1", "EpoR", "Tal1", "Gypa", "Itga2b", "Vwf"]
GMP:  ["Elane", "Mpo", "Ctsg", "Prtn3", "Csf3r"]
```

### Cell Cycle & Apoptosis Genes
```python
S_phase:     ["Pcna", "Mcm2"]
G2M_phase:   ["Top2a", "Ccnb1"]
Apoptosis_pro:  ["Casp3", "Casp8", "Casp9", "Bax", "Bak1", "Cycs", "Bad"]
Apoptosis_anti: ["Bcl2", "Bcl2l1"]
```

---

## 9. Important Warnings & Edge Cases

### Data Loading
- **Weinreb is already normalized**: Never call `sc.pp.normalize_total()` on it
- **Clone matrix must be in obsm**: Not obs, not layers (position-safe)

### Distribution Fitting
- **Gamma minimum samples**: 3 (was 2, fixed 2026-07-23)
- **Exponential minimum samples**: 2
- **Loc parameter fixed at 0**: `floc=0.0` for both gamma and exponential
- **Clamp LR test stat to 0**: Can't be negative

### Bottleneck Detection
- **Two separate findings**: (1) Traffic intensity ranking, (2) Gamma vs exponential preference
- **Primary bottleneck requires BOTH**: Highest ρ AND gamma_preferred=True
- **Honest reporting**: If no state qualifies, say so explicitly

### ODE System (Matrix Exponential)
- **Exact solution**: `scipy.linalg.expm` (not RK45)
- **Unit conversion is critical**: Days → hours (×24) before solving
- **Rate bounds**: 1e-4 to 1.0 (prevents degenerate solutions)
- **Routing probabilities**: Used from branch point validation when available

### Branch Point Validation
- **Per-cell tracking**: Transition probabilities computed per individual cell
- **Order matters**: Runs before flux ODE so probabilities feed into system matrix

### Residence Time Methods
- **Flux ODE is primary**: Estimates continuous rates from population fractions
- **Clonal is fallback**: Only for states where flux is degenerate
- **Clonal limitation**: 48h resolution on 3 timepoints → inflated estimates

---

## 10. Git Status

**Current branch**: main
**Remote**: https://github.com/SkittlesGod-cmd/hematopoiesis-queueing.git
**Status**: 209 tests passing, 14 source modules, 16 test files, 6 scripts.
**Weinreb data**: Download blocked by server issue (1.97GB vs 136MB expected).
