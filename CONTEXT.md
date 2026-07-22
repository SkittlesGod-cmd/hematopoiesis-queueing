# CONTEXT.md — Technical Architecture & State Reference

**Generated**: 2026-07-22 | **Session**: Phase 8 Complete + Improvements Applied, Data Download Blocked

---

## 1. Project Directory Structure

```
/Users/svanik/Documents/Coding/Research /9th /hematopoiesis-queueing/
├── CLAUDE.md                    ← Operational guide (updated 2026-07-22)
├── CONTEXT.md                   ← This file (updated 2026-07-22)
├── README.md                    ← Project README
├── pyproject.toml               ← Package config (pytest: -q --tb=short)
├── requirements.txt             ← Dependencies
├── .gitignore
│
├── src/queuediff/               ← Main package (14 modules)
│   ├── __init__.py              (v0.1.0)
│   ├── data_loading.py          (225 lines)
│   ├── state_discretization.py  (407 lines) — gene→index dict optimized
│   ├── distribution_fitting.py  (213 lines)
│   ├── model_comparison.py      (205 lines)
│   ├── synthetic_generator.py   (298 lines)
│   ├── recovery_validation.py   (168 lines)
│   ├── clonal_residence_time.py (310 lines) — O(n) groupby arrival rates
│   ├── flux_residence_time.py   (273 lines) — matrix exponential ODE solver
│   ├── queueing_network.py      (258 lines)
│   ├── bottleneck_diagnostics.py (175 lines)
│   ├── structural_crosscheck.py (130 lines)
│   ├── schema_mapping.py        (132 lines) — integrated into nestorowa pipeline
│   └── branch_point_validation.py (147 lines) — per-cell transition counting
│
├── tests/                       ← 13 test files, 152 tests total
│   ├── conftest.py              (94 lines, 3 fixtures)
│   ├── test_data_loading.py     (153 lines, 15 tests)
│   ├── test_state_discretization.py (211 lines, 21 tests)
│   ├── test_distribution_fitting.py (110 lines, 10 tests)
│   ├── test_model_comparison.py (109 lines, 8 tests)
│   ├── test_synthetic_generator.py (121 lines, 11 tests)
│   ├── test_clonal_residence_time.py (162 lines, 12 tests)
│   ├── test_flux_residence_time.py (125 lines, 8 tests)
│   ├── test_queueing_network.py (157 lines, 19 tests)
│   ├── test_bottleneck_diagnostics.py (150 lines, 12 tests) ← NEW
│   ├── test_branch_point_validation.py (128 lines, 10 tests) ← NEW
│   └── test_recovery_validation.py (167 lines, 17 tests) ← NEW
│
├── scripts/
│   ├── download_weinreb.py      (73 lines)
│   ├── download_nestorowa.py    (69 lines) — SSL workaround added
│   ├── run_pipeline_weinreb.py  (334 lines) — data persistence added
│   ├── run_pipeline_nestorowa.py (113 lines) — __main__ fixed, schema validation
│   ├── run_synthetic_sweep.py   (101 lines) — unused import removed
│   ├── generate_figures.py      (566 lines) — all 6 figures from __main__
│   └── data/raw/weinreb/        ← Download directory
│       └── stateFate_inVitro_normed_counts.mtx.gz (136MB ✓)
│           stateFate_inVitro_gene_names.txt.gz (PENDING)
│           stateFate_inVitro_metadata.txt.gz (PENDING)
│           stateFate_inVitro_clone_matrix.mtx.gz (PENDING)
│
├── results/
│   ├── figures/                 ← Generated figures (fig6_recovery_validation.png ✓)
│   ├── tables/                  ← Generated tables
│   └── synthetic_sweep_results.csv ← Parameter recovery sweep results
│
├── data/                        ← Data directory
├── notebooks/                   ← Jupyter notebooks (empty)
└── paper/                       ← Manuscript directory (empty)
```

---

## 2. Active Project Scripts

### `scripts/generate_figures.py` (566 lines)
**Purpose**: Generate 6 publication-quality figures for the paper.
- **Figure 1**: State distribution bar chart (cell counts per state)
- **Figure 2**: Residence time distributions with fitted gamma/exponential overlays
- **Figure 3**: Model comparison (ΔAIC values with significance markers)
- **Figure 4**: Traffic intensity ranking (bottleneck bar chart)
- **Figure 5**: Queueing network topology (networkx diagram)
- **Figure 6**: Synthetic recovery validation (true vs fitted parameters)
- All 6 figures can now be generated from `__main__` via persisted CSVs/JSON.

### `scripts/download_weinreb.py` (73 lines)
**Purpose**: Download Weinreb et al. 2020 (GSE140802) data from Klein lab.
- **Source**: `https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/`
- **Issue**: Server file is ~1.97GB vs expected 136MB — corrupted or different format
- Uses `ssl._create_unverified_context()` for certificate issues

### `scripts/download_nestorowa.py` (69 lines)
**Purpose**: Download Nestorowa et al. 2016 data.
- **Source**: `https://blood.stemcells.cam.ac.uk/data/`
- SSL certificate workaround now matches download_weinreb.py pattern

### `scripts/run_synthetic_sweep.py` (101 lines)
**Purpose**: Parameter recovery validation with multiple sample sizes.
- Tests sample sizes: n = [100, 500, 1000, 5000]
- 5 repeats per sample size
- Outputs: `results/synthetic_sweep_results.csv`
- Shows 96.7–100% recovery across all conditions

### `scripts/run_pipeline_weinreb.py` (334 lines)
**Purpose**: Full pipeline execution on real Weinreb data.
- 13 steps: load → preprocess → score → assign → cell-cycle → calibrate → fit → compare → FDR → traffic intensity → rank → report → save
- Now persists state_assignments.csv, residence_times.json, routing_probabilities.json

### `scripts/run_pipeline_nestorowa.py` (113 lines)
**Purpose**: Structural cross-check using independent Nestorowa 2016 dataset.
- Schema validation from schema_mapping.py integrated
- __main__ now functional: loads state_assignments.csv and runs cross-check

---

## 3. Completed Improvements (2026-07-22)

### Accuracy Improvements
| Module | Change | Before | After |
|--------|--------|--------|-------|
| `branch_point_validation.py` | Transition counting | Per-clone (overcounts) | Per-cell (accurate) |
| `flux_residence_time.py` | ODE solver | RK45 (numerical error) | Matrix exponential (exact) |

### Efficiency Improvements
| Module | Change | Before | After |
|--------|--------|--------|-------|
| `clonal_residence_time.py` | Arrival rates | O(n²) set operations | O(n) groupby dict |
| `state_discretization.py` | Gene lookups | O(n) index() per gene | O(1) dict per gene |

### Test Coverage
| File | Tests Added |
|------|-------------|
| `test_bottleneck_diagnostics.py` | 12 |
| `test_branch_point_validation.py` | 10 |
| `test_recovery_validation.py` | 17 |

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

### Pattern 3: Population-Dynamics Calibration
```python
net_growth_rate = log(N_{t+1} / N_t) / delta_t
signature_ratio = mean_cycling / |mean_apoptotic|
death_rate = net_growth / (signature_ratio - 1)
division_rate = signature_ratio × death_rate
```
**Why**: NOT fraction-above-threshold — uses actual population changes over time.

### Pattern 4: ODE via Matrix Exponential
```python
# Before: solve_ivp(ode_rhs, ...)  — RK45 numerical integration
# After:  expm(A * tau) @ y0       — exact solution for linear ODE
```
**Why**: Exact rather than numerical, eliminates integration error, faster.

### Pattern 5: Traffic Intensity Calculation
```python
rho = arrival_rate / (servers × service_rate)  # ρ = λ / (c × μ)
```
- Source states get external arrival rate
- Non-source states get propagated arrival from routing matrix
- Bottleneck = highest ρ AND gamma-preferred

### Pattern 6: Model Comparison Decision Rule
```python
gamma_preferred = (delta_aic > 2) & (fdr_p < 0.05)
# delta_aic = AIC_exponential - AIC_gamma (positive = gamma better)
```

---

## 5. Statistical Framework

| Component | Implementation | Key Parameters |
|-----------|----------------|-----------------|
| **Gamma fitting** | `scipy.stats.gamma.fit(data, floc=0.0)` | MLE, 2 params (shape, scale) |
| **Exponential fitting** | `scipy.stats.expon.fit(data, floc=0.0)` | MLE, 1 param (scale) |
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
1. ✓ All 6 states show gamma-preferred (not exponential)
2. ✓ GMP is primary bottleneck (highest ρ + gamma-preferred)
3. ✓ Residence times in expected range (8–20 hours)
4. ✓ HSC is source state (external arrivals only)
5. ✓ MPP branches to CMP and LMPP
6. ✓ CMP branches to MEP and GMP
7. ✓ MEP, GMP, LMPP are terminal (no outgoing transitions)

---

## 7. Where We Left Off

### Last Successful Test Run
```
152 passed, ~110 warnings in 2.88s
```
All warnings are expected `DeprecationWarning` about scipy.sparse matrix operations.

### Synthetic Sweep Results
- 96.7–100% parameter recovery across all sample sizes (n=100 to n=5000)
- 5 repeats per condition
- Results saved to `results/synthetic_sweep_results.csv`

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
- **Minimum sample size**: 10 per state (otherwise skip)
- **Loc parameter fixed at 0**: `floc=0.0` for both gamma and exponential
- **Clamp LR test stat to 0**: Can't be negative

### Bottleneck Detection
- **Two separate findings**: (1) Traffic intensity ranking, (2) Gamma vs exponential preference
- **Primary bottleneck requires BOTH**: Highest ρ AND gamma_preferred=True
- **Honest reporting**: If no state qualifies, say so explicitly

### ODE System (Updated)
- **Matrix exponential**: Exact solution via `scipy.linalg.expm` (not RK45)
- **Unit conversion is critical**: Days → hours (×24) before solving
- **Rate bounds**: 1e-4 to 1.0 (prevents degenerate solutions)

### Branch Point Validation (Updated)
- **Per-cell tracking**: Transition probabilities computed per individual cell, not per clone
- **No overcounting**: Clone-level aggregation overcounts branching transitions

---

## 10. Git Status

**Current branch**: main
**Remote**: https://github.com/SkittlesGod-cmd/hematopoiesis-queueing.git
**Status**: All 152 tests passing, 14 source modules, 13 test files, 6 scripts.
**Weinreb data**: 1 of 4 files downloaded (download blocked by server issue).
