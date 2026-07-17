# Research Project Plan

## Titles

**Paper title:**
*Semi-Markovian Queueing Network Inference of Rate-Limiting Bottlenecks in Hematopoietic Stem Cell Differentiation from Single-Cell Transcriptomic Time-Series*

**ISEF title:**
*A Semi-Markov Queueing Framework for Identifying Rate-Limiting Stages in Hematopoietic Differentiation*

**Field:** Computational Systems Biology / Operations Research (Queueing Theory) / Applied Statistics

**Deadline:** December 1, 2026 (both journal submission and ISEF via NEOSEF)

---

## Core Research Question

Can hematopoietic differentiation be modeled as a semi-Markov queueing network — with empirically-fitted (gamma-distributed) service times rather than the classical memoryless (exponential) assumption — to identify which differentiation stage is the rate-limiting bottleneck, directly from single-cell transcriptomic time-series data?

---

## Biological System

Hematopoiesis (blood cell differentiation from hematopoietic stem cells)

---

## Datasets

### Primary: Weinreb et al. 2020 (GEO: GSE140802)
- 130,887 cells, 25,289 genes (10,220 highly variable after filtering)
- Real timepoints: days 2, 4, and 6
- Lineage-barcoded (LARRY expressed barcode system) — enables direct clonal residence-time estimation for a subset of cells
- Carries the full quantitative analysis: service-rate estimation, gamma fitting, AIC/BIC model comparison, bottleneck detection

### Secondary: Nestorowa et al. 2016
- 1,656 cells, classic 6-state marker-based hierarchy (HSC, MPP, LMPP, CMP, MEP, GMP)
- **Corrected role:** single-timepoint snapshot data — cannot support service-rate estimation or quantitative model comparison (no real time axis). Used strictly as a **structural cross-check**, confirming that the marker-based state schema generalizes to an independently-collected dataset. Not a second full quantitative validation.

### Synthetic
- Multiple simulated differentiation hierarchies with known, continuously-varied bottleneck severity
- Ground-truth parameters set by design, so pipeline recovery accuracy can be directly measured
- Fully self-contained; validates the inference pipeline before it's trusted on real data

---

## State / Schema Design

- Hybrid approach: one shared coarse meta-hierarchy (Stem/Multipotent → Myeloid-primed Progenitor → Lymphoid-primed Progenitor → Committed Progenitor → Mature)
- Each dataset's native resolution nested underneath (Nestorowa's 6 states nest naturally; Weinreb's ~11 mature lineage outcomes nest under "Mature")
- States defined via known marker genes, cross-checked against unsupervised clustering

---

## Residence / Service-Time Estimation

- Direct estimation from lineage-tracing (clonal barcode) data where available (Weinreb subset)
- Hybrid population-flux inference for unbarcoded cells

---

## Statistical Modeling

- **Service-time distribution:** Gamma (nests the exponential as a special case, enabling direct nested-model comparison)
- **Parameter estimation:** Maximum likelihood estimation (MLE)
- **Model comparison / success criterion:** AIC/BIC comparison between gamma-fit semi-Markov model and classical exponential baseline, per differentiation stage
- **Multiple-comparisons correction:** Benjamini-Hochberg (FDR), applied across states and tests

---

## Validation Strategy

### Synthetic (primary rigor check)
Continuous severity sweep across simulated bottleneck scenarios with known ground-truth locations — tests whether the pipeline correctly recovers the true bottleneck under controlled, varied conditions.

### Real data (corrected framing)
Originally scoped as validating against Weinreb et al.'s reported finding of two distinct monocyte differentiation routes. **Corrected:** that finding demonstrates lineage *branching*, not rate-limiting *congestion* — a related but distinct claim from what this project measures. Revised validation target: test whether the model detects elevated traffic intensity / congestion specifically at the branch point where the two monocyte routes diverge, directly connecting the framework's bottleneck-detection claim to the dataset's own documented biology, rather than treating branching itself as proof of bottleneck detection.

### Structural (secondary, from Nestorowa)
Confirms the marker-based state schema is not an artifact of one dataset's specific processing.

---

## Deliverable

Open-source Python package: ingests a labeled `AnnData` object, outputs fitted service-time distributions, AIC/BIC model comparison results, and ranked bottleneck diagnostics. Reusable for other differentiation systems, not a one-off analysis.

---

## Software Stack

`scanpy` (single-cell data handling) · `lifelines` (survival/hazard estimation) · `scipy.stats` (distribution fitting)

---

## Target Venues

- **Paper:** PLOS Computational Biology / Bioinformatics / PLOS ONE
- **Competition:** ISEF via NEOSEF

## Scope Split

- **Paper:** full analysis — both real datasets (Weinreb quantitative + Nestorowa structural), full synthetic severity sweep, complete statistical detail
- **ISEF poster:** narrowed to primary dataset (Weinreb), headline AIC/BIC result, and the revised congestion-at-branch-point validation result

---

## Background Knowledge Roadmap (prerequisite order)

1. Math foundations (probability, calculus, linear algebra)
2. Stochastic processes / Markov chains
3. Queueing theory **and** survival analysis (parallel tracks)
4. Semi-Markov processes (synthesizes 3)
5. Single-cell genomics / pseudotime (parallel track, no dependency on 1–4)
6. Programming implementation (`scanpy`, `lifelines`, `scipy.stats`)

---

## Known, Stated Limitations

- Three real timepoints (Weinreb) is thin for fully reliable gamma shape-parameter estimation per state; synthetic validation is designed to characterize this limitation, not eliminate it — must be stated plainly in the paper.
- Nestorowa's role is intentionally limited to structural validation, not quantitative cross-validation, due to its single-timepoint design.
- December 1 deadline with full two-dataset scope is tight; narrowing Nestorowa's role reduces required work but the timeline remains ambitious and untested against the calendar.
