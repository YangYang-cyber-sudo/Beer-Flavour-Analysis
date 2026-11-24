# Beer-Flavour-Analysis

Research software and archived results for:

> **Above-threshold flavour compounds and trained-panel scores in commercial beers:
> weak correspondence for terpenoids and sulfur volatiles**
> Yang Yang, Yuxuan Jiang, Yalu Cai, Honghao Cai
> School of Science, Jimei University, Xiamen, Fujian 361021, China
> *Microchemical Journal* (submitted)

This repository contains the nested cross-validation pipeline, the frozen
five-fold outer / three-fold inner splits, the reanalysis scripts, the archived
out-of-fold predictions and the derived tables behind the manuscript. The raw
measurements are **not** redistributed; they are downloaded from the public
Zenodo deposit of Schreurs et al. (2024).

---

## 1. Research question and main findings

Beer quality control relies on routine chemical measurements and on trained
sensory panels, but it is unresolved which sensory attributes a chemical
measurement can legitimately monitor. Using 250 commercial beers characterised
by 231 chemical properties and 50 trained-panel descriptors, the study asks
whether exceeding a compound's literature sensory threshold is sufficient
evidence that the compound can serve as an instrument-based monitor.

Main findings:

- **Threshold status alone does not establish monitorability.** Ethanol and
  iso-alpha acids were above threshold in 97.6% and 86.4% of beers and tracked
  their descriptors closely (Spearman ρ = 0.82 and 0.68). Linalool,
  citronellol and dimethyl sulfide were above threshold in 62–100% of beers yet
  showed little correspondence with their descriptors (ρ = 0.09–0.11).
- **Significance is not predictive strength.** Rank correspondence between
  out-of-fold predictions and panel scores was significant for 49 of 50
  attributes, but prediction error was similar for well- and weakly supported
  attributes (RMSE 0.457 vs. 0.477 panel units, P = 0.27); the contrast tracked
  the spread of the panel scores rather than model error.
- **Candidate lists are unstable.** Agreement between three attribution
  principles (univariate rank correlation, conditional linear coefficients,
  tree permutation importance) was low (mean top-three Jaccard 0.15–0.26), and
  only 10 of 150 top-three links recurred in ≥ 80% of 50 beer-level bootstrap
  resamples.
- **Exact grade reproduction is modest.** On the panel's original 0–6 integer
  scale, chemical profiles reproduced the exact grade for 53.60% of
  beer–attribute pairs, 7.67 percentage points above a training-fold mean
  baseline.

The practical conclusion is a three-way allocation of monitoring
responsibility: instrument-based monitoring where the chemical measurement is
the physicochemical basis of the attribute; threshold screening as a release
gate only for defect compounds that normally sit below threshold; and trained
panel assessment for fine flavour and defect descriptors. Any reassignment
should be preceded by a panel-correspondence check.

---

## 2. Data source

The measurements come from the public dataset of Schreurs et al. (2024),
*Nature Communications* 15:2368, deposited under CC BY 4.0:

- **DOI:** [10.5281/zenodo.10653704](https://doi.org/10.5281/zenodo.10653704)
- **Paper:** [10.1038/s41467-024-46346-0](https://doi.org/10.1038/s41467-024-46346-0)

Place the required files in `data/raw/` (or set `BEER_DATA_DIR`). The file
list, shapes and SHA-256 checksums are in [`docs/DATASET.md`](docs/DATASET.md)
and [`data/README.md`](data/README.md). No measurement data are stored in this
repository.

The dataset is CC BY 4.0. That licence covers the data, not this repository's
code; see [License](#license).

---

## 3. Repository structure

```text
Beer-Flavour-Analysis/
├── README.md
├── CITATION.cff
├── LICENSE                     # placeholder: authors' licence decision pending
├── requirements.txt
├── .gitignore
├── docs/
│   ├── DATASET.md              # source files, shapes, checksums, download steps
│   ├── REPRODUCING.md          # clean-environment reproduction instructions
│   ├── DATA_DICTIONARY.md      # every input, split, prediction and result field
│   ├── RELEASE_AUDIT.md        # what was recovered, fixed and verified
│   ├── MISSING_ASSETS.md       # assets referenced but not found in the archive
│   └── TEST_RECORD.md          # commands and output of the release tests
├── modelling/                  # nested-CV pipeline (data, protocol, search, engine)
│   ├── data.py                 # X/Y alignment and frozen-split loader
│   ├── config.py               # the 57-candidate search grid (frozen before scoring)
│   ├── protocol.json           # the grid as frozen
│   ├── models.py               # fold-internal preprocessing + 8 model families
│   ├── engine.py               # inner selection, outer evaluation, atomic caches
│   ├── increment.py            # IncrementTransform arm used by the preprocessing study
│   ├── plots.py                # diagnostic figures
│   ├── __main__.py             # CLI: runtime / evaluate / fit-final / predict
│   └── feature_metadata.csv    # unit, transformation state and role of 231 features
├── analysis/                   # reanalysis scripts
│   ├── prep_h2h.py             # 3-arm preprocessing head-to-head (pre-registered)
│   ├── prep_h2h_addendum.py    # post-hoc fair-baseline arm
│   ├── prep_h2h_r2_metrics.py  # R² / RMSE recomputation from the frozen OOF arrays
│   ├── prep_paper_vs_routine.py# paper configuration vs three routine alternatives
│   ├── raw_panel_audit.py      # raw 0–6 records, aggregation A/B/C, reconstruction check
│   ├── style_recheck.py        # style-stratified attribution recheck and null model
│   ├── error_metrics_reanalysis.py # RMSE / nRMSE / SD decomposition
│   ├── threshold_definition_check.py # threshold mean vs median sensitivity
│   ├── metric_equivalence_check.py   # R² vs error-reduction equivalence
│   └── preregistration/        # protocols frozen before the corresponding runs
├── support/                    # shared helpers (paths, JSON, table readers, audit)
├── data/
│   ├── README.md               # how to obtain the five public input files
│   └── derived/                # archived upstream inputs read by the reanalysis
├── validation/
│   └── splits/                 # frozen outer.csv, inner.csv, definition.json
├── results/
│   ├── tables/                 # archived derived tables (CSV/JSON)
│   ├── predictions/            # archived out-of-fold arrays (NPZ)
│   └── reports/                # archived analysis reports (Markdown)
└── tests/
    ├── smoke_test.py           # imports, data, splits and archive readability
    └── verify_archive.py       # recomputes summaries from the stored OOF arrays
```

A generated, complete listing is in [`docs/FILE_TREE.txt`](docs/FILE_TREE.txt).

---

## 4. Software environment

- Python 3.12 (tested with CPython 3.12.14)
- NumPy, pandas, SciPy, scikit-learn, CatBoost, LightGBM, Matplotlib, OpenPyXL,
  joblib and PyTorch (CPU) — direct pins in [`requirements.txt`](requirements.txt),
  complete tested freeze in [`requirements-lock.txt`](requirements-lock.txt)

Create a clean environment with:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pipeline records the exact Python/library versions in each run's
`run_definition.json`. The analysis scripts depend only on NumPy, pandas,
SciPy and scikit-learn; CatBoost is needed by `prep_paper_vs_routine.py`, and
PyTorch only by the `mlp_torch` arm of `prep_h2h.py`.

---

## 5. Running the code

Download the five public files into `data/raw/` first (see
[`data/README.md`](data/README.md)), then:

```bash
# 1. environment and data check (no model fitting)
python tests/smoke_test.py

# 2. verify that the archived tables follow from the archived OOF arrays
python tests/verify_archive.py

# 3. pipeline runtime capture (writes only a runtime record)
python -m modelling --stage runtime --run-dir runs/runtime

# 4. full nested cross-validation (8 model families, 5 outer x 3 inner folds)
python -m modelling --stage evaluate --run-dir runs/evaluate

# 5. reanalysis scripts
python analysis/threshold_definition_check.py
python analysis/error_metrics_reanalysis.py
python analysis/metric_equivalence_check.py
python analysis/style_recheck.py
python analysis/raw_panel_audit.py
python analysis/prep_h2h.py                 # heavy: 3 arms x 8 models x 5 folds
python analysis/prep_h2h_addendum.py        # heavy: one extra arm
python analysis/prep_h2h_r2_metrics.py      # recomputation only
python analysis/prep_paper_vs_routine.py    # heavy: CatBoost, 4 arms x 5 folds
```

`modelling/__main__.py` refuses to overwrite an existing run directory, checks
that the protocol seed matches the frozen split definition, and never passes
outer-fold labels into the inner search. Reanalysis scripts write their outputs
to `results/` by default; set `BEER_RESULTS_DIR` to write elsewhere.

Detailed instructions, expected runtimes and the verified/not-verified matrix
are in [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

---

## 6. Results files

| Path | Content |
|---|---|
| `results/predictions/prep_pvr_oof.npz` | Out-of-fold predictions for the published configuration plus the three preprocessing arms, all 50 descriptors. |
| `results/predictions/prep_h2h_oof.npz` | Three pre-registered preprocessing arms × eight fixed models. |
| `results/predictions/prep_h2h_addendum_oof.npz` | Adds the post-hoc `A_clip` fair-baseline arm. |
| `results/tables/prep_pvr_summary.csv` | Table 6 macro metrics by arm and target scope. |
| `results/tables/error_metrics_by_target.csv` | Per-attribute RMSE, MAE, R², nRMSE, SD, rank correlation and FDR. |
| `results/tables/compound_threshold_profile.csv` | Table 1 threshold profile of 18 compounds. |
| `results/tables/threshold_definition_check.csv` | Mean-vs-median threshold sensitivity. |
| `results/tables/sensory_monitorability.csv` | Attribute-level monitorability summary. |
| `results/tables/raw_panel_{A,B,C}.csv` | Three beer-level 0–6 aggregation rules reconstructed from the raw records. |
| `results/tables/q2_*.csv` | Style-stratified attribution recheck and null baselines. |

Every prediction archive stores `y` (250 × 50 observed scores), `outer` (250
fold labels) and `targets` (50 names), followed by one 250 × 50 array per
prediction source. Arrays are aligned by beer and target order. Field-level
documentation is in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

---

## 7. Reproducibility status

This release is an **archival, traceable snapshot**, not a claim that every
number can be regenerated in a few minutes:

- Verified in a clean virtual environment: package imports, public-data
  shapes, frozen-split invariants, protocol consistency, prediction-archive
  integrity, and numerical agreement between the archived tables and the
  archived out-of-fold arrays (`tests/smoke_test.py`,
  `tests/verify_archive.py`).
- Archived OOF predictions reproduce the Table 6 macro metrics exactly
  (0.2608 / 0.2541 / 0.2553 / 0.2551 macro R²).
- The per-attribute values printed in Table 5 of the manuscript do **not**
  match the archived published OOF exactly; the source snapshot for Table 5
  was not found in the supplied archive. This is recorded in
  [`docs/MISSING_ASSETS.md`](docs/MISSING_ASSETS.md) and no substitute table was
  generated.
- The two heavy scripts (`prep_h2h.py`, `prep_paper_vs_routine.py`) were not
  re-executed end-to-end in this release; their archived outputs are verified
  against the stored predictions. See [`docs/TEST_RECORD.md`](docs/TEST_RECORD.md).

No code path is described as verified unless the corresponding test ran.

---

## 8. Citation

If you use this code or the derived tables, please cite the manuscript. A
machine-readable draft is provided in [`CITATION.cff`](CITATION.cff); add the
final article DOI and the Zenodo software DOI after they are assigned.

```text
Yang, Y., Jiang, Y., Cai, Y., & Cai, H. (2026).
Above-threshold flavour compounds and trained-panel scores in commercial beers:
weak correspondence for terpenoids and sulfur volatiles.
Manuscript submitted to Microchemical Journal.
```

Please also cite the source dataset:

```text
Schreurs, M., et al. (2024). Predicting and improving complex beer flavor
through machine learning. Nature Communications, 15, 2368.
https://doi.org/10.1038/s41467-024-46346-0
```

---

## 9. License

The software licence has **not** been decided by the authors. `LICENSE`
currently grants no rights beyond viewing the repository; do not redistribute
or reuse the code until the authors confirm a licence. The CC BY 4.0 licence of
the source dataset does not apply to this repository's code.

---

## 10. Contact

Honghao Cai, School of Science, Jimei University — hhcai@jmu.edu.cn

## 11.Supplementary Materials

The supplementary materials associated with this study are provided
in the `supplementary/` directory.

- `Supplementary_Material_JFS.docx`: Supplementary Tables S1–S9
  and Supplementary Figure S1.
- `S1_Code_and_Splits.zip`: Analysis code, frozen cross-validation
  splits, out-of-fold predictions for all 50 sensory descriptors,
  and supporting analysis scripts and outputs.

The original public dataset is not redistributed.
