# Release test record

Date: 2026-11-18 (Asia/Shanghai)
Host: Windows 11 (10.0.19045), 8-thread CPU
Repository under test: this directory, copied to a separate clean test
directory; the public data were supplied from a separate directory through
`BEER_DATA_DIR`.

Nothing in this record is inferred: each line is a command that was run and
its observed output. Checks that were **not** run are listed explicitly at the
end.

---

## 1. Clean environment

A fresh virtual environment was created from CPython 3.12.14:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Installed direct dependencies:

| Package | Version |
|---|---|
| Python | 3.12.14 |
| numpy | 2.2.6 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.6.1 |
| catboost | 1.2.10 |
| lightgbm | 4.6.0 |
| matplotlib | 3.10.3 |
| openpyxl | 3.1.5 |
| joblib | 1.5.1 |
| torch | 2.7.1 (CPU build, `2.7.1+cpu`) |

`python -m pip check` reported no broken requirements. The complete transitive
freeze is `requirements-lock.txt`.

A second, independent virtual environment was then created and populated with
**only** `python -m pip install -r requirements.txt`; `pip check` again
reported no broken requirements and the import line
`numpy pandas scipy sklearn catboost lightgbm matplotlib openpyxl joblib torch`
printed `REQ_ENV_OK`. The smoke and archive tests below were re-run from a
fresh copy of the final tree with that second environment and passed.

The public input files were placed in a directory outside the repository and
pointed to with `BEER_DATA_DIR`. Their SHA-256 values matched
`docs/DATASET.md`.

---

## 2. Smoke test

```text
> python tests/smoke_test.py

repository : ...\Beer-Flavour-Analysis
data dir   : ...\testdata
python     : 3.12.14

PASS  import modelling and support packages  [modelling, support]
PASS  frozen protocol is complete  [57 candidates, 8 families]
PASS  frozen split definition  [seed 2601731, beer-level, 5x3, labels not used]
PASS  public data loads  [250 x 231, 250 x 50]
PASS  frozen splits load  [5 outer folds x 50 beers, 3 inner folds per scope]
PASS  recovered IncrementTransform runs  [fold 0 transform -> 462 columns]
PASS  feature metadata aligns  [37 log10-transformed features]
PASS  archived prediction archives  [3 archives, 250 x 50]
PASS  archived result tables  [6 key tables readable]

passed=9 failed=0 skipped=0
SMOKE_TEST_PASS
```

The `IncrementTransform` check fits the recovered transform on outer fold 0's
training beers and transforms the 50 test beers; 462 columns = 231 features +
231 missing indicators for the `A_plain` configuration. The script's own
runtime assertion also compared the recovered transform with the legacy
`FoldTransform`:

```text
H2H_TRANSFORM_ASSERT_OK
```

(maximum absolute difference ≤ 1e-12 over the fold-0 test matrix).

---

## 3. Archive verification

```text
> python tests/verify_archive.py

PASS  prep_h2h summary matches archived OOF  [24 arm x model rows recomputed]
PASS  prep_h2h R2 summary matches archived OOF  [32 arm x model rows recomputed]
PASS  published OOF table equals PUBLISHED array  [max |difference| = 4.4e-16]
PASS  Table 6 macro R2 reproduced  [W_paper=0.2608, R_control=0.2541, R_fold=0.2553, R_global=0.2551]
PASS  error metrics table matches archived OOF  [50 targets, max |RMSE gap| = 5.0e-06]
PASS  frozen split invariants  [250 beers, 5 balanced outer folds, 6 inner scopes]
PASS  raw panel label reconstruction  [250 beers x 50 descriptors, 2,197 records]

NOTE  Archived PUBLISHED OOF reproduces the Table 6 macro R2 (0.260843) but not every per-attribute value printed in Table 5 of the manuscript.
NOTE  See docs/MISSING_ASSETS.md; this release does not substitute a regenerated Table 5.

passed=7 failed=0
ARCHIVE_VERIFICATION_PASS
```

---

## 4. Reanalysis scripts re-run from the public data

All commands below were run in the clean test copy with `BEER_DATA_DIR` set to
the public-data directory. Outputs were compared with the archived files in
this release.

### 4.1 Threshold definition check

```text
> python analysis/threshold_definition_check.py
saved 18 rows -> results/tables/threshold_definition_check.csv
```

Comparison with the archived table: 18 rows, 13 columns, **maximum numeric
difference 0.0**; all string columns identical.

### 4.2 Error-metric decomposition

```text
> python analysis/error_metrics_reanalysis.py
```

Comparison: `error_metrics_by_target.csv` 50 rows, maximum difference
1.1 × 10⁻¹⁶; `error_metrics_by_group.csv` 13 rows, maximum difference 0.0;
`error_metrics_summary.json` numerically identical (up to floating-point
representation).

### 4.3 Metric equivalence

```text
> python analysis/metric_equivalence_check.py
saved -> results/tables/metric_equivalence_check.json
```

Comparison: numerically identical; `max_abs_gap_percentage_points = 0.005512`
in both.

### 4.4 Style recheck

```text
> python analysis/style_recheck.py
```

Elapsed: 50 s. Result in the regenerated summary:

```text
targets=50, mean_jaccard=0.162, median_jaccard=0.200,
identical=0, disjoint=17, partial=33, mean_jaccard_strict=0.150,
disjoint_strict=17, n_styles=22
```

Comparison: `q2_within_style_top3.csv`, `q2_full_pool_top3.csv`,
`q2_null_baselines.csv` all differ by 0.0; `q2_style_recheck_summary.json` and
`q2_null_baselines.json` are identical.

### 4.5 Raw-panel audit

```text
> python analysis/raw_panel_audit.py
outputs: raw_panel_audit.json, raw_panel_beer_map.csv,
         raw_panel_labels_A.csv, raw_panel_labels_B.csv,
         raw_panel_labels_C.csv, raw_panel_reconstruction_check.json
```

Elapsed: 1.4 s. `n_records=2197`, `n_beers=250`, `n_tasters=16`,
`n_scored_cells=109850`, `all_integer_0_to_6=true`,
`n_taster_descriptor_zero_variance_combos=15`,
`reconstruction_independently_reproduced=false`.

Comparison: all four CSV outputs are exactly equal to the archived files; both
JSON files are exactly equal (including `source`).

### 4.6 h2h R² recomputation

```text
> python analysis/prep_h2h_r2_metrics.py
输出: prep_h2h_r2_summary.csv / _by_target.csv / _paired.csv / _stratified.csv
```

Comparison: 32 / 1,600 / 1,600 / 16 rows, maximum numeric difference
1.1 × 10⁻¹⁶, all rows equal.

---

## 5. Pipeline runtime capture

```text
> python -m modelling --stage runtime --run-dir runs/runtime
RUNTIME {"python": "3.12.14", "numpy": "2.2.6", "pandas": "2.2.3",
         "openpyxl": "3.1.5", "scipy": "1.15.3", "sklearn": "1.6.1",
         "joblib": "1.5.1", "matplotlib": "3.10.3", "catboost": "1.2.10",
         "lightgbm": "4.6.0"}
Q1_RUNTIME_PASS
```

This confirms that the pipeline entry point imports and that every declared
library is present. A full `--stage evaluate` was not run (see §7).

---

## 6. Heavy script data paths

The expensive optimisation loops were not executed, but the repaired data and
configuration paths were exercised:

```text
> python -c "import analysis.prep_h2h, analysis.prep_h2h_addendum, analysis.prep_paper_vs_routine"
HEAVY_IMPORTS_OK

> python -c "... prep_paper_vs_routine.load_data / load_splits / frozen_iterations ..."
[13:45:05] 冻结迭代数: {0: (1760, 'prep_pvr_meta.json'), 1: (1737, 'prep_pvr_meta.json'),
                        2: (1627, 'prep_pvr_meta.json'), 3: (1182, 'prep_pvr_meta.json'),
                        4: (1295, 'prep_pvr_meta.json')}
PVR_DATA_OK (250, 231) (250, 50)
```

The five frozen iteration counts are read from the archived
`results/tables/prep_pvr_meta.json` and match the original development values.

---

## 7. What was not executed

| Command | Reason |
|---|---|
| `python -m modelling --stage evaluate` | multi-hour search (57 candidates × 3 inner folds × 5 outer folds); no claim of a full rerun |
| `python analysis/prep_h2h.py` | 3 arms × 8 models × 5 folds, including PyTorch |
| `python analysis/prep_h2h_addendum.py` | one extra arm over the same grid |
| `python analysis/prep_paper_vs_routine.py` | CatBoost, 4 arms × 5 folds |

Their archived outputs are verified against the stored predictions by
`tests/verify_archive.py`; the scripts themselves were imported and their data
paths were exercised. They are **not** described as fully re-executed.

---

## 8. Known failure modes checked

- A GBK Windows console crashed the unmodified script on a `²` character.
  After adding `configure_console()`, `prep_h2h_r2_metrics.py` completed under
  the same console. File encoding is unchanged.
- Running `python -m modelling` from outside the repository root fails with
  `No module named modelling`; this is expected. The documented commands run
  from the repository root.
- The pipeline refuses to write into an existing run directory; the test used
  a fresh `runs/runtime` directory.

---

## 9. Reproduction of archived tables: summary

| Output | Max numeric difference from archive |
|---|---|
| `threshold_definition_check.csv` | 0.0 |
| `error_metrics_by_target.csv` | 1.1 × 10⁻¹⁶ |
| `error_metrics_by_group.csv` | 0.0 |
| `error_metrics_summary.json` | numerically equal |
| `metric_equivalence_check.json` | numerically equal |
| `q2_within_style_top3.csv` | 0.0 |
| `q2_full_pool_top3.csv` | 0.0 |
| `q2_null_baselines.csv` | 0.0 |
| `q2_style_recheck_summary.json` | equal |
| `q2_null_baselines.json` | equal |
| `raw_panel_labels_A/B/C.csv` | 0.0 |
| `raw_panel_beer_map.csv` | 0.0 |
| `raw_panel_audit.json` | equal |
| `raw_panel_reconstruction_check.json` | equal |
| `prep_h2h_r2_summary.csv` | 0.0 |
| `prep_h2h_r2_by_target.csv` | 0.0 |
| `prep_h2h_r2_paired.csv` | 1.1 × 10⁻¹⁶ |
| `prep_h2h_r2_stratified.csv` | 0.0 |

The only unexplained result is the manuscript's Table 5 per-attribute snapshot;
see `MISSING_ASSETS.md`.
