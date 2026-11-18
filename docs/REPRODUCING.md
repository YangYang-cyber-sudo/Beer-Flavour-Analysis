# Reproducing the analysis

This page describes a clean-environment reproduction of the repository. It
distinguishes three levels:

1. **Read the archived results** — no model fitting, a few seconds.
2. **Recompute the summaries from the archived out-of-fold arrays** — no model
   fitting, a few seconds; this is the strongest check that the tables and
   predictions form one consistent set.
3. **Re-run the analysis from the public data** — models are refitted; the
   `modelling` pipeline can take hours, while the light reanalysis scripts take
   seconds to minutes.

---

## 1. Environment

Tested with CPython 3.12.14 on Windows 11. `requirements.txt` lists the direct
dependencies with exact pins; `requirements-lock.txt` is the complete
transitive freeze of the tested environment.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# bash:               source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch is used only by the `mlp_torch` arm of `prep_h2h.py`. CatBoost is used
by the paper configuration and by `prep_paper_vs_routine.py`. LightGBM is used
by one candidate family. If a package is missing, the affected family fails
with a recorded `failure.json`; it is never silently replaced by a mean model.

The pipeline writes the interpreter, library versions and compute options into
`run_definition.json` at the start of a run.

---

## 2. Obtain the public data

Follow `docs/DATASET.md`. The short version:

1. Download the Zenodo deposit `10.5281/zenodo.10653704`.
2. Put `1.csv`, `4.csv`, `6.csv`, `8.xlsx` and
   `Raw sensory data anonymized.csv` into `data/raw/`.
3. Optionally verify the SHA-256 values listed in `docs/DATASET.md`.

Alternatively set `BEER_DATA_DIR` to any directory containing those five files:

```bash
# Windows PowerShell
$env:BEER_DATA_DIR = "D:\beer-data"
# bash
export BEER_DATA_DIR=/data/beer
```

`data/raw/` is git-ignored. Nothing in the repository requires the original
machine's directory layout.

---

## 3. Level 1 — read-only checks

```bash
python tests/smoke_test.py
python tests/verify_archive.py
```

`smoke_test.py` checks imports, protocol, split invariants, public-data shapes
(when the data are present) and archive readability. It never writes to the
repository.

`verify_archive.py` recomputes the stored macro metrics from
`results/predictions/*.npz` and compares them with the archived CSV/JSON
tables. It also checks that `prep_pvr_oof.npz['PUBLISHED']` equals
`data/derived/q1_final/OOF_depth3_l2_30.csv` to floating-point precision.

Expected tail of a successful run:

```text
SMOKE_TEST_PASS
ARCHIVE_VERIFICATION_PASS
```

The archive check prints a `NOTE` about the per-attribute Table 5 values; see
`docs/MISSING_ASSETS.md`. That note is expected and is not a failure.

---

## 4. Level 2 — re-run the analysis scripts

These scripts refit no upstream model. They read the public data, the frozen
splits and the archived upstream inputs, and write to `results/` (override with
`BEER_RESULTS_DIR`). Set `BEER_DATA_DIR` first.

| Command | Inputs | Typical time |
|---|---|---|
| `python analysis/threshold_definition_check.py` | `8.xlsx`, threshold profile, field dictionary | < 5 s |
| `python analysis/error_metrics_reanalysis.py` | `4.csv`, `6.csv`, `8.xlsx`, Q2 OOF, monitorability table | < 10 s |
| `python analysis/metric_equivalence_check.py` | `error_metrics_by_target.csv` | < 2 s |
| `python analysis/style_recheck.py` | `1.csv`, `4.csv`, Q2 top-three table | 1–3 min |
| `python analysis/raw_panel_audit.py` | raw panel CSV, `4.csv` | 10–60 s |
| `python analysis/prep_h2h_r2_metrics.py` | two h2h prediction archives | < 5 s |
| `python analysis/prep_h2h.py` | data, splits, preregistration | heavy: 3 arms × 8 models × 5 folds, including PyTorch |
| `python analysis/prep_h2h_addendum.py` | as above plus one extra arm | heavy |
| `python analysis/prep_paper_vs_routine.py` | data, splits, preregistration, Q1 final tables | heavy: CatBoost, 4 arms × 5 folds |

The heavy commands are part of the protocol but were not re-executed
end-to-end for this release; their archived outputs are verified against the
stored predictions. See `docs/TEST_RECORD.md` for the exact commands that were
run and their output.

### Determinism

- Splits are frozen in `validation/splits/`; no seed controls them at run time.
- CatBoost, LightGBM and scikit-learn estimators are seeded from
  `protocol.json` (`seed = 2601731`). Small last-digit differences can still
  appear across library versions or thread counts, especially when early
  stopping selects a slightly different iteration count.
- The paper configuration clips, standardises and selects features inside each
  training fold only; the outer test fold is never touched by any adaptive step.

---

## 5. Level 3 — full nested cross-validation

This is the expensive path. It refits 57 candidates × 3 inner folds per outer
fold, then the selected families on the outer training set.

```bash
python -m modelling --stage runtime --run-dir runs/runtime
python -m modelling --stage evaluate --run-dir runs/evaluate --threads 8
```

`evaluate` writes, per outer fold:

- `outer_<k>/predictions.npz` — OOF predictions for every candidate policy;
- `outer_<k>/selection_and_costs.json` — selected family, preprocessing and timings;
- `outer_<k>/search/` — inner-fold caches, per-candidate scores and failures;
- `summary.csv`, `metrics_per_target*.csv`, `oof/*.csv` and diagnostic figures.

To add the final deployment fit after `evaluate` completes:

```bash
python -m modelling --stage fit-final --run-dir runs/evaluate
```

`--stage predict` needs `--predict-input` and a new `--prediction-dir`; it
refuses to write into an input directory.

### Interruptions and resume

A run directory is append-only. If a run is interrupted, re-issue the same
command with `--resume`. The pipeline refuses to resume if the protocol,
library versions, compute options, data snapshot, source code or frozen splits
changed; start a new run directory instead. This is deliberate: a result must
not silently mix two configurations.

### Expected compute

The exact time depends on the machine. On a 8-thread desktop CPU the full
`evaluate` stage is a multi-hour job; the inner caches let `--resume` continue
without recomputing completed fits.

---

## 6. What is verified in this release

| Check | Status |
|---|---|
| All modules import in a clean venv | verified |
| Public data shapes 250 × 231 / 250 × 50 | verified |
| Frozen outer/inner splits load and satisfy invariants | verified |
| Protocol seed/families are complete | verified |
| Three prediction archives load (250 × 50, no NaN) | verified |
| Archived h2h summary tables reproduce from archived OOF | verified |
| Archived published OOF equals `OOF_depth3_l2_30.csv` | verified (max diff 4.4 × 10⁻¹⁶) |
| Table 6 macro R² reproduced (0.2608 / 0.2541 / 0.2553 / 0.2551) | verified |
| Error-metric table reproduces from archived Q2 OOF | verified |
| Raw-panel A/B/C label tables and audit | verified |
| Per-attribute Table 5 values match the archived OOF | **not matched**; source snapshot missing |
| `prep_h2h.py` full rerun | not executed (heavy) |
| `prep_paper_vs_routine.py` full rerun | not executed (heavy) |
| `modelling --stage evaluate` full rerun | not executed (multi-hour) |

A check is only called verified here if `docs/TEST_RECORD.md` records the
command and its successful output. Nothing in this repository should be read as
claiming that the heavy reruns were performed.

---

## 7. Troubleshooting

**`FileNotFoundError: Missing raw input file(s) …`**
The data directory does not contain the five files. See `docs/DATASET.md`.

**`ImportError: No module named 'catboost'` (or `torch`, `lightgbm`)**
Install the affected package. `requirements.txt` includes all of them.

**`ValueError: 官方数据结构与已确认版本不同`**
`1.csv` or `4.csv` is not the expected Zenodo table, or a metadata column was
dropped. Re-download the deposit instead of editing the table.

**`ValueError: 冻结折被修改`**
`validation/splits/` was edited. Restore the three archived files; do not
regenerate the folds.

**`ValueError: 运行目录已存在；续跑需明确 --resume`**
The pipeline never overwrites a run. Use a new directory or pass `--resume`.
