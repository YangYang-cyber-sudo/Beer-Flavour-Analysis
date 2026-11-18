# Release audit

This document records what was found, recovered, changed and verified while
turning the supplied supplementary archive into this repository. It is the
audit trail for the release; it does not replace `TEST_RECORD.md`.

## 0. Inputs

| Input | SHA-256 |
|---|---|
| `S1_Code_and_Splits.zip` (77 files, 6,665,623 bytes) | `a2a6ed5fcde23d8254ed814b13a976df48b78ea14025e7f49fe3bec08d90db6f` |
| `Beer-Flavour-Analysis_GitHub_Staging.zip` (staging copy) | `a26c73ed5b4c1908ce26b49065bda4ba441f2a59252d67a8dcce05cd774bc361` |
| Manuscript `Manuscript_JFS_v1_clean.docx` | 2,469,804 bytes, 315 paragraphs |
| Supplementary `Supplementary_Material_MicrochemJ (1).docx` | 285,719 bytes |

The S1 archive is the scientific baseline. The staging copy was used only as a
checklist: its `RELEASE_AUDIT.md` identified the missing modules, machine paths,
dependency gaps and supplementary files that this release had to resolve.

The original local analysis workspace (2026-10/11) was searched for every
module imported by S1 and for every file named in the supplementary material.
No file was regenerated from manuscript summary numbers.

## 1. Missing project-local modules — resolved

The staging audit listed six imports that did not exist in S1. All six were
located in the original project and recovered:

| Import in S1 | Recovered from | Placed at |
|---|---|---|
| `mcprep.common` | `mcprep/common.py` | `support/common.py` |
| `mcprep.tabular` | `mcprep/tabular.py` | `support/tabular.py` (already present, byte-identical to the original) |
| `analysis_2026a.audit` | `analysis_2026a/audit.py` | `support/audit.py` (already present, byte-identical) |
| `analysis_2026a.q1.data` | `analysis_2026a/q1/data.py` | `modelling/data.py` (S1 already contained it) |
| `analysis_2026a.q1.models` | `analysis_2026a/q1/models.py` | `modelling/models.py` (S1 already contained it) |
| `analysis_2026a.q1_v2.models` | `analysis_2026a/q1_v2/models.py` | `modelling/increment.py` (`IncrementTransform` and its helper only) |

The import statements were rewritten to the new package names
(`support.*`, `modelling.*`). The S1 code did not contain an `analysis_2026a`
package, so the previous `sys.path` insertion pointed at a directory that does
not exist in the archive; the new layout removes that requirement.

`q1_v2/models.py` also contains `IncrementRegressor` and other development
classes. Only `IncrementTransform` is imported by the S1 scripts; the estimator
code was not copied, so no model definition was changed.

## 2. Machine-specific paths — removed

Five scripts contained the absolute path
`E:/啤酒/MathorCup_2026A_Reproducible_Package/...`:

| Script | Old target | New target |
|---|---|---|
| `analysis/prep_h2h.py` | `analysis_2026a/paper_reanalysis`, `official/附件`, `q1/splits` | `results/`, `data/raw/`, `validation/splits/` |
| `analysis/prep_h2h_addendum.py` | `analysis_2026a/paper_reanalysis` | `results/` |
| `analysis/prep_h2h_r2_metrics.py` | `analysis_2026a/paper_reanalysis` | `results/` |
| `analysis/prep_paper_vs_routine.py` | project root, `table1_analog`, `paper_reanalysis` | `data/derived/q1_final/`, `results/` |
| `analysis/raw_panel_audit.py` | `E:/啤酒/Raw sensory data anonymized (1).csv` | `data/raw/` |

All other paths now derive from `support/paths.py`:

- `BEER_DATA_DIR` (default `data/raw`) — public inputs;
- `validation/splits/` — frozen folds;
- `results/{tables,predictions,reports}` — outputs (`BEER_RESULTS_DIR`
  overrides);
- `analysis/preregistration/` — frozen protocols.

`modelling/__main__.py` defaults changed from `official/附件` and
`modelling/splits` to `data/raw` and `validation/splits`.

The only remaining old-layout strings are provenance notes inside the
preregistration JSON files (`analysis_2026a/q1/splits/outer.csv`, etc.). They
are relative historical paths, not machine paths, and are part of the frozen
protocol text; they were intentionally not rewritten.

## 3. Code changes (complete list)

No model parameter, prediction, split, protocol value or conclusion was
modified. The following changes are path/interface fixes:

1. **Package layout.** `modelling/` is the renamed `analysis_2026a/q1`;
   `analysis/` holds the renamed `reanalysis/` scripts; `support/` holds the
   recovered helper modules; `validation/splits/` holds the frozen splits.
2. **Import rewrites.** `mcprep.*` → `support.*`; `analysis_2026a.audit` →
   `support.audit`; `analysis_2026a.q1.*` → `modelling.*`;
   `analysis_2026a.q1_v2.models` → `modelling.increment`.
3. **Output paths.** Reanalysis outputs are written to
   `results/tables|predictions|reports`; the scripts create those directories.
4. **`prep_paper_vs_routine.py`.** The original `frozen_iterations()` read
   `analysis_2026a/q1_v3_runs/*` joblib files and a CSV that are not in the
   archive. It now reads the same five iteration counts from the archived
   `results/tables/prep_pvr_meta.json` (0: 1760, 1: 1737, 2: 1627, 3: 1182,
   4: 1295; identical to the original values) and falls back to those numbers
   if the file is absent. The published OOF is read from
   `data/derived/q1_final/OOF_depth3_l2_30.csv`, the archived copy of the file
   the original script read.
5. **`prep_paper_vs_routine.py`.** Fixed an undefined name
   (`af30_fallback`) in a fallback branch; the branch is now
   `set(AF_FIELDS)`. The branch only runs if the `af30` flags cannot be parsed.
6. **`raw_panel_audit.py`.** The audit JSON now records the input file name
   (`Raw sensory data anonymized.csv`) instead of the author's local absolute
   path. No count, statistic or conclusion changed.
7. **`modelling/__main__.py`.** `--expected-python` now defaults to `None`
   (it was pinned to `3.12.7`). An exact patch version is not portable across
   clean environments; the version actually used is still written to
   `run_definition.json`. The check is still available via the CLI.
8. **Console encoding.** Added `configure_console()` in `support/paths.py`,
   called by the scripts, so a UTF-8 progress message cannot crash a Windows
   GBK console. File encoding is unchanged.
9. **Documentation and tests.** New `docs/`, package READMEs and
   `tests/smoke_test.py` / `tests/verify_archive.py`. No analysis code was
   added.
10. **`modelling/data.py` metadata index.** `metadata.set_index('field')`
    followed by `.loc[x.columns]` lost the index name in pandas, so the
    per-run `feature_metadata.csv` stored the field names in a column named
    `index`. Added `.rename_axis('field')` so the column remains `field`.
    The values, row order and all model behaviour are unchanged.
11. **Internal wording.** Removed two internal workflow phrases from
    `support/audit.py` and one from the `modelling/config.py` docstring.
    No scientific text or output changed.

## 4. Content and hash preservation

Of the 77 files in S1:

- **60 files are byte-identical** after relocation (all result tables,
  prediction archives, reports, protocol, splits, feature metadata,
  `models.py`, `plots.py`, `__init__.py`).
- **15 files changed**: `modelling/__main__.py`, `modelling/data.py`,
  `modelling/engine.py`, `modelling/config.py` (docstring only), nine
  `analysis/*.py` files, `support/audit.py`, and
  `results/tables/raw_panel_audit.json`. The reasons are listed in §3.
- **2 new code files** were created to restore missing imports:
  `modelling/increment.py` and `support/common.py`.
- The only result file whose bytes changed is
  `results/tables/raw_panel_audit.json`, and only its `source` field
  (the author's original absolute path → `Raw sensory data anonymized.csv`).
  Rerunning the corrected script reproduces the released JSON exactly.
- The S1 `README.md` and `requirements.txt` were replaced by the repository
  README and the pinned environment files.

The frozen splits, protocol and all numerical results keep their original
bytes. `tests/verify_archive.py` confirms that the released tables follow from
the released predictions.

## 5. Supplementary machine-readable files

`docs/MISSING_ASSETS.md` gives the per-file status. In summary:

- **Found and included:** `SAMPLE_SIZE_REFERENCE_REPRODUCTION.csv` (archived
  Q2 stability file), the archived Q2 reference/OOF tables read by
  `error_metrics_reanalysis.py`, the Q1 final per-target table and the Q3 unit
  dictionary.
- **Not found:** `Table_S1_beer_inventory.csv`,
  `Table_S4_grade_distribution.csv`, `Table_S6_style_transfer_by_style.csv`,
  and exact `Table_S3` correlation-matrix exports.
- **Partially available:** the three full correlation matrices are the public
  deposit's `2.csv`, `5.csv` and `6.csv`; they are not redistributed, but they
  are directly downloadable. No derived `Table_S3` export was found.

None of the missing files was recreated from manuscript numbers.

## 6. Dependency declaration

The S1 `requirements.txt` (five unpinned entries) was replaced by
`requirements.txt` with exact pins for the tested clean environment. See
`docs/TEST_RECORD.md` for the installation log and version table.

## 7. Verification summary

| Check | Result |
|---|---|
| Clean venv install from `requirements.txt` | see `TEST_RECORD.md` |
| Import of every package and script | pass |
| Public data shape and alignment checks | pass |
| Frozen split invariants (5 × 50 outer, 3 inner per scope) | pass |
| Protocol seed, 8 families, 57 candidates | pass |
| Archived prediction arrays readable, 250 × 50, finite | pass |
| h2h summary tables reproduce from archived OOF (24 + 32 rows) | pass, max difference 0 |
| Published OOF table ↔ array | pass, max difference 4.4 × 10⁻¹⁶ |
| Table 6 macro R² (0.2608 / 0.2541 / 0.2553 / 0.2551) | pass |
| Error-metric table reproduces from archived Q2 OOF | pass, max RMSE gap 5 × 10⁻⁶ |
| Threshold table reproduces exactly (18 rows) | pass, difference 0 |
| Style recheck reproduces (mean Jaccard 0.162, 17 disjoint) | pass, difference 0 |
| Raw-panel A/B/C tables and audit reproduce exactly | pass |
| h2h R² tables reproduce exactly (1,600 rows) | pass, difference 0 |
| Table 5 per-attribute values match the archived OOF | **not matched** (see §8) |
| Full `prep_h2h.py` / `prep_paper_vs_routine.py` rerun | not executed (heavy) |
| Full `modelling --stage evaluate` rerun | not executed (multi-hour) |

## 8. Known issues and items for the authors

1. **Table 5 provenance.** The per-attribute values printed in Table 5 of the
   manuscript (e.g. Alcohol ρ = 0.846, RMSE = 0.418, R² = 0.695) do not match
   the archived published OOF (`OOF_depth3_l2_30`, Alcohol ρ = 0.838,
   RMSE = 0.400, R² = 0.721). The archived OOF reproduces Table 6 exactly
   (macro R² 0.2608), so the difference is specific to the Table 5 snapshot.
   The source snapshot for Table 5 was not found. The authors should confirm
   which run produced Table 5 before the repository is described as
   reproducing every manuscript number.
2. **Scope.** The repository covers the S1 scope: the nested-CV pipeline and
   the threshold/correspondence/attribution/error/aggregation/preprocessing
   reanalyses. The separate upstream Q2/Q3 development trees (full candidate
   search, style-conditional generation) are not included. Their archived
   outputs that the included scripts read are in `data/derived/`.
3. **Supplementary CSVs.** See `MISSING_ASSETS.md`; four named files are not
   present as such in the archive.
4. **Licence.** `LICENSE` is a placeholder. The authors must choose the code
   licence; the dataset's CC BY 4.0 does not apply to the code.
5. **Heavy reruns.** No claim of a full re-execution is made. The archived
   predictions and tables are verified against each other, not re-fitted.
