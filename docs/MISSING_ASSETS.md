# Missing assets and unresolved provenance

This page lists files that the manuscript or supplementary material refers to
but that were **not found** as original files, together with what was found
instead and what the release does about it. Nothing here was regenerated from
manuscript summary numbers.

## 1. How the search was done

- Read the S1 archive (77 files) and the staging copy.
- Extracted the manuscript (`Manuscript_JFS_v1_clean.docx`) and the
  supplementary material (`Supplementary_Material_MicrochemJ (1).docx`) and
  read every table note that names a machine-readable file.
- Searched the original local MathorCup analysis workspace (including all
  Q1/Q2/Q3 delivery folders and the final revision package) by file name and
  by content for each named file.
- Ran a whole-drive file-name search for `Table_S*`, `*style_transfer*`,
  `*beer_inventory*`, `*grade_distribution*`, `SAMPLE_SIZE_REFERENCE*`, the
  `q1_v3_runs` joblib/search-cost files and `oof_depth3_l2_30_full5.csv`.

## 2. Modules imported by the S1 code

| Import | Status |
|---|---|
| `mcprep.common`, `mcprep.tabular` | **resolved** from the original `mcprep/` package |
| `analysis_2026a.audit` | **resolved** (same bytes as `support/audit.py`) |
| `analysis_2026a.q1.data`, `analysis_2026a.q1.models` | **resolved** (S1 `modelling/`) |
| `analysis_2026a.q1_v2.models.IncrementTransform` | **resolved** into `modelling/increment.py` |

No unresolved import remains in the released code.

## 3. Machine-readable files named in the supplementary material

| File named in the supplement | Status | What was found |
|---|---|---|
| `Table_S1_beer_inventory.csv` | **not found** | The 250-beer inventory can be assembled from the metadata of `1.csv`/`4.csv` (style, ABV, alcohol class), but no archived table with this name or content was found. Not regenerated. |
| `Table_S2` descriptor list | in the DOCX only | Supplement Table S2 lists the 50 keys, modality, flavour source and scale. No separate CSV was named or found; the same fields are in `results/tables/sensory_monitorability.csv` and are listed in `docs/DATA_DICTIONARY.md`. |
| `Table_S3` full correlation matrices | **not found as exports** | The public deposit's `2.csv` (231 × 231 chemical), `5.csv` (50 × 50 sensory) and `6.csv` (44 × 50 chemical–sensory) are the underlying matrices. They are not redistributed here but are downloadable from Zenodo. No derived `Table_S3_*` export with FDR flags was found. |
| `Table_S4_grade_distribution.csv` | **not found** | `results/tables/raw_panel_labels_A/B/C.csv` and `results/tables/raw_panel_audit.json` contain the underlying 0–6 grades and the pooled distribution; the exact attribute-level export named in the note was not archived. |
| `SAMPLE_SIZE_REFERENCE_REPRODUCTION.csv` | **found and included** | Archived Q2 stability file, copied to `data/derived/SAMPLE_SIZE_REFERENCE_REPRODUCTION.csv`. |
| `Table_S6_style_transfer_by_style.csv` | **not found** | Q3 198-unit records exist outside this repository (`q3_fig10_198_revision_20261029/Q3_PLOT_RECORDS.csv`), but no per-style feasibility export with this name was found and the Q3 generation code is outside the S1 scope. Not regenerated. |
| `Table_S7` six-layer attribution test | partially available | `results/tables/q2_*` cover the style-removal and permutation-null layers. The complete six-layer table was produced by the upstream Q2 development tree, which is not part of S1. |
| `Table_S8` per-target performance | available in two derived forms | `results/tables/error_metrics_by_target.csv` (50 targets) and `data/derived/q1_final/Q1_FINAL_PER_TARGET.csv`. |
| `Table_S9` four-arm preprocessing | available | `results/tables/prep_pvr_*` and `results/predictions/prep_pvr_oof.npz`. |

## 4. Development run directories

| Asset | Status | Consequence |
|---|---|---|
| `analysis_2026a/q1_v3_runs/development_v3_folds234_01/search_costs_folds234.csv` | not found | `prep_paper_vs_routine.py` no longer reads it. |
| `analysis_2026a/q1_v3_runs/development_v3_01/outer_{0,1}/depth3_l2_30.joblib` | only outer 2/3/4 joblibs found elsewhere; 0/1 not found | The five frozen iteration counts are archived in `results/tables/prep_pvr_meta.json` and are used instead. |
| `analysis_2026a/q1_v3_runs/v3_full5_summary/oof_depth3_l2_30_full5.csv` | not found under that name | The same predictions were archived as `q1_final_results_20261026/OOF_depth3_l2_30.csv`; verified identical to `prep_pvr_oof.npz['PUBLISHED']` (max difference 4.4 × 10⁻¹⁶). |
| `q3_delivery/01_Generated_Chemistry/field_dictionary.csv` | found under `q3_runs/` | Copied to `data/derived/q3_field_dictionary.csv`. |

## 5. The Table 5 provenance gap

The manuscript's Table 5 reports per-attribute ρ, RMSE, score SD and R². The
archived published OOF does **not** reproduce those values exactly:

| Attribute | Manuscript Table 5 (ρ / RMSE / R²) | Archived `OOF_depth3_l2_30` (ρ / RMSE / R²) |
|---|---|---|
| Alcohol | 0.846 / 0.418 / 0.695 | 0.838 / 0.400 / 0.721 |
| Bitterness | 0.791 / 0.461 / 0.601 | 0.793 / 0.434 / 0.646 |
| Body | 0.741 / 0.391 / 0.521 | 0.742 / 0.378 / 0.552 |
| Acidity | 0.516 / 0.493 / 0.669 | 0.495 / 0.482 / 0.683 |

The same archived OOF **does** reproduce Table 6 exactly (macro R² 0.2608,
0.2541, 0.2553, 0.2551), so the two tables were not produced from the same
prediction snapshot, or Table 5 was transcribed from an earlier run.

No file that reproduces Table 5 was found: not in S1, not in the Q1 final
results, not among the `OOF_l2_30`, `OOF_v1_catboost`, `OOF_v1_soft_blend`,
`OOF_depth3_l2_30` files, and not among the `q1_paper_runs/fixed5_01` OOF
tables. The release therefore does **not** substitute a regenerated Table 5.

**Action for the authors:** identify the run that produced Table 5 (possibly a
100-iteration refit, a different library version, or an earlier draft),
archive its OOF array, and add it to `results/predictions/`. Until then,
`docs/REPRODUCING.md` and the README must not claim that Table 5 is reproduced.

## 6. Repository-scope exclusions

The following are outside the S1 scope and are not included:

- the upstream Q2 candidate-search/attribution pipeline (the Q2 module tree);
- the Q3 style-conditional generation pipeline and its 198-unit run records;
- exploratory Q1 variants (`q1_v2`, `q1_v3`, `q1_v4`, `q1_paper_runs`);
- figure-generation scripts and manuscript DOCX files from the journal
  submission package.

Their archived outputs that the included scripts read are documented in
`data/derived/README.md`.

## 7. Licence

`LICENSE` is a placeholder. The software licence has not been confirmed by the
authors; the dataset's CC BY 4.0 does not cover the code. See
`RELEASE_AUDIT.md` §8.
