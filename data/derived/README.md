# Archived upstream inputs

These files are intermediate outputs of the upstream Q1/Q2 analysis. They are
included because the reanalysis scripts in `analysis/` read them. They are
**not** new results produced for this release, and none of them contains raw
measurements.

| Path | Original location in the project | Why it is here |
|---|---|---|
| `q2_reference/q1_oof_truth_continuous.csv` | `analysis_2026a/q2/reference/` | truth table for `analysis/error_metrics_reanalysis.py` |
| `q2_eligible_only/crossfit_oof_ridge.csv` | `analysis_2026a/q2/local_eligible_only_20261024/` | ridge out-of-fold predictions for the error decomposition |
| `q2_eligible_only/crossfit_oof_extra.csv` | same | extra-trees out-of-fold predictions |
| `q2_eligible_only/crossfit_metrics.csv` | same | run-level metrics used as a cross-check |
| `q2_eligible_only/top3_all_available.csv` | same | top-three candidate lists for the style recheck |
| `q1_final/OOF_depth3_l2_30.csv` | `q1_final_results_20261026/` | published wide-format OOF predictions |
| `q1_final/Q1_FINAL_PER_TARGET.csv` | `q1_final_results_20261026/` | per-target metrics and target-scope flags |
| `q3_field_dictionary.csv` | `analysis_2026a/q3_runs/final_smoke_01/` | unit dictionary for the threshold unit check |
| `SAMPLE_SIZE_REFERENCE_REPRODUCTION.csv` | `q2_stability_figure_20261027/inputs/key_tables/` | learning-curve evidence named in Supplementary Table S5 |

## Integrity

These files were copied byte-for-byte from the archived analysis tree; their
SHA-256 values are listed in the repository-level `SHA256SUMS.txt`. They were
not regenerated from manuscript summary numbers.

## Regenerating them

The upstream Q1/Q2 search and cross-fit code is not part of this repository
(see `docs/MISSING_ASSETS.md`). Do not overwrite these files by hand; if the
upstream pipeline is rerun, archive the new outputs under a new directory and
document the change.
