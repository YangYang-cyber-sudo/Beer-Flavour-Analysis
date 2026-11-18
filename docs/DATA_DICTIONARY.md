# Data dictionary

This page documents every file type in the repository: the two public input
tables, the frozen splits, the modelling protocol, the prediction archives and
the derived result tables. All scientific content is unchanged from the
archived analysis; only directory locations and import paths were adapted for
this release (see `RELEASE_AUDIT.md`).

---

## 1. Conventions

- **Beer order.** All arrays and tables are ordered by `beer_id` as an integer,
  i.e. the 250 beers sorted 1 … 250. `modelling/data.py` enforces this order
  for X and Y before any model is fitted.
- **Target order.** The 50 sensory descriptors follow the column order of
  `4.csv` after removing the metadata columns.
- **Feature order.** The 231 chemical properties follow the column order of
  `1.csv` after removing the metadata columns, and are the row order of
  `modelling/feature_metadata.csv`.
- **Panel units.** Continuous sensory scores are the processed labels
  distributed with the dataset (beer-level, centred within descriptor). They are
  not the original 0–6 integer grades; the two representations are related but
  not interchangeable.
- **Encoding.** CSV files use UTF-8 with BOM (`utf-8-sig`) where the original
  analysis wrote them; readers should use `encoding="utf-8-sig"`.
- **Frozen means frozen.** `validation/splits/` is the only source of folds.
  No script regenerates it; `modelling.data.make_splits` refuses to overwrite an
  existing split directory.

---

## 2. Chemical matrix (`1.csv`)

- 250 rows, one per beer.
- Metadata columns: `beer_id`, `tasting_category_fine`.
- 231 numeric columns: alcohols, esters, acids, carbonyls, sulfur compounds,
  terpenes, phenols, plus aggregate measures (IBU/iso-alpha acids, total
  protein, colour, pH, original/final gravity, ABV, etc.).
- 70 columns have a calibrated concentration in a stated physical unit; 18 of
  those also have a numeric literature threshold and form the threshold
  analysis.
- Missing values are encoded as `NA` or an empty cell. `-1.81781854` is a
  documented sentinel (repeated low value) and is retained by the paper
  configuration through explicit indicator columns.

### `modelling/feature_metadata.csv` (231 rows)

| Column | Meaning |
|---|---|
| `field` | column name in `1.csv` |
| `unit_raw_metadata` | unit recorded in the source metadata (blank if none) |
| `calibrated_metadata` | whether a calibrated concentration exists (`yes`/`no`) |
| `chemical_display_name` | human-readable compound name |
| `origin_metadata` | process/biosynthetic group used for group-balanced preprocessing |
| `threshold_mean_raw_unit` | compiled mean literature threshold, raw unit |
| `threshold_median_raw_unit` | compiled median literature threshold, raw unit |
| `raw8_available` | whether the concentration is present in `8.xlsx` |
| `one_missing`, `one_negative`, `one_zero` | counts of those conditions in the primary calibration table |
| `transform_status` | `log10_positive_half_min_for_nonpositive` or `mostly_identity_with_exceptions`; drives the signed-log transform |
| `positive_log10_max_abs_error` | maximum absolute error of the log10 reconstruction |
| `nonpositive_raw_count` | number of non-positive calibrated values |
| `replacement_raw` | value used for non-positive entries before log10 |
| `raw_missing` | missing count in the raw table |
| `comparison_tolerance` | tolerance used when comparing duplicate exports |
| `finite_mismatch_count`, `raw_missing_but_one_present`, `raw_present_but_one_missing` | duplicate-export consistency audit |

`origin_metadata` defines the eight broad groups used by the group-balanced
preprocessing candidate; empty values become `Unspecified`.

---

## 3. Sensory matrix (`4.csv`)

- 250 rows, one per beer.
- Metadata columns: `beer_id`, `tasting_category_fine`.
- 50 numeric descriptors on the processed continuous scale.
- The 50 keys (also the `targets` array in every NPZ archive):

```text
alcohol, acidity, lactic, bitternes, acetic, body, aftertaste,
F_hops_all, F_malt_all, A_malt_all, F_hops_noble, F_malt_burn,
A_hops_all, co2, A_malt_cara, F_malt_cara, A_hops_noble,
F_esters_all, sweetness, F_esters_ethac, F_malt_bread, A_malt_burn,
X4vg, F_hops_woody, A_esters_all, F_hops_citrus, A_esters_ethac,
coriander, orange, A_hops_citrus, A_hops_woody, metallic,
A_hops_tropical, F_esters_isoaa, F_hops_tropical, A_malt_bread,
overall, dms, diacetyl, F_malt_grain, barnyard, A_esters_isoaa,
A_esters_flower, F_esters_fruity, A_malt_grain, stale_hops,
F_esters_flower, A_esters_fruity, t2n, clove
```

Prefixes: `A_` = aroma, `F_` = taste/retronasal flavour. Modality counts:
Aroma 15, Taste 30, Palate 4, Overall impression 1.

`results/tables/sensory_monitorability.csv` gives, per target:
`target`, `clean_name`, `modality`, `flavour_source`, `r2_ridge`,
`r2_extra`, `r2_mean`, `weak_below_0_1`, `monitorability`. The 30
aroma/flavour keys are also listed in `modelling/data.py::AF_FIELDS`.

---

## 4. Frozen validation splits (`validation/splits/`)

| File | Columns | Content |
|---|---|---|
| `definition.json` | — | `seed: 2601731`, `unit: beer`, `outer: 5`, `inner: 3`, `label_access_for_split_generation: false`, note on scope `-1` |
| `outer.csv` | `beer`, `outer_fold` | 250 rows, five folds of exactly 50 beers |
| `inner.csv` | `scope`, `beer`, `inner_fold` | for each outer fold 0–4, the three inner folds of its 200 training beers; scope `-1` gives the three inner folds over all 250 beers for the final deployment fit |

Invariants enforced on load (`modelling.data.load_splits`):

1. every beer appears exactly once in `outer.csv`;
2. outer folds are five equal, disjoint folds;
3. for each scope, the inner table contains exactly the outer-training beers
   (or all beers for `-1`) and three inner folds;
4. the inner table never contains a beer from the corresponding outer test fold.

The splits were generated before any label was accessed; the flag
`label_access_for_split_generation: false` is part of the definition.

---

## 5. Modelling protocol (`modelling/protocol.json`)

- `version`: `2026A_Q1_v1`
- `seed`: 2601731 (must equal the split seed)
- `outer_folds`: 5, `inner_folds`: 3
- `focus_families`: 3 (coarse-stage winners promoted to the refine stage)
- `selection`: macro normalised RMSE (by score SD) plus 0.2 × the mean of the
  worst 20 % of targets
- `candidates`: 57 candidate specifications over 8 families —
  `mean`, `ridge`, `elastic`, `pls`, `krr`, `extra`, `catboost`, `lightgbm` —
  each with a preprocessing configuration and fixed hyperparameters
- `exploration_disclosure`: records that an earlier seed informed the candidate
  families, so this cohort is not an independent external test

`modelling/config.py` is the generator that produced the file; it refuses to
overwrite an existing `protocol.json`.

---

## 6. Prediction archives (`results/predictions/*.npz`)

All archives are uncompressed-key NumPy `.npz` files with `allow_pickle=False`
compatible content. Common keys:

| Key | Shape | Meaning |
|---|---|---|
| `y` | 250 × 50 | observed processed sensory scores |
| `outer` | 250 | outer fold of each beer (0–4) |
| `targets` | 50 | descriptor names, in target order |

`prep_pvr_oof.npz` additional keys:

| Key | Meaning |
|---|---|
| `PUBLISHED` | published out-of-fold predictions of the paper configuration |
| `W_paper` | rerun of the paper configuration |
| `R_control` | paper configuration without supervised feature selection |
| `R_fold` | routine fold-wise standardisation, no indicators, no selection |
| `R_global` | routine globally fitted standardisation, no indicators, no selection |

`prep_h2h_oof.npz` and `prep_h2h_addendum_oof.npz` use keys
`<arm>__<model>` for arms `A_plain`, `A_clip` (addendum only), `B_log`,
`B_increment` and models `ridge`, `pls`, `krr`, `elastic`, `extratrees`,
`randomforest`, `mlp_sklearn`, `mlp_torch`.

---

## 7. Derived result tables (`results/tables/`)

| File | Rows | Key columns |
|---|---|---|
| `compound_threshold_profile.csv` | 18 | `compound`, `unit`, thresholds, beers above threshold, percentages |
| `threshold_definition_check.csv` | 18 | mean vs median percentages, unit consistency, sensitivity flag |
| `threshold_sensory_link.csv` | 18 | compound–descriptor link used in the correspondence analysis |
| `sensory_monitorability.csv` | 50 | per-target R², modality, monitorability class |
| `monitorability_by_group.csv` | groups | monitorability aggregated by attribute group |
| `error_metrics_by_target.csv` | 50 | SD, range, near-floor fraction, RMSE/MAE/R²/nRMSE, rank ρ, FDR q |
| `error_metrics_by_group.csv` | groups | the same metrics by R² tier, modality and flavour source |
| `error_metrics_summary.json` | — | key contrasts (R² vs SD, high vs low tier tests) |
| `metric_equivalence_check.json` | — | identity between error reduction, nRMSE and R² |
| `prep_h2h_summary.csv` | 24 | arm × model macro RMSE/MAE/ρ for the pre-registered comparison |
| `prep_h2h_r2_summary.csv` | 32 | arm × model macro/median R² and positive-R² count |
| `prep_h2h_r2_by_target.csv`, `_paired.csv`, `_stratified.csv` | many | per-target R², paired Wilcoxon + BH-FDR, modality strata |
| `prep_pvr_summary.csv` | arms × scopes | paper configuration vs three routine arms |
| `prep_pvr_by_target.csv`, `_paired.csv`, `_consistency.csv`, `_stratified.csv` | many | per-target and per-fold contrasts |
| `q2_within_style_top3.csv`, `q2_full_pool_top3.csv` | 50 | style-stratified attribution top-three lists |
| `q2_null_baselines.csv/.json` | 30 | style-label permutation null |
| `q2_style_recheck_summary.json` | — | reproduction and null-model summary |
| `raw_panel_audit.json` | — | structure audit of the 2,197 raw records |
| `raw_panel_reconstruction_check.json` | — | independent attempt to reconstruct the official labels |
| `raw_panel_beer_map.csv` | 250 | inferred raw-name → official-beer mapping (inferred, not official) |
| `raw_panel_labels_A/B/C.csv` | 250 × 51 | three beer-level 0–6 aggregations (mean, assessor-equal mean, median) |

`results/reports/REPORT.md` and `results/reports/prep_pvr_report.md` are the
working reports written by the corresponding scripts.

---

## 8. Archived upstream inputs (`data/derived/`)

These files are not new analyses. They are intermediate outputs of the
upstream Q1/Q2 work that the included reanalysis scripts read. They are kept so
the reanalysis can run without re-fitting the upstream models.

| File | Origin | Role |
|---|---|---|
| `q2_reference/q1_oof_truth_continuous.csv` | Q1 final continuous labels | truth table for the error-metric reanalysis |
| `q2_eligible_only/crossfit_oof_ridge.csv` | Q2 eligible-only cross-fit | ridge OOF predictions for the error-metric reanalysis |
| `q2_eligible_only/crossfit_oof_extra.csv` | Q2 eligible-only cross-fit | extra-trees OOF predictions |
| `q2_eligible_only/crossfit_metrics.csv` | Q2 run metrics | comparison values for the archived metrics |
| `q2_eligible_only/top3_all_available.csv` | Q2 attribution run | top-three candidate lists for the style recheck |
| `q1_final/OOF_depth3_l2_30.csv` | Q1 final published OOF | published predictions, wide format |
| `q1_final/Q1_FINAL_PER_TARGET.csv` | Q1 final per-target table | target scope flags (`main37`, `af30`) and per-target metrics |
| `q3_field_dictionary.csv` | Q3 generated chemistry | unit dictionary for the threshold unit check |
| `SAMPLE_SIZE_REFERENCE_REPRODUCTION.csv` | Q2 stability module | learning-curve evidence behind Supplementary Table S5 |

`data/derived/SAMPLE_SIZE_REFERENCE_REPRODUCTION.csv` is the archived file named
in the supplementary material. `q3_field_dictionary.csv` is an archived unit
dictionary, not a table in the manuscript.

---

## 9. What each script reads and writes

| Script | Reads | Writes |
|---|---|---|
| `modelling/__main__.py` | `data/raw/{1,4}.csv`, `validation/splits/`, `modelling/protocol.json` | a new run directory (`runs/...`) |
| `analysis/prep_h2h.py` | data, splits, `analysis/preregistration/prep_h2h_preregistration.json` | `results/tables/prep_h2h_*`, `results/predictions/prep_h2h_oof.npz` |
| `analysis/prep_h2h_addendum.py` | data, splits, both h2h preregistrations, `prep_h2h_oof.npz` | `prep_h2h_addendum_*`, `prep_h2h_addendum_oof.npz` |
| `analysis/prep_h2h_r2_metrics.py` | `prep_h2h_addendum_oof.npz`, `sensory_monitorability.csv` | `prep_h2h_r2_*` |
| `analysis/prep_paper_vs_routine.py` | data, splits, `prep_paper_vs_routine_preregistration.json`, `data/derived/q1_final/*` | `prep_pvr_*`, `prep_pvr_oof.npz` |
| `analysis/raw_panel_audit.py` | `data/raw/Raw sensory data anonymized.csv`, `4.csv` | `raw_panel_*` |
| `analysis/style_recheck.py` | `1.csv`, `4.csv`, `derived/q2_eligible_only/top3_all_available.csv` | `q2_*` |
| `analysis/error_metrics_reanalysis.py` | `4.csv`, `6.csv`, `8.xlsx`, `derived/q2_*`, `sensory_monitorability.csv` | `error_metrics_*` |
| `analysis/threshold_definition_check.py` | `8.xlsx`, `compound_threshold_profile.csv`, `derived/q3_field_dictionary.csv` | `threshold_definition_check.csv` |
| `analysis/metric_equivalence_check.py` | `error_metrics_by_target.csv` | `metric_equivalence_check.json` |
