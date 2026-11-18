# Dataset

## Source

All measurements come from the public data deposit accompanying:

> Schreurs, M., Piampongsant, S., Roncoroni, M., Cool, L., Herrera-Malaver, B.,
> Vanderaa, C., Theßeling, F. A., Kreft, Ł., Botzki, A., Malcorps, P.,
> Daenen, L., Wenseleers, T., Verstrepen, K. J. (2024).
> *Predicting and improving complex beer flavor through machine learning.*
> **Nature Communications** 15, 2368.
> https://doi.org/10.1038/s41467-024-46346-0

Data deposit (CC BY 4.0):

> https://doi.org/10.5281/zenodo.10653704

The deposit contains 250 commercial beers from 22 fine-grained styles, measured
by 16 trained assessors and by routine chemical analysis. No beer was purchased,
stored or measured for the present study; no new sensory session was performed.

## Required files

Download the Zenodo record and place these five files in `data/raw/`
(create the directory). `BEER_DATA_DIR` can point to any other directory.

| Canonical name | Zenodo table | Role | Shape / content | SHA-256 |
|---|---|---|---|---|
| `1.csv` | chemical measurements | **X**: 231 chemical properties | 250 rows × (231 + metadata) | `419849227a02e0c792cd405e6865035b7f58af1a92efa362dbb68a7ca2882f69` |
| `4.csv` | sensory results | **Y**: 50 trained-panel descriptors | 250 rows × (50 + metadata) | `43a7b38682ed81ce64d97ebc1874c09dffac023ade4877f34e3c83ef7d42dc1c` |
| `6.csv` | chemical × sensory correlations | univariate compound–descriptor matrix | 44 compounds × 50 descriptors | `a55004fb0390299a5c8089f0c37d3bfc006e7049963c84d7d9652affabe614ae` |
| `8.xlsx` | thresholds and calibrated chemistry | 18 threshold compounds; calibrated concentrations; descriptor names | workbook with 7 sheets | `f2316766792111d98281bd35c8403682be8c8069d166a7a7d5887ffbb227700a` |
| `Raw sensory data anonymized.csv` | raw anonymised panel records | 2,197 session records × 50 integer ratings (cp1252) | 2,197 rows × 56 columns | `78525c8cbbbd498578c451b06641969a20ada87d0fe4c67fe1e634ad34569628` |

`8.xlsx` sheets used by the code:

| Sheet | Used by | Purpose |
|---|---|---|
| `Sensory thresholds` | `analysis/threshold_definition_check.py` | compiled mean and median literature thresholds |
| `Sensory names` | `analysis/error_metrics_reanalysis.py` | descriptor modality, flavour source and display name |
| `Calibrated chemical dataset` | `analysis/threshold_definition_check.py` | calibrated concentration per compound |

Optional files from the same deposit (not read by the included scripts):
`2.csv` (chemical × chemical Spearman matrix), `5.csv` (sensory × sensory
Spearman matrix), `3.xlsx` (panel form), `7.csv` (free-text comments).

## Alignment and integrity rules

`modelling/data.py` aligns X and Y on `beer_id` and enforces all of the
following; it refuses to proceed instead of dropping or imputing rows:

1. the X and Y beer sets are identical;
2. `tasting_category_fine` agrees between the two tables for every beer;
3. X is exactly 250 × 231 and Y exactly 250 × 50 after removing the metadata
   columns `beer_id` and `tasting_category_fine`;
4. all 30 aroma/flavour target keys are present in Y;
5. Y is finite; X contains no infinite values.

Missing chemical values are kept as missing. The value `-1.81781854` is a
documented sentinel in the supplied table and is retained as an explicit
missing-indicator feature by the paper configuration; it is not silently
converted to zero.

## Licence

The dataset is distributed under **CC BY 4.0**. That licence covers the data.
It does not license the code in this repository; see `LICENSE`.

## What is not distributed here

- No measurement table, raw sensory record, or threshold workbook is stored in
  this repository.
- No participant-identifiable information is present in the raw panel file;
  it is the anonymised release of the original authors.
