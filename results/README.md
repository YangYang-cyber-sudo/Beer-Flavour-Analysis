# Results

Archived outputs of the analysis. **These files are the research record; do not
edit them by hand.** Their content is unchanged from the archived release; only
their directory location was normalised (see `../docs/RELEASE_AUDIT.md`).

| Directory | Content |
|---|---|
| `tables/` | derived CSV and JSON tables (thresholds, correspondence, monitorability, error decomposition, attribution stability, preprocessing comparisons, raw-panel labels) |
| `predictions/` | out-of-fold prediction archives (`.npz`) |
| `reports/` | Markdown working reports written by the analysis scripts |

Every table that carries a manuscript result is listed with its columns in
[`../docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md). The mapping from
manuscript section to script is in [`../analysis/README.md`](../analysis/README.md).

## Provenance

- `prep_h2h_*` and `prep_h2h_r2_*` come from the pre-registered preprocessing
  comparison and its frozen post-hoc addendum.
- `prep_pvr_*` compares the paper configuration with three routine
  preprocessing arms under the same frozen outer folds.
- `q2_*` and `raw_panel_*` come from the attribution and label-aggregation
  reanalyses.
- `error_metrics_*` decomposes prediction error into absolute and
  score-spread-normalised components.

## Reproducing the tables

`python tests/verify_archive.py` recomputes the stored summary values from the
stored predictions and checks that they agree. Re-running the analysis scripts
overwrites these files; work in a copy if you need to keep the archived bytes.
