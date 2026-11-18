# Reanalysis scripts

Each script is a self-contained command-line program. Run it from the
repository root. Paths come from `support/paths.py`:

- `BEER_DATA_DIR` (default `data/raw`) — the public Zenodo files;
- `validation/splits/` — frozen folds;
- `results/` — outputs (`BEER_RESULTS_DIR` overrides).

| Script | Manuscript / supplementary item | Notes |
|---|---|---|
| `threshold_definition_check.py` | §3.1, Table 1; mean-vs-median sensitivity | reads `8.xlsx` |
| `error_metrics_reanalysis.py` | §3.6; error-scale decomposition and nRMSE | reads Q2 OOF + `4.csv`, `6.csv`, `8.xlsx` |
| `metric_equivalence_check.py` | metric equivalence argument (R² vs error reduction) | reads the error table |
| `style_recheck.py` | §3.7; style-stratified attribution and permutation null | reads `1.csv`, `4.csv`, Q2 top-three table |
| `raw_panel_audit.py` | §3.8, §2.3; raw 0–6 records and A/B/C aggregation | reads the raw panel CSV and `4.csv` |
| `prep_h2h.py` | Table S9 / preprocessing head-to-head | heavy: 3 arms × 8 models × 5 folds |
| `prep_h2h_addendum.py` | post-hoc fair baseline `A_clip` | heavy; declares its post-hoc status |
| `prep_h2h_r2_metrics.py` | R² recomputation from the frozen h2h OOF | fast; no fitting |
| `prep_paper_vs_routine.py` | Table 6 / Table S9; paper config vs three routine arms | heavy: CatBoost |

`preregistration/` holds the protocols that were fixed before the corresponding
runs. Do not edit them to match a rerun.

The scripts write into `results/tables/`, `results/predictions/` and
`results/reports/`. The released repository ships the archived versions of
those outputs; rerunning overwrites them, so work in a copy if you need to keep
both.

Uncertainties and provenance:

- `prep_paper_vs_routine.py` reads the frozen early-stopping iteration counts
  from `results/tables/prep_pvr_meta.json` because the original development run
  directories are not part of the archive. The fallback values are the same
  numbers and are labelled `documented_fallback`.
- `raw_panel_audit.py` deliberately reports that the competition manuscript's
  2.84e-8 label-reconstruction claim was **not** independently reproduced; do
  not cite it as reproduced.
- `style_recheck.py` writes an attribution **sensitivity** result, not a causal
  claim.
