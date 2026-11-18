# Frozen validation splits

`outer.csv`, `inner.csv` and `definition.json` are the frozen splits used by
every model and every table in this repository.

```text
seed                 2601731
unit                 beer (never the label)
outer folds          5, exactly 50 beers each
inner folds          3, rebuilt inside each outer training set
scope -1             3 inner folds over all 250 beers, for the deployment fit
label access         none during split generation
```

- `outer.csv`: `beer,outer_fold`.
- `inner.csv`: `scope,beer,inner_fold`; `scope` 0–4 corresponds to the outer
  fold whose training beers are being split, `scope -1` is the full-data split.
- `definition.json`: the provenance record quoted by the pipeline.

`modelling/data.py::load_splits` verifies that the outer folds are complete and
balanced, that each inner table covers exactly the corresponding training
beers, and that no inner table contains an outer test beer. It raises instead
of repairing an invalid split.

Do not regenerate these files. `modelling.data.make_splits` refuses to write
into a directory that already contains `outer.csv` or `inner.csv`.
