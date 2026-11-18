# Modelling pipeline

`modelling/` is the nested cross-validation pipeline: five outer folds and
three inner folds over 250 beers, 231 chemical features and 50 sensory targets.

| File | Role |
|---|---|
| `data.py` | aligns `1.csv` and `4.csv` on `beer_id`, validates shapes, loads the frozen splits |
| `config.py` | generates the 57-candidate grid (8 families, coarse/refine stages) |
| `protocol.json` | the grid as frozen before any outer-fold score was inspected |
| `models.py` | fold-internal preprocessing (`FoldTransform`) and the eight model families |
| `engine.py` | inner selection, outer evaluation, atomic caches, failure records |
| `increment.py` | `IncrementTransform`, the additional preprocessing arm used in the head-to-head study |
| `plots.py` | diagnostic figures written into the run directory |
| `feature_metadata.csv` | unit, transformation state and group of each of the 231 features |

## Design rules

- Every data-dependent transform (clip, centre, scale, feature selection,
  missing indicators) is fitted inside the training fold only.
- The inner search never receives the outer test labels.
- The protocol, data snapshot, source code and frozen splits are copied into
  the run directory and checked on resume.
- A failed candidate is recorded with its traceback; it is never replaced by a
  mean prediction and reported as a success.
- The pipeline refuses to overwrite an existing run directory.

## Commands

```bash
python -m modelling --stage runtime   --run-dir runs/runtime
python -m modelling --stage evaluate  --run-dir runs/evaluate
python -m modelling --stage fit-final --run-dir runs/evaluate
python -m modelling --stage predict   --run-dir runs/evaluate \
       --predict-input new.csv --prediction-dir predictions/new
```

`--data-dir` defaults to `data/raw`; `--split-dir` defaults to
`validation/splits`; `--protocol` defaults to `modelling/protocol.json`.
`--expected-python` is optional and off by default, because pinning an exact
patch version is not portable; the exact version used by a run is recorded in
its `run_definition.json`.

## What the pipeline is not

The out-of-fold results are an internal nested-CV evaluation on a cohort that
was used in earlier exploratory work. They are not an independent external
test. `protocol.json` records this explicitly in `exploration_disclosure`.
