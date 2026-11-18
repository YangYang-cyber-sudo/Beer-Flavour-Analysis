#!/usr/bin/env python
"""Clean-environment smoke test for the Beer-Flavour-Analysis repository.

Checks, without fitting any model:
  1. every package imports from a fresh interpreter;
  2. the five public input files load and have the documented shapes;
  3. the frozen outer/inner splits load and satisfy their invariants;
  4. the frozen protocol, feature metadata and split definition agree;
  5. every archived prediction archive and key result table is readable.

Usage
-----
    python tests/smoke_test.py [--data-dir PATH]

``--data-dir`` overrides ``BEER_DATA_DIR`` / ``data/raw``.  If the public
data are not available the data-dependent checks are reported as SKIP and the
exit code stays 0, so the import/archive checks can still run in CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

PASSED = []
FAILED = []
SKIPPED = []


def check(name, fn):
    try:
        detail = fn()
        PASSED.append((name, detail or ""))
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    except AssertionError as exc:
        FAILED.append((name, str(exc)))
        print(f"FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - report the actual failure
        FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")


def skip(name, reason):
    SKIPPED.append((name, reason))
    print(f"SKIP  {name}: {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    from support import paths

    data_dir = (args.data_dir or paths.DATA).expanduser().resolve()
    print(f"repository : {ROOT}")
    print(f"data dir   : {data_dir}")
    print(f"python     : {sys.version.split()[0]}")
    print()

    # ------------------------------------------------------------ imports
    def imports():
        import modelling
        import modelling.data
        import modelling.engine
        import modelling.increment
        import modelling.models
        import support.audit
        import support.common
        import support.paths
        import support.tabular
        return "modelling, support"

    check("import modelling and support packages", imports)

    # ------------------------------------------------------------ protocol
    def protocol():
        p = json.loads((ROOT / "modelling" / "protocol.json").read_text(encoding="utf-8"))
        assert p["seed"] == 2601731, p["seed"]
        assert p["outer_folds"] == 5 and p["inner_folds"] == 3
        families = {c["family"] for c in p["candidates"]}
        expected = {"mean", "ridge", "elastic", "pls", "krr", "extra", "catboost", "lightgbm"}
        assert families == expected, families
        return f"{len(p['candidates'])} candidates, {len(families)} families"

    check("frozen protocol is complete", protocol)

    def split_definition():
        d = json.loads((ROOT / "validation" / "splits" / "definition.json").read_text(encoding="utf-8"))
        assert d["seed"] == 2601731
        assert d["unit"] == "beer" and d["outer"] == 5 and d["inner"] == 3
        assert d["label_access_for_split_generation"] is False
        return "seed 2601731, beer-level, 5x3, labels not used"

    check("frozen split definition", split_definition)

    # ------------------------------------------------------------ data
    required = ("1.csv", "4.csv", "6.csv", "8.xlsx",
                "Raw sensory data anonymized.csv")
    missing = [n for n in required if not (data_dir / n).is_file()]
    if missing:
        for name in ("public data loads", "frozen splits load", "feature metadata aligns"):
            skip(name, f"missing in {data_dir}: {', '.join(missing)}")
    else:
        def data_loads():
            from modelling.data import load_data
            d = load_data(data_dir)
            assert d["X"].shape == (250, 231), d["X"].shape
            assert d["Y"].shape == (250, 50), d["Y"].shape
            assert len(d["beers"]) == 250 and len(set(d["beers"])) == 250
            assert np.isfinite(d["Y"]).all(), "non-finite sensory label"
            assert np.isfinite(d["X"]).sum() + np.isnan(d["X"]).sum() == d["X"].size
            assert np.isinf(d["X"]).sum() == 0
            return "250 x 231, 250 x 50"

        check("public data loads", data_loads)

        def splits_load():
            from modelling.data import load_data, load_splits
            d = load_data(data_dir)
            outer, inner = load_splits(d["beers"], ROOT / "validation" / "splits")
            counts = np.bincount(outer)
            assert sorted(counts.tolist()) == [50] * 5, counts
            for scope in list(range(5)) + [-1]:
                train, folds = inner[scope]
                expected = (np.flatnonzero(outer != scope) if scope != -1
                            else np.arange(250))
                assert np.array_equal(np.sort(train), expected), scope
                assert set(folds.tolist()) == {0, 1, 2}, scope
            return "5 outer folds x 50 beers, 3 inner folds per scope"

        check("frozen splits load", splits_load)

        def increment_transform():
            import json
            from modelling.data import load_data, load_splits
            from modelling.increment import IncrementTransform
            d = load_data(data_dir)
            outer, _ = load_splits(d["beers"], ROOT / "validation" / "splits")
            prereg = json.loads(
                (ROOT / "analysis" / "preregistration"
                 / "prep_h2h_preregistration.json").read_text(encoding="utf-8"))
            cfg = prereg["arms"]["A_plain"]["config"]
            identity = (d["metadata"]["transform_status"]
                        .eq("mostly_identity_with_exceptions").to_numpy())
            train = np.flatnonzero(outer != 0)
            test = np.flatnonzero(outer == 0)
            transform = IncrementTransform(
                dict(cfg), d["groups"], d["verified_log"], identity).fit(
                    d["X"][train], d["Y"][train])
            z = transform.transform(d["X"][test])
            assert z.shape[0] == 50 and z.shape[1] > 0 and np.isfinite(z).all()
            return f"fold 0 transform -> {z.shape[1]} columns"

        check("recovered IncrementTransform runs", increment_transform)

        def metadata_aligns():
            from modelling.data import load_data
            d = load_data(data_dir)
            meta = d["metadata"]
            assert len(meta) == 231, len(meta)
            assert list(meta["field"]) == list(d["features"])
            assert d["groups"].shape == (231,) and d["verified_log"].shape == (231,)
            return f"{int(d['verified_log'].sum())} log10-transformed features"

        check("feature metadata aligns", metadata_aligns)

    # ------------------------------------------------------------ archive
    def npz_files():
        expected = {
            "prep_h2h_oof.npz": (250, 50),
            "prep_h2h_addendum_oof.npz": (250, 50),
            "prep_pvr_oof.npz": (250, 50),
        }
        for name, shape in expected.items():
            path = ROOT / "results" / "predictions" / name
            assert path.is_file(), f"missing {name}"
            with np.load(path, allow_pickle=False) as z:
                assert z["y"].shape == shape, (name, z["y"].shape)
                assert z["outer"].shape == (250,)
                assert z["targets"].shape == (50,)
        return "3 archives, 250 x 50"

    check("archived prediction archives", npz_files)

    def result_tables():
        expected_rows = {
            "prep_h2h_summary.csv": 24,
            "prep_h2h_r2_summary.csv": 32,
            "prep_pvr_summary.csv": None,
            "error_metrics_by_target.csv": 50,
            "compound_threshold_profile.csv": 18,
            "raw_panel_labels_B.csv": 250,
        }
        import pandas as pd
        for name, rows in expected_rows.items():
            path = ROOT / "results" / "tables" / name
            assert path.is_file(), f"missing {name}"
            frame = pd.read_csv(path, encoding="utf-8-sig")
            if rows is not None:
                assert len(frame) == rows, (name, len(frame))
        return "6 key tables readable"

    check("archived result tables", result_tables)

    # ------------------------------------------------------------ summary
    print()
    print(f"passed={len(PASSED)} failed={len(FAILED)} skipped={len(SKIPPED)}")
    if FAILED:
        print("SMOKE_TEST_FAIL")
        return 1
    print("SMOKE_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
