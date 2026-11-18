#!/usr/bin/env python
"""Numeric consistency checks for the archived outputs shipped in results/.

The script never refits a model.  It recomputes the reported summaries from
the stored out-of-fold arrays and compares them with the archived tables, so a
reader can see that the tables and the predictions are one consistent set.

Checks
------
1. ``prep_h2h_summary.csv`` equals metrics recomputed from ``prep_h2h_oof.npz``.
2. ``prep_h2h_r2_summary.csv`` equals metrics recomputed from
   ``prep_h2h_addendum_oof.npz``.
3. ``prep_pvr_oof.npz['PUBLISHED']`` equals the archived published table
   ``data/derived/q1_final/OOF_depth3_l2_30.csv``.
4. The four preprocessing arms reproduce the macro metrics of Table 6.
5. ``error_metrics_by_target.csv`` equals metrics recomputed from the archived
   Q2 out-of-fold predictions.
6. Frozen splits, label tables and prediction shapes satisfy their invariants.

A NOTE line reports the known manuscript-vs-archive difference for the
per-attribute Table 5 values; it is not treated as a failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

TABLES = ROOT / "results" / "tables"
PREDICTIONS = ROOT / "results" / "predictions"
DERIVED = ROOT / "data" / "derived"
SPLITS = ROOT / "validation" / "splits"

PASSED, FAILED = [], []


def check(name, fn):
    try:
        detail = fn()
        PASSED.append(name)
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    except AssertionError as exc:
        FAILED.append((name, str(exc)))
        print(f"FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")


def macro_metrics(pred, truth):
    error = pred - truth
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    rho = np.array([stats.spearmanr(pred[:, j], truth[:, j]).statistic
                    for j in range(truth.shape[1])])
    return rmse, mae, rho


def main() -> int:
    # ---------------------------------------------------------------- h2h
    def h2h_summary():
        z = np.load(PREDICTIONS / "prep_h2h_oof.npz", allow_pickle=False)
        y = z["y"]
        stored = pd.read_csv(TABLES / "prep_h2h_summary.csv", encoding="utf-8-sig")
        assert len(stored) == 24, len(stored)
        checked = 0
        for row in stored.itertuples(index=False):
            key = f"{row.arm}__{row.model}"
            assert key in z.files, key
            rmse, mae, rho = macro_metrics(z[key], y)
            np.testing.assert_allclose(rmse.mean(), row.macro_rmse, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(mae.mean(), row.macro_mae, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(np.nanmean(rho), row.macro_rho, rtol=1e-9, atol=1e-12)
            checked += 1
        return f"{checked} arm x model rows recomputed"

    check("prep_h2h summary matches archived OOF", h2h_summary)

    def h2h_r2_summary():
        z = np.load(PREDICTIONS / "prep_h2h_addendum_oof.npz", allow_pickle=False)
        y = z["y"]
        var = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        stored = pd.read_csv(TABLES / "prep_h2h_r2_summary.csv", encoding="utf-8-sig")
        assert len(stored) == 32, len(stored)
        checked = 0
        for row in stored.itertuples(index=False):
            key = f"{row.arm}__{row.model}"
            o = z[key]
            se = (o - y) ** 2
            r2 = 1 - se.sum(axis=0) / var
            rmse = np.sqrt(se.mean(axis=0))
            rho = np.array([stats.spearmanr(o[:, j], y[:, j]).statistic
                            for j in range(y.shape[1])])
            np.testing.assert_allclose(r2.mean(), row.macro_r2, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(np.median(r2), row.median_r2, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(rmse.mean(), row.macro_rmse, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(np.nanmean(rho), row.macro_rho, rtol=1e-9, atol=1e-12)
            assert int((r2 > 0).sum()) == int(row.n_r2_positive)
            checked += 1
        return f"{checked} arm x model rows recomputed"

    check("prep_h2h R2 summary matches archived OOF", h2h_r2_summary)

    # ---------------------------------------------------------------- pvr
    def published_identity():
        z = np.load(PREDICTIONS / "prep_pvr_oof.npz", allow_pickle=False)
        targets = [str(t) for t in z["targets"]]
        table = pd.read_csv(DERIVED / "q1_final" / "OOF_depth3_l2_30.csv", index_col="beer")
        table.index = table.index.astype(str)
        order = sorted(table.index, key=int)
        array = table.loc[order, targets].to_numpy()
        gap = float(np.abs(array - z["PUBLISHED"]).max())
        assert gap < 1e-12, gap
        assert np.array_equal(table.loc[order, "outer_fold"].to_numpy(), z["outer"])
        return f"max |difference| = {gap:.1e}"

    check("published OOF table equals PUBLISHED array", published_identity)

    def table6_macro():
        z = np.load(PREDICTIONS / "prep_pvr_oof.npz", allow_pickle=False)
        y = z["y"]
        var = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        expected = {
            "W_paper": 0.2608,
            "R_control": 0.2541,
            "R_fold": 0.2553,
            "R_global": 0.2551,
        }
        got = {}
        for arm, published in expected.items():
            r2 = 1 - ((z[arm] - y) ** 2).sum(axis=0) / var
            got[arm] = float(r2.mean())
            np.testing.assert_allclose(got[arm], published, atol=5e-5,
                                       err_msg=f"{arm}: {got[arm]} vs {published}")
        return ", ".join(f"{k}={v:.4f}" for k, v in got.items())

    check("Table 6 macro R2 reproduced", table6_macro)

    # ---------------------------------------------------------------- error table
    def error_metrics_table():
        truth = pd.read_csv(DERIVED / "q2_reference" / "q1_oof_truth_continuous.csv",
                            encoding="utf-8-sig")
        targets = [c for c in truth.columns if c != "beer"]
        beers = truth["beer"].astype(str).tolist()
        ridge = pd.read_csv(DERIVED / "q2_eligible_only" / "crossfit_oof_ridge.csv",
                            encoding="utf-8-sig")
        extra = pd.read_csv(DERIVED / "q2_eligible_only" / "crossfit_oof_extra.csv",
                            encoding="utf-8-sig")
        for frame in (ridge, extra):
            frame["beer"] = frame["beer"].astype(str)
            frame.set_index("beer", inplace=True)
        truth = truth.set_index(truth["beer"].astype(str))
        stored = pd.read_csv(TABLES / "error_metrics_by_target.csv", encoding="utf-8-sig")
        stored = stored.set_index("target")
        assert len(stored) == 50, len(stored)
        max_gap = 0.0
        for target in targets:
            y = truth.loc[beers, target].to_numpy(float)
            pred = (ridge.loc[beers, target].to_numpy(float)
                    + extra.loc[beers, target].to_numpy(float)) / 2
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            rho = float(stats.spearmanr(pred, y).statistic)
            r2 = 1 - float(np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))
            row = stored.loc[target]
            # The archived table rounds to five decimals.
            np.testing.assert_allclose(rmse, row.rmse_ensemble, atol=1.1e-5)
            np.testing.assert_allclose(rho, row.rho_ensemble, atol=1.1e-5)
            np.testing.assert_allclose(r2, row.r2_ensemble, atol=1.1e-5)
            max_gap = max(max_gap, abs(rmse - row.rmse_ensemble))
        return f"50 targets, max |RMSE gap| = {max_gap:.1e}"

    check("error metrics table matches archived OOF", error_metrics_table)

    # ---------------------------------------------------------------- splits
    def split_invariants():
        outer = pd.read_csv(SPLITS / "outer.csv", dtype={"beer": str})
        inner = pd.read_csv(SPLITS / "inner.csv", dtype={"beer": str})
        assert len(outer) == 250 and outer["beer"].is_unique
        counts = outer["outer_fold"].value_counts().sort_index().to_dict()
        assert counts == {0: 50, 1: 50, 2: 50, 3: 50, 4: 50}, counts
        assert set(inner["scope"].unique()) == {0, 1, 2, 3, 4, -1}
        for scope, frame in inner.groupby("scope"):
            expected = (outer.loc[outer["outer_fold"] != scope, "beer"]
                        if scope != -1 else outer["beer"])
            assert set(frame["beer"]) == set(expected), scope
            assert set(frame["inner_fold"]) == {0, 1, 2}
        return "250 beers, 5 balanced outer folds, 6 inner scopes"

    check("frozen split invariants", split_invariants)

    # ---------------------------------------------------------------- labels
    def raw_panel_labels():
        frame = pd.read_csv(TABLES / "raw_panel_labels_B.csv", encoding="utf-8-sig")
        assert frame.shape == (250, 51), frame.shape
        assert frame.isna().sum().sum() == 0
        audit = json.loads((TABLES / "raw_panel_audit.json").read_text(encoding="utf-8"))
        assert audit["n_records"] == 2197 and audit["n_beers"] == 250
        assert audit["n_tasters"] == 16
        return "250 beers x 50 descriptors, 2,197 records"

    check("raw panel label reconstruction", raw_panel_labels)

    # ---------------------------------------------------------------- note
    note = json.loads((TABLES / "prep_pvr_meta.json").read_text(encoding="utf-8"))
    print()
    print("NOTE  Archived PUBLISHED OOF reproduces the Table 6 macro R2 "
          f"({note['published_macro_r2']:.6f}) but not every per-attribute value "
          "printed in Table 5 of the manuscript.")
    print("NOTE  See docs/MISSING_ASSETS.md; this release does not substitute a "
          "regenerated Table 5.")

    print()
    print(f"passed={len(PASSED)} failed={len(FAILED)}")
    if FAILED:
        print("ARCHIVE_VERIFICATION_FAIL")
        return 1
    print("ARCHIVE_VERIFICATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
