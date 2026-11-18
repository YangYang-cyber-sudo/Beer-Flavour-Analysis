"""事后敏感性对照：公平基线 A_clip（常规 z-score + 折内 1/99% 截尾）。

明确标注为 post-hoc，不能当作预注册确证性检验。全部 8 个模型、同一超参、同一划分。
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support.paths import (PREREG, TABLES, PREDICTIONS,  # noqa: E402
                           ensure_output_dirs, configure_console)
from analysis.prep_h2h import (ROOT, DATA, SPLITS, build_model, oof_predict,  # noqa: E402
                               metrics, paired, load_data, load_splits, log)

configure_console()
ensure_output_dirs()
PRE = json.loads((PREREG / "prep_h2h_preregistration.json").read_text(encoding="utf-8"))
ADD = json.loads((PREREG / "prep_h2h_addendum_preregistration.json").read_text(encoding="utf-8"))


def write_csv(name, rows):
    with (TABLES / name).open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    started = datetime.now()
    log("ADDENDUM START", started.isoformat(timespec="seconds"))
    d = load_data(DATA)
    X, y = d["X"], d["Y"]
    targets = list(d["targets"])
    groups, logs = d["groups"], d["verified_log"]
    identity = d["metadata"]["transform_status"].eq(
        "mostly_identity_with_exceptions").to_numpy()
    outer, _ = load_splits(d["beers"], SPLITS)

    with np.load(PREDICTIONS / "prep_h2h_oof.npz") as z:
        oofs = {k: z[k].copy() for k in z.files if k not in ("y", "outer", "targets")}

    cfg = ADD["added_arm"]["A_clip"]["config"]
    models = PRE["models"]
    summary, per_target = [], []
    for m in models:
        key = f"A_clip__{m['id']}"
        log("RUN", key)
        oof = oof_predict(cfg, m["family"], m["params"], X, y, outer,
                          groups, logs, identity)
        oofs[key] = oof
        mt = metrics(oof, y)
        summary.append({"arm": "A_clip", "model": m["id"], "tier": m["tier"],
                        "macro_rmse": mt["macro_rmse"], "macro_mae": mt["macro_mae"],
                        "macro_rho": mt["macro_rho"]})
        for j, name in enumerate(targets):
            per_target.append({"arm": "A_clip", "model": m["id"], "target": name,
                               "rmse": float(mt["rmse"][j]), "mae": float(mt["mae"][j]),
                               "rho": float(mt["rho"][j])})
        log("    macro_rmse=%.6f macro_rho=%.4f" % (mt["macro_rmse"], mt["macro_rho"]))

    # 与 ladder L1 (Ridge) 交叉核对
    ladder = list(csv.DictReader((TABLES / "prep_h2h_ladder.csv").open(encoding="utf-8-sig")))
    l1 = next(r for r in ladder if r["step"] == "L1")
    ridge = next(r for r in summary if r["model"] == "ridge")
    assert abs(float(l1["macro_rmse"]) - ridge["macro_rmse"]) < 1e-9, "L1 与 A_clip 不一致"
    log("ASSERT A_clip(ridge) == ladder L1: PASS")

    contrasts = [("B_log", "A_clip", "JIA_vs_fair_baseline"),
                 ("B_increment", "A_clip", "YI_vs_fair_baseline"),
                 ("A_clip", "A_plain", "clip_effect_vs_no_protection")]
    paired_rows, consistency = [], []
    for b, a, label in contrasts:
        for m in models:
            oa, ob = oofs[f"{a}__{m['id']}"], oofs[f"{b}__{m['id']}"]
            ma, mb = metrics(oa, y), metrics(ob, y)
            rows = paired(y, oa, ob, targets)
            for r in rows:
                j = targets.index(r["target"])
                r.update({"contrast": label, "model": m["id"], "tier": m["tier"],
                          "rmse_a": float(ma["rmse"][j]), "rmse_b": float(mb["rmse"][j]),
                          "delta_rmse": float(mb["rmse"][j] - ma["rmse"][j])})
                paired_rows.append(r)
            consistency.append({
                "contrast": label, "model": m["id"], "tier": m["tier"],
                "macro_rmse_a": ma["macro_rmse"], "macro_rmse_b": mb["macro_rmse"],
                "delta_macro_rmse": mb["macro_rmse"] - ma["macro_rmse"],
                "macro_rho_a": ma["macro_rho"], "macro_rho_b": mb["macro_rho"],
                "delta_macro_rho": mb["macro_rho"] - ma["macro_rho"],
                "n_targets_improved": int((mb["rmse"] < ma["rmse"]).sum()),
                "n_targets_regressed": int((mb["rmse"] > ma["rmse"]).sum()),
                "n_sig_fdr05_B_better": sum(1 for r in rows if r["significant_fdr05"] and r["direction"] == "B_better"),
                "n_sig_fdr05_A_better": sum(1 for r in rows if r["significant_fdr05"] and r["direction"] == "A_better"),
                "overall_better": bool(mb["macro_rmse"] < ma["macro_rmse"])})

    write_csv("prep_h2h_addendum_summary.csv", summary)
    write_csv("prep_h2h_addendum_per_target.csv", per_target)
    write_csv("prep_h2h_addendum_paired.csv", paired_rows)
    write_csv("prep_h2h_addendum_consistency.csv", consistency)
    np.savez_compressed(PREDICTIONS / "prep_h2h_addendum_oof.npz", y=y, outer=outer,
                        targets=np.array(targets), **oofs)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "post_hoc_sensitivity_control",
        "declared_in": "prep_h2h_addendum_preregistration.json",
        "summary": summary, "consistency": consistency,
        "cross_model_count": {
            label: {"n_models": len(models),
                    "n_overall_better": sum(1 for c in consistency if c["contrast"] == label and c["overall_better"]),
                    "mean_delta_macro_rmse": float(np.mean([c["delta_macro_rmse"] for c in consistency if c["contrast"] == label])),
                    "mean_delta_macro_rho": float(np.mean([c["delta_macro_rho"] for c in consistency if c["contrast"] == label]))}
            for _, _, label in contrasts},
    }
    (TABLES / "prep_h2h_addendum_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log("DONE", datetime.now().isoformat(timespec="seconds"),
        "elapsed", str(datetime.now() - started))
    log("CROSS_MODEL", json.dumps(result["cross_model_count"], ensure_ascii=False))


if __name__ == "__main__":
    main()
