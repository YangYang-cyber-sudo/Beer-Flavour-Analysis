"""Re-analysis of per-attribute monitorability using error-scale and rank metrics.

Motivation
----------
R2 = 1 - SSE/SST depends on the target's own variance (SST).  We previously
observed Spearman(R2, sensory-score SD) = 0.756 (n=50, p=2.3e-10), so a low
R2 may reflect a narrow attribute range rather than a genuine absence of a
chemical-sensory relation.  This script recomputes, for all 50 attributes:

  * truth SD / range / near-floor fraction (label-side dispersion)
  * OOF RMSE, MAE, R2, nRMSE = RMSE/SD (error scale, cross-attribute comparable)
  * prediction-vs-truth Spearman rho and Pearson r, with BH-FDR q values
  * mean-baseline RMSE (= SD) and relative error reduction
  * univariate chemical-sensory rho from the supplied 44x50 correlation matrix

All quantities are computed from per-sample out-of-fold predictions, not from
aggregated R2 values.  Nothing in the existing results is modified.

Inputs
  data/raw/4.csv, data/raw/6.csv, data/raw/8.xlsx
  data/derived/q2_reference/q1_oof_truth_continuous.csv
  data/derived/q2_eligible_only/crossfit_oof_ridge.csv
  data/derived/q2_eligible_only/crossfit_oof_extra.csv
  data/derived/q2_eligible_only/crossfit_metrics.csv
  results/tables/sensory_monitorability.csv

Outputs (results/tables/)
  error_metrics_by_target.csv, error_metrics_by_group.csv,
  error_metrics_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.paths import DATA, DERIVED, TABLES, ensure_output_dirs, configure_console  # noqa: E402

configure_console()
OUT = TABLES
ensure_output_dirs()

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_table(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def col_index(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    return n - 1


def xlsx_sheet(z: zipfile.ZipFile, shared: list[str], i: int):
    x = ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml"))
    out = []
    for row in x.findall(".//" + NS + "row"):
        d = {}
        for c in row:
            v = c.find(NS + "v")
            t = c.get("t")
            val = v.text if v is not None else ""
            if t == "s" and val != "":
                val = shared[int(val)]
            if t == "inlineStr":
                val = "".join(tt.text or "" for tt in c.iter(NS + "t"))
            d[col_index(c.get("r"))] = val
        out.append([d.get(k, "") for k in range(max(d) + 1)] if d else [])
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        val = pvals[i] * n / rank
        prev = min(prev, val)
        q[i] = min(1.0, prev)
    return q


# ---------------------------------------------------------------- load data
truth, truth_hdr = read_table(DERIVED / "q2_reference/q1_oof_truth_continuous.csv")
beers = [r["beer"] for r in truth]
targets = [h for h in truth_hdr if h != "beer"]
Y = {t: np.array([float(r[t]) for r in truth], dtype=float) for t in targets}

four, four_hdr = read_table(DATA / "4.csv")
four_by_beer = {r["beer"]: r for r in four}
max_label_diff = 0.0
for t in targets:
    v = np.array([float(four_by_beer[b][t]) for b in beers], dtype=float)
    max_label_diff = max(max_label_diff, float(np.max(np.abs(v - Y[t]))))


def load_oof(path: Path):
    rows, hdr = read_table(path)
    by_beer = {r["beer"]: r for r in rows}
    assert set(by_beer) == set(beers), f"beer ids mismatch in {path.name}"
    return {t: np.array([float(by_beer[b][t]) for b in beers], dtype=float)
            for t in targets}


PRED = {"ridge": load_oof(DERIVED / "q2_eligible_only/crossfit_oof_ridge.csv"),
        "extra": load_oof(DERIVED / "q2_eligible_only/crossfit_oof_extra.csv")}

met, _ = read_table(DERIVED / "q2_eligible_only/crossfit_metrics.csv")
MET = {(m["target"], m["model"]): m for m in met}

mon_rows, _ = read_table(OUT / "sensory_monitorability.csv")
MON = {r["target"]: r for r in mon_rows}

# Sensory names -> human readable labels / modality / flavour source
z = zipfile.ZipFile(DATA / "8.xlsx")
shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
sn = xlsx_sheet(z, shared, 2)
META = {x[1]: (x[2], x[3], x[4]) for x in sn[1:] if len(x) > 4}

# 44 x 50 univariate chemical-sensory Spearman matrix (attachment 6)
with open(DATA / "6.csv", encoding="utf-8-sig", newline="") as f:
    a6 = list(csv.reader(f))
a6_hdr = [c.strip().replace("\r\n", " ") for c in a6[0]]
a6_rows = {r[0]: r for r in a6[1:] if r and r[0]}
# best |rho| between any of the 44 compounds and each sensory attribute,
# mapped by the human-readable names in sensory_monitorability.csv
univar = {}
for t in targets:
    label = MON.get(t, {}).get("clean_name", t).strip()
    cand = [i for i, h in enumerate(a6_hdr) if i > 0 and h.lower() == label.lower()]
    if not cand:
        cand = [i for i, h in enumerate(a6_hdr) if i > 0 and label.lower() in h.lower()]
    if not cand:
        univar[t] = float("nan")
        continue
    j = cand[0]
    vals = []
    for row in a6_rows.values():
        try:
            vals.append(abs(float(row[j])))
        except (ValueError, IndexError):
            pass
    univar[t] = max(vals) if vals else float("nan")

# ---------------------------------------------------------------- per target
records = []
for t in targets:
    y = Y[t]
    sd = float(np.std(y, ddof=1))
    mn, mx = float(np.min(y)), float(np.max(y))
    span = mx - mn
    near_floor = float(np.mean(y <= mn + 0.05 * span))
    modality, source, clean = META.get(t, ("NA", "NA", t))

    rec = dict(target=t, clean_name=clean, modality=modality, flavour_source=source,
               truth_sd=round(sd, 5), truth_range=round(span, 5),
               near_floor_frac=round(near_floor, 5),
               max_abs_univariate_chem_rho=(round(univar[t], 4)
                                            if not math.isnan(univar[t]) else ""))
    for model in ("ridge", "extra"):
        p = PRED[model][t]
        rho, prho = spearmanr(p, y)
        r, pr = pearsonr(p, y)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        mae = float(np.mean(np.abs(p - y)))
        r2 = 1.0 - float(np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
        rec[f"rho_{model}"] = round(float(rho), 5)
        rec[f"p_rho_{model}"] = float(prho)
        rec[f"pearson_{model}"] = round(float(r), 5)
        rec[f"rmse_{model}"] = round(rmse, 5)
        rec[f"mae_{model}"] = round(mae, 5)
        rec[f"r2_{model}"] = round(r2, 5)
        m = MET.get((t, model))
        if m:
            rec[f"csv_r2_gap_{model}"] = round(abs(r2 - float(m["r2"])), 6)

    # ensemble-mean prediction, consistent with the published 2-model mean R2
    pa = (PRED["ridge"][t] + PRED["extra"][t]) / 2
    rho_a, p_a = spearmanr(pa, y)
    rmse_a = float(np.sqrt(np.mean((pa - y) ** 2)))
    r2_a = 1.0 - float(np.sum((pa - y) ** 2) / np.sum((y - y.mean()) ** 2))
    rec.update(rho_ensemble=round(float(rho_a), 5), p_rho_ensemble=float(p_a),
               rmse_ensemble=round(rmse_a, 5), r2_ensemble=round(r2_a, 5),
               nrmse_sd=round(rmse_a / sd, 5),
               baseline_rmse=round(sd, 5),
               error_reduction_pct=round(100 * (1 - rmse_a / sd), 2))

    pub = float(MON[t]["r2_mean"]) if t in MON else float("nan")
    rec["published_r2_mean"] = round(pub, 5)
    rec["tier_by_r2"] = ("high" if pub >= 0.4 else "moderate" if pub >= 0.1
                         else "unsupported")
    records.append(rec)

qvals = bh_fdr([r["p_rho_ensemble"] for r in records])
for r, q in zip(records, qvals):
    r["q_rho_ensemble"] = float(q)
    r["rank_signal_significant"] = int(q < 0.05 and r["rho_ensemble"] > 0)

# ---------------------------------------------------------------- group stats
rows = []
for grouping in ("tier_by_r2", "modality", "flavour_source"):
    groups = defaultdict(list)
    for r in records:
        groups[r[grouping]].append(r)
    for g, items in groups.items():
        rows.append(dict(grouping=grouping, group=g, n=len(items),
                         mean_r2=round(statistics.mean(x["published_r2_mean"] for x in items), 4),
                         mean_truth_sd=round(statistics.mean(x["truth_sd"] for x in items), 4),
                         mean_rmse=round(statistics.mean(x["rmse_ensemble"] for x in items), 4),
                         mean_nrmse_sd=round(statistics.mean(x["nrmse_sd"] for x in items), 4),
                         mean_rho=round(statistics.mean(x["rho_ensemble"] for x in items), 4),
                         median_rho=round(statistics.median(x["rho_ensemble"] for x in items), 4),
                         n_significant_rank=sum(x["rank_signal_significant"] for x in items),
                         mean_error_reduction_pct=round(statistics.mean(x["error_reduction_pct"] for x in items), 2),
                         mean_near_floor=round(statistics.mean(x["near_floor_frac"] for x in items), 4)))

# ---------------------------------------------------------------- key tests
rho_r2_sd, p_r2_sd = spearmanr([r["published_r2_mean"] for r in records],
                               [r["truth_sd"] for r in records])
rho_r2_rho, p_r2_rho = spearmanr([r["published_r2_mean"] for r in records],
                                 [r["rho_ensemble"] for r in records])
rho_sd_rho, p_sd_rho = spearmanr([r["truth_sd"] for r in records],
                                 [r["rho_ensemble"] for r in records])
rho_r2_rmse, p_r2_rmse = spearmanr([r["published_r2_mean"] for r in records],
                                   [r["rmse_ensemble"] for r in records])

hi = [r for r in records if r["tier_by_r2"] == "high"]
lo = [r for r in records if r["tier_by_r2"] == "unsupported"]
mw_rho = mannwhitneyu([x["rho_ensemble"] for x in hi], [x["rho_ensemble"] for x in lo],
                      alternative="greater")
mw_rmse = mannwhitneyu([x["rmse_ensemble"] for x in hi], [x["rmse_ensemble"] for x in lo],
                       alternative="two-sided")
mw_sd = mannwhitneyu([x["truth_sd"] for x in hi], [x["truth_sd"] for x in lo],
                     alternative="greater")

summary = dict(
    n_targets=len(records), n_samples=len(beers),
    max_abs_label_diff_4csv_vs_truth=max_label_diff,
    max_abs_r2_gap_vs_published_csv=max(
        max(r.get(f"csv_r2_gap_{m}", 0) or 0 for m in ("ridge", "extra")) for r in records),
    spearman_r2_vs_sd=dict(rho=round(float(rho_r2_sd), 5), p=float(p_r2_sd)),
    spearman_r2_vs_pred_rho=dict(rho=round(float(rho_r2_rho), 5), p=float(p_r2_rho)),
    spearman_sd_vs_pred_rho=dict(rho=round(float(rho_sd_rho), 5), p=float(p_sd_rho)),
    spearman_r2_vs_rmse=dict(rho=round(float(rho_r2_rmse), 5), p=float(p_r2_rmse)),
    n_significant_rank_all=sum(r["rank_signal_significant"] for r in records),
    n_significant_rank_unsupported=sum(r["rank_signal_significant"] for r in lo),
    unsupported_targets=len(lo),
    unsupported_significant=[r["target"] for r in lo if r["rank_signal_significant"]],
    unsupported_not_significant=[r["target"] for r in lo if not r["rank_signal_significant"]],
    mannwhitney=dict(rho_high_gt_low=dict(U=float(mw_rho.statistic), p=float(mw_rho.pvalue)),
                     rmse_high_vs_low=dict(U=float(mw_rmse.statistic), p=float(mw_rmse.pvalue)),
                     sd_high_gt_low=dict(U=float(mw_sd.statistic), p=float(mw_sd.pvalue))),
    group_high={k: v for k, v in next(x for x in rows if x["grouping"] == "tier_by_r2"
                                      and x["group"] == "high").items()},
    group_unsupported={k: v for k, v in next(x for x in rows if x["grouping"] == "tier_by_r2"
                                              and x["group"] == "unsupported").items()},
)

# ---------------------------------------------------------------- save
def write_csv(path, data):
    fields = list(data[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


write_csv(OUT / "error_metrics_by_target.csv", records)
write_csv(OUT / "error_metrics_by_group.csv", rows)
with open(OUT / "error_metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n=== group summary ===")
for r in rows:
    print(f"  {r['grouping']:15s} {r['group']:12s} n={r['n']:2d} "
          f"R2={r['mean_r2']:.4f} SD={r['mean_truth_sd']:.4f} "
          f"RMSE={r['mean_rmse']:.4f} nRMSE={r['mean_nrmse_sd']:.4f} "
          f"rho={r['mean_rho']:.4f} sig={r['n_significant_rank']:2d}")
print("\n=== previously called 'uninterpretable' targets ===")
for r in sorted(lo, key=lambda x: -x["rho_ensemble"]):
    print(f"  {r['target']:20s} R2={r['published_r2_mean']:.4f} "
          f"SD={r['truth_sd']:.4f} RMSE={r['rmse_ensemble']:.4f} "
          f"rho={r['rho_ensemble']:+.4f} q={r['q_rho_ensemble']:.4g} "
          f"sig={r['rank_signal_significant']} nearfloor={r['near_floor_frac']:.2f}")
