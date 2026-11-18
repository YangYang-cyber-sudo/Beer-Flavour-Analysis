"""核算：相对均值基线误差降低 与 R² 是否数学同源。

背景：2026-11-14 v3 大纲曾把「误差降低 35.25% vs 3.73%」当作绕开
R² 陷阱的独立证据。本脚本检验这一前提。

结论（预期）：error_reduction_pct = 1 - RMSE/SD，与 R² 的 Spearman
rho ≈ 1，两者是同一量（RMSE/SD）的单调变换，不构成独立验证。
真正独立于 SST 的信息只有：绝对 RMSE 与预测-实测秩相关 rho。

用法：python analysis/metric_equivalence_check.py
输入：results/tables/error_metrics_by_target.csv
输出：results/tables/metric_equivalence_check.json
"""

import csv
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.paths import TABLES, ensure_output_dirs, configure_console  # noqa: E402

configure_console()
ensure_output_dirs()
SRC = TABLES / "error_metrics_by_target.csv"
DST = TABLES / "metric_equivalence_check.json"

with SRC.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))


def col(name):
    return [float(r[name]) for r in rows]


r2 = col("published_r2_mean")
err_red = col("error_reduction_pct")
rmse = col("rmse_ensemble")
sd = col("truth_sd")
rho_pred = col("rho_ensemble")

# 逐点核对恒等式：误差降低% = 100 * (1 - RMSE/SD)
reconstructed = [100.0 * (1.0 - m / s) for m, s in zip(rmse, sd)]
max_gap = max(abs(a - b) for a, b in zip(err_red, reconstructed))


def sp(a, b):
    r, p = spearmanr(a, b)
    return {"rho": round(float(r), 5), "p": float(p)}


result = {
    "n_targets": len(rows),
    "identity": "error_reduction_pct = 100 * (1 - RMSE / label_SD)",
    "max_abs_gap_percentage_points": round(max_gap, 6),
    "spearman": {
        "r2_vs_error_reduction": sp(r2, err_red),
        "r2_vs_label_sd": sp(r2, sd),
        "error_reduction_vs_label_sd": sp(err_red, sd),
        "r2_vs_absolute_rmse": sp(r2, rmse),
        "r2_vs_pred_obs_rank_rho": sp(r2, rho_pred),
        "pred_obs_rank_rho_vs_label_sd": sp(rho_pred, sd),
    },
    "interpretation": (
        "Error reduction, nRMSE and R2 are monotonic transforms of the same "
        "quantity RMSE/SD; they cannot serve as independent validation of one "
        "another. Only absolute RMSE and prediction-observation rank correlation "
        "add information, and the latter still correlates with label SD."
    ),
    "reporting_rule": (
        "Do not place error_reduction_pct or R2 in the abstract, title or figure "
        "caption as a primary result. Report SD, absolute RMSE and rank rho "
        "together, and state the shared dependence on label SD in Limitations."
    ),
}

with DST.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))
print("\nsaved ->", DST)
