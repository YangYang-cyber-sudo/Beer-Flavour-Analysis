"""风格分层归因复核 · 论文用增量分析

目的：
  1) 精确复现 Q1_T37_BASELINE_AND_ANALYSIS.md 报告的 0.162 / 17/50，并补出逐目标名单；
  2) 检验该结论是否被「候选池仅 23 个、且由 pooled rho 预选」这一算法限定所驱动；
  3) 给出 0.162 的零模型参照系；
  4) 检验 17 个完全不相交目标是否集中在低 R² 或香气类目标（可推翻「风格混杂」论断的替代解释）。

纯增量：只写 results/tables/，不修改任何原有文件。
方法主体与 analysis_2026a/next_stage_analysis.py::q2_style_recheck 保持一致，便于逐位核对。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.paths import DATA, DERIVED, TABLES, ensure_output_dirs, configure_console  # noqa: E402

configure_console()
OFF = DATA
Q2 = DERIVED / "q2_eligible_only"
OUT = TABLES
ensure_output_dirs()
META = ("beer", "beer_id", "tasting_category_fine")
SEED = 20261114
rng = np.random.default_rng(SEED)
random.seed(SEED)


def finite(v):
    """与 pd.read_csv 后 np.isfinite 的语义一致（NA -> NaN -> False）。"""
    return np.isfinite(v)


def abs_spearman(x, y):
    r = stats.spearmanr(x, y).statistic
    return abs(r) if np.isfinite(r) else 0.0


def within_style_abs_spearman(x, y, styles, ok, strict=False):
    """按风格组内去均值后取 |Spearman|。

    strict=False：复现原脚本——只对 m.sum()>2 的组覆盖去均值值，小组样本保留原值。
    strict=True ：所有组都去均值，且小组(m.sum()<=2)样本不进入相关计算。
    """
    xc = x.copy()
    yc = y.copy()
    mask = ok.copy()
    for s in np.unique(styles):
        m = ok & (styles == s)
        n = int(m.sum())
        if n > 2:
            xc[m] = x[m] - x[m].mean()
            yc[m] = y[m] - y[m].mean()
        elif strict:
            mask[m] = False
    if mask.sum() < 30:
        return np.nan
    return abs_spearman(xc[mask], yc[mask])


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def load():
    c1 = pd.read_csv(OFF / "1.csv")
    c4 = pd.read_csv(OFF / "4.csv")
    styles = c1["tasting_category_fine"].astype(str).to_numpy()
    feat = [c for c in c1.columns if c not in META]
    top3 = pd.read_csv(Q2 / "top3_all_available.csv")
    keep = [t for t in c4.columns if t not in META]
    X = {f: c1[f].to_numpy(float) for f in feat}
    Y = {t: c4[t].to_numpy(float) for t in keep}
    return c1, c4, styles, feat, top3, keep, X, Y


def original_pool(g, target, y, X, feat):
    """原脚本候选池 = 该目标 Top-3 + 其余高 |pooled rho| 化合物前 20。"""
    pool = list(g.sort_values("rank").field)
    extra = []
    for f in feat:
        if f in pool:
            continue
        x = X[f]
        ok = finite(x) & finite(y)
        if ok.sum() < 30:
            continue
        r = abs_spearman(x[ok], y[ok])
        if np.isfinite(r):
            extra.append((f, r))
    extra = [f for f, _ in sorted(extra, key=lambda z: -z[1])[:20]]
    return list(dict.fromkeys(pool + extra))


def target_class(t):
    if t.startswith("A_"):
        return "aroma"
    if t.startswith("F_"):
        return "flavour"
    return "other/taste/palate"


def crossfit_r2(g):
    """取该目标 Q2 固定模型的 crossfit R²（Top-3 三行应同值）。"""
    if "q2_best_fixed_crossfit_r2" not in g.columns:
        return np.nan
    v = pd.to_numeric(g["q2_best_fixed_crossfit_r2"], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else np.nan


def main():
    c1, c4, styles, feat, top3, keep, X, Y = load()
    n_styles = len(np.unique(styles))

    # ---------- 1. 精确复现 + 逐目标名单 ----------
    rows, pools = [], {}
    for t, g in top3.groupby("target"):
        if t not in keep:
            continue
        y = Y[t]
        cand = original_pool(g, t, y, X, feat)
        pools[t] = cand
        orig = list(g.sort_values("rank").field)[:3]
        scores = {}
        for f in cand:
            x = X[f]
            ok = finite(x) & finite(y)
            scores[f] = within_style_abs_spearman(x, y, styles, ok)
        within = [f for f, _ in sorted(scores.items(), key=lambda z: -z[1])[:3]]
        # 严格版（小组样本不参与）
        strict_scores = {}
        for f in cand:
            x = X[f]
            ok = finite(x) & finite(y)
            strict_scores[f] = within_style_abs_spearman(x, y, styles, ok, strict=True)
        within_strict = [f for f, _ in sorted(strict_scores.items(), key=lambda z: -z[1])[:3]]
        j = jaccard(orig, within)
        r2 = crossfit_r2(g)
        rows.append(dict(
            target=t,
            target_class=target_class(t),
            crossfit_r2=round(r2, 6) if np.isfinite(r2) else "",
            original_top3=";".join(orig),
            within_style_top3=";".join(within),
            jaccard=round(j, 4),
            overlap=len(set(orig) & set(within)),
            strict_top3=";".join(within_strict),
            strict_jaccard=round(jaccard(orig, within_strict), 4),
            pool_size=len(cand),
        ))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "q2_within_style_top3.csv", index=False, encoding="utf-8-sig")

    summ = dict(
        targets=len(D),
        mean_jaccard=float(D.jaccard.mean()),
        median_jaccard=float(D.jaccard.median()),
        identical=int((D.jaccard == 1).sum()),
        disjoint=int((D.jaccard == 0).sum()),
        partial=int(((D.jaccard > 0) & (D.jaccard < 1)).sum()),
        mean_jaccard_strict=float(D.strict_jaccard.mean()),
        disjoint_strict=int((D.strict_jaccard == 0).sum()),
        n_styles=n_styles,
    )

    # ---------- 2. 全 231 列候选池复算 ----------
    full_rows = []
    for t, g in top3.groupby("target"):
        if t not in keep:
            continue
        y = Y[t]
        orig = list(g.sort_values("rank").field)[:3]
        scores = {}
        for f in feat:
            x = X[f]
            ok = finite(x) & finite(y)
            if ok.sum() < 30:
                continue
            v = within_style_abs_spearman(x, y, styles, ok)
            if np.isfinite(v):
                scores[f] = v
        within = [f for f, _ in sorted(scores.items(), key=lambda z: -z[1])[:3]]
        full_rows.append(dict(
            target=t, original_top3=";".join(orig),
            full_pool_top3=";".join(within),
            jaccard_full_pool=round(jaccard(orig, within), 4),
            overlap=len(set(orig) & set(within)),
            n_scored=len(scores),
        ))
    F = pd.DataFrame(full_rows)
    F.to_csv(OUT / "q2_full_pool_top3.csv", index=False, encoding="utf-8-sig")
    full_sum = dict(mean_jaccard=float(F.jaccard_full_pool.mean()),
                    median_jaccard=float(F.jaccard_full_pool.median()),
                    disjoint=int((F.jaccard_full_pool == 0).sum()),
                    identical=int((F.jaccard_full_pool == 1).sum()))

    # ---------- 3. 零模型 ----------
    # Null A：从同一候选池随机抽 3 个（无信息排序）的 Jaccard 分布
    null_a = []
    for t, cand in pools.items():
        orig = D.loc[D.target == t, "original_top3"].iloc[0].split(";")
        for _ in range(200):
            pick = rng.choice(cand, size=min(3, len(cand)), replace=False)
            null_a.append(jaccard(orig, pick))
    # Null B：随机置换风格标签后重算 within-style Jaccard（30 次）
    null_b = []
    orig_map = {t: list(g.sort_values("rank").field)[:3]
                for t, g in top3.groupby("target") if t in keep}
    for rep in range(30):
        perm_styles = rng.permutation(styles)
        vals = []
        for t, cand in pools.items():
            y = Y[t]
            sc = {}
            for f in cand:
                x = X[f]
                ok = finite(x) & finite(y)
                sc[f] = within_style_abs_spearman(x, y, perm_styles, ok)
            w = [f for f, _ in sorted(sc.items(), key=lambda z: -z[1])[:3]]
            vals.append(jaccard(orig_map[t], w))
        null_b.append(float(np.mean(vals)))
    null_sum = dict(
        random_top3_mean=float(np.mean(null_a)),
        random_top3_median=float(np.median(null_a)),
        random_top3_p05=float(np.quantile(null_a, 0.05)),
        random_top3_disjoint_fraction=float(np.mean(np.array(null_a) == 0)),
        style_label_permutation_reps=len(null_b),
        style_perm_mean_jaccard_mean=float(np.mean(null_b)),
        style_perm_mean_jaccard_sd=float(np.std(null_b, ddof=1)),
        style_perm_min=float(np.min(null_b)),
        style_perm_max=float(np.max(null_b)),
        observed_vs_perm_z=float((summ["mean_jaccard"] - np.mean(null_b)) / np.std(null_b, ddof=1))
        if np.std(null_b, ddof=1) > 0 else None,
    )
    pd.DataFrame({
        "replicate": list(range(1, len(null_b) + 1)),
        "null": ["style_label_permutation"] * len(null_b),
        "mean_jaccard": null_b,
    }).to_csv(OUT / "q2_null_baselines.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "q2_null_baselines.json", "w", encoding="utf-8") as f:
        json.dump(null_sum, f, ensure_ascii=False, indent=2)

    # ---------- 4. 分层检验：17 个不相交目标是否集中在低 R² / 香气类 ----------
    dis = D[D.jaccard == 0]
    oth = D[D.jaccard > 0]
    r2_dis = pd.to_numeric(dis.crossfit_r2, errors="coerce").dropna()
    r2_oth = pd.to_numeric(oth.crossfit_r2, errors="coerce").dropna()
    if len(r2_dis) > 1 and len(r2_oth) > 1:
        u = stats.mannwhitneyu(r2_dis, r2_oth, alternative="less")
        strat = dict(
            n_disjoint=len(dis), n_non_disjoint=len(oth),
            r2_mean_disjoint=float(r2_dis.mean()), r2_mean_others=float(r2_oth.mean()),
            r2_median_disjoint=float(r2_dis.median()), r2_median_others=float(r2_oth.median()),
            mannwhitney_U=float(u.statistic), mannwhitney_p_one_sided_less=float(u.pvalue),
            weak_r2_below_0_1_disjoint=int((r2_dis < 0.1).sum()),
            weak_r2_below_0_1_others=int((r2_oth < 0.1).sum()),
            class_counts_disjoint=dis.target_class.value_counts().to_dict(),
            class_counts_others=oth.target_class.value_counts().to_dict(),
            disjoint_targets=sorted(dis.target.tolist()),
        )
    else:
        strat = dict(n_disjoint=len(dis), n_non_disjoint=len(oth),
                     disjoint_targets=sorted(dis.target.tolist()))

    result = dict(protocol=dict(seed=SEED, source="next_stage_analysis.py::q2_style_recheck",
                                pool_rule="Top-3 + top20 |pooled Spearman|",
                                full_pool_columns=len(feat), n_styles=n_styles),
                  reproduction=summ, full_pool=full_sum, null_models=null_sum, stratification=strat)
    with open(OUT / "q2_style_recheck_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
