# -*- coding: utf-8 -*-
"""匿名逐次品评记录审计 + 三种啤酒级 0-6 聚合标签构造 + 官方标签重建核验.

数据源: data/raw/Raw sensory data anonymized.csv (cp1252 编码；Zenodo 公开数据)
官方标签: data/raw/4.csv

只做只读审计与标签构造, 不训练模型, 不改写任何原始数据.
输出写入 results/tables/, 文件前缀 raw_panel_*.

重要: 竞赛论文 A2601731.pdf §3.2 自述「重建官方 50 维标签最大绝对误差 2.84e-8」.
本脚本独立尝试了多种标准化口径, 均未能达到该量级; 结论见
raw_panel_reconstruction_check.json, 不得在论文中把 2.84e-8 当作已独立复现.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.paths import DATA, TABLES, ensure_output_dirs, configure_console  # noqa: E402

configure_console()
RAW_PATH = DATA / "Raw sensory data anonymized.csv"
OFFICIAL = DATA / "4.csv"
OUT = TABLES
ensure_output_dirs()

# 原始表与 4.csv 的两个拼写差异
COLUMN_ALIAS = {
    "A_ester_fruity": "A_esters_fruity",
    "F_ester_fruity": "F_esters_fruity",
}
META_COLS = {"beer", "beer_id", "tasting_category_fine"}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_raw():
    # 0x85 在 cp1252 中是省略号, 出现在 comments 自由文本里; utf-8 / gbk 均无法解码
    text = RAW_PATH.read_bytes().decode("cp1252")
    rows = list(csv.reader(io.StringIO(text)))
    header, data = rows[0], [r for r in rows[1:] if r and any(x.strip() for x in r)]
    sensory = header[5:55]          # 50 个 0-6 感官维度
    comments_col = header.index("comments")
    return header, data, sensory, comments_col


def main():
    header, data, sensory, comments_col = load_raw()
    r4 = list(csv.reader(OFFICIAL.open(encoding="utf-8-sig")))
    h4 = r4[0]
    official_cols = [c for c in h4 if c not in META_COLS]
    alias_back = {v: k for k, v in COLUMN_ALIAS.items()}
    raw_index = {c: i for i, c in enumerate(sensory)}

    # ---------- 1. 结构审计 ----------
    values = []
    bad = []
    for x in data:
        for j, c in enumerate(sensory):
            v = x[5 + j].strip()
            try:
                values.append(int(v))
            except ValueError:
                bad.append((x[1], c, v))
    beers = sorted({x[1] for x in data}, key=lambda z: int(z) if z.isdigit() else 10**9)
    tasters = sorted({x[3] for x in data}, key=lambda z: int(z))
    dates = sorted({x[0] for x in data})
    styles = sorted({x[2] for x in data})
    rec_per_beer = Counter(x[1] for x in data)
    dup_pairs = Counter((x[1], x[3]) for x in data)

    audit = {
        "source": RAW_PATH.name,
        "encoding": "cp1252",
        "n_records": len(data),
        "n_columns": len(header),
        "n_sensory_descriptors": len(sensory),
        "n_beers": len(beers),
        "n_tasters": len(tasters),
        "n_tasting_dates": len(dates),
        "date_range": [dates[0], dates[-1]],
        "n_style_labels_raw": len(styles),
        "n_scored_cells": len(values),
        "n_non_integer_cells": len(bad),
        "n_empty_cells": sum(1 for x in data for j in range(50) if not x[5 + j].strip()),
        "value_range": [min(values), max(values)],
        "all_integer_0_to_6": all(isinstance(v, int) and 0 <= v <= 6 for v in values),
        "value_counts": {str(k): v for k, v in sorted(Counter(values).items())},
        "n_free_text_comments": sum(
            1 for x in data if x[comments_col].strip()
            and not _is_number(x[comments_col].strip())),
        "records_per_beer": dict(sorted(Counter(rec_per_beer.values()).items())),
        "n_beer_taster_pairs_with_multiple_records": sum(1 for v in dup_pairs.values() if v > 1),
        "official_beer_col_note": "4.csv 的 beer 为官方编号, 与原始表 beer_name 非同一编号体系; 不可直接 join",
        "official_style_n": len({x[h4.index("tasting_category_fine")] for x in r4[1:]}),
    }

    # ---------- 2. 三种啤酒级 0-6 聚合标签 ----------
    # A = 记录均值; B = 评委等权均值(先按评委聚合再对评委等权); C = 记录中位数
    grouped = defaultdict(list)
    for x in data:
        grouped[x[1]].append((x[3], np.array([float(v) for v in x[5:55]])))
    by_taster = defaultdict(list)
    for b, recs in grouped.items():
        for t, v in recs:
            by_taster[(b, t)].append(v)

    LA = np.zeros((len(beers), 50))
    LB = np.zeros((len(beers), 50))
    LC = np.zeros((len(beers), 50))
    for i, b in enumerate(beers):
        recs = grouped[b]
        arr = np.stack([v for _, v in recs])
        LA[i] = arr.mean(0)
        LC[i] = np.median(arr, axis=0)
        tmeans = np.stack([np.stack(by_taster[(b, t)]).mean(0) for t in {t for t, _ in recs}])
        LB[i] = tmeans.mean(0)

    for tag, mat in [("A", LA), ("B", LB), ("C", LC)]:
        with (OUT / f"raw_panel_labels_{tag}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["beer_name"] + [COLUMN_ALIAS.get(c, c) for c in sensory])
            for i, b in enumerate(beers):
                w.writerow([b] + [f"{v:.10g}" for v in mat[i]])

    audit["ABC_label_stats"] = {
        tag: {"min": float(m.min()), "max": float(m.max()), "mean": float(m.mean()),
              "sd": float(m.std()), "macro_sd_per_descriptor": float(m.std(0).mean())}
        for tag, m in [("A", LA), ("B", LB), ("C", LC)]
    }
    audit["AB_max_abs_diff"] = float(np.abs(LA - LB).max())

    # ---------- 3. 官方标签重建核验(独立尝试) ----------
    bi = {b: i for i, b in enumerate(beers)}
    ti = {t: i for i, t in enumerate(tasters)}
    V = np.full((len(beers), len(tasters), 50), np.nan)
    for x in data:
        V[bi[x[1]], ti[x[3]], :] = [float(v) for v in x[5:55]]
    zero_var = int((np.nanstd(V, axis=0) == 0).sum())
    audit["n_taster_descriptor_zero_variance_combos"] = zero_var

    OFFM = np.array([[float(x[h4.index(c)]) for c in official_cols] for x in r4[1:]])
    idx = [raw_index[alias_back.get(c, c)] for c in official_cols]

    def build(mode: str, ddof: int):
        Z = np.full_like(V, np.nan)
        for t in range(len(tasters)):
            for j in range(50):
                v = V[:, t, j]
                m = np.isfinite(v)
                q = v[m]
                mu, sd = q.mean(), q.std(ddof=ddof)
                Z[m, t, j] = (q - mu) / (sd if sd else 1.0)
        return np.nanmean(Z, axis=1)

    def closest(M):
        M2 = M[:, idx]
        D = ((M2[:, None, :] - OFFM[None, :, :]) ** 2).sum(2)
        ri, ci = linear_sum_assignment(D)
        diff = M2[ri] - OFFM[ci]
        return {
            "max_abs_error": float(np.abs(diff).max()),
            "median_row_rmse": float(np.median(np.sqrt((diff ** 2).mean(1)))),
            "n_rows_below_1e-6": int((np.sqrt((diff ** 2).mean(1)) < 1e-6).sum()),
            "n_unique_matched_official_beers": len(set(ci.tolist())),
            "mean_column_pearson_r": float(np.mean([np.corrcoef(M2[ri, j], OFFM[ci, j])[0, 1]
                                                    for j in range(50)])),
        }, ri, ci

    attempts = {}
    for ddof in [0, 1]:
        r, ri, ci = closest(build("taster_desc", ddof))
        attempts[f"taster_descriptor_standardization_ddof{ddof}"] = r
    recon = {
        "claim_in_competition_paper": "max absolute reconstruction error 2.84e-8 (A2601731.pdf §3.2)",
        "independently_reproduced": False,
        "attempts": attempts,
        "diagnosis": (
            "按 (评委 × 描述词) 逐列标准化后取啤酒级均值, 与官方标签列相关 >0.99、"
            "斜率≈1、截距≈0、逐列 SD 高度一致, 250 行一一匹配唯一; 但行 RMSE 中位约 0.02, "
            "且与每款啤酒的记录数相关 (r≈0.73), 达不到 1e-8. 按品评日期分组标准化更差. "
            "说明官方重建另需: 精确的 beer_name→官方 beer 映射、零方差组合的回退定义、"
            "以及可能的场次/权重口径; 竞赛论文未提供该脚本与参数."
        ),
        "paper_writing_rule": (
            "2.84e-8 目前只能作为竞赛论文的自述结果引用, 不能写成『本文独立验证』; "
            "若拟作为 novelty, 必须先补齐可复现的重建脚本并达到该量级, "
            "否则只写『原始 0-6 评分已获得, 可直接构造 A/B/C 三种啤酒级标签』."
        ),
    }
    (OUT / "raw_panel_reconstruction_check.json").write_text(
        json.dumps(recon, ensure_ascii=False, indent=2), encoding="utf-8")

    # beer 映射(指纹匹配结果, 供后续 0-6 五折复跑使用; 声明为推断而非官方确认)
    _, ri, ci = closest(build("taster_desc", 0))
    with (OUT / "raw_panel_beer_map.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beer_name_raw", "official_beer_4csv", "style_raw", "first_tasting_date",
                    "n_records", "mapping_method", "confidence"])
        style_of = {x[1]: x[2] for x in data}
        date_of = {x[1]: x[0] for x in data}
        for i, j in zip(ri, ci):
            b = beers[i]
            w.writerow([b, r4[j + 1][h4.index("beer")], style_of[b], date_of[b], rec_per_beer[b],
                        "hungarian_on_50d_standardized_fingerprint", "inferred_NOT_officially_confirmed"])

    # 用风格分类做独立交叉验证: 原始表 15 类粗分 vs 4.csv 22 类细分
    STYLE_OK = {
        "Low-No Alcohol": {"Low/No alcohol"}, "Pils": {"Pils/Lager"},
        "Blond": {"Blond"}, "Tripel": {"Tripel"}, "Stout & Porter": {"Stout/Porter"},
        "Saison": {"Saison"}, "Hoppy": {"Hoppy"}, "Wit": {"Wheat"},
        "Brett": {"Brett/cofermented"}, "Christmas": {"Christmas"},
        "Strong ale": {"Strong ale"},
        "Amber-Brown-Dubbel": {"Amber", "Brown", "Dubbel"},
        "Flemish ales": {"Flanders old brown", "West Flanders ale"},
        "Geuze & Faro": {"Lambic", "Faro"},
        "Kriek & Fruitbeer": {"Kriek", "Fruitbeer"},
    }
    fine_of = {x[h4.index("beer")]: x[h4.index("tasting_category_fine")] for x in r4[1:]}
    style_of = {x[1]: x[2] for x in data}
    hits = 0
    mismatch = Counter()
    for i, j in zip(ri, ci):
        s, f = style_of[beers[i]], fine_of[r4[j + 1][h4.index("beer")]]
        if f in STYLE_OK.get(s, set()):
            hits += 1
        else:
            mismatch[(s, f)] += 1
    audit["beer_map_style_cross_check"] = {
        "n_style_consistent": hits,
        "n_total": len(beers),
        "pct": round(100 * hits / len(beers), 1),
        "top_mismatches": [{"raw": k[0], "official": k[1], "n": v}
                           for k, v in mismatch.most_common(12)],
        "note": "不一致项多为粗分类无对应的细分风格(Scotch/Brut 等), 支持映射可信; 但映射仍属推断, 未经官方确认",
    }

    (OUT / "raw_panel_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": {k: audit[k] for k in
                                ["n_records", "n_beers", "n_tasters", "n_sensory_descriptors",
                                 "n_scored_cells", "all_integer_0_to_6", "n_non_integer_cells",
                                 "n_taster_descriptor_zero_variance_combos", "AB_max_abs_diff"]},
                      "reconstruction_independently_reproduced": recon["independently_reproduced"]},
                     ensure_ascii=False, indent=2))
    print("outputs:", ", ".join(sorted(p.name for p in OUT.glob("raw_panel_*"))))


if __name__ == "__main__":
    main()
