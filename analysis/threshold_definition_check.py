# -*- coding: utf-8 -*-
"""
核查 compound_threshold_profile.csv 中「超阈值样品比例」的定义与稳健性。

定义:
    pct_above_threshold = 100 * #{v_i > threshold_mean} / N
    其中 v_i 为 8.xlsx -> Calibrated chemical dataset 中该化合物的真实校准浓度,
    N 为该列可解析为数值的样品数(非恒定 250),
    threshold_mean 为同文件 Sensory thresholds 表的文献感官阈值汇总均值。

输出:
    threshold_definition_check.csv —— 18 化合物, 含分母、单位一致性、
    mean/median 两种口径下的超阈比例及差值。
"""
import csv
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.paths import DATA, DERIVED, TABLES, ensure_output_dirs, configure_console  # noqa: E402

configure_console()
ensure_output_dirs()
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX = DATA / "8.xlsx"
PROFILE = TABLES / "compound_threshold_profile.csv"
FIELD_DICT = DERIVED / "q3_field_dictionary.csv"


def col_idx(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(z: zipfile.ZipFile, shared: list, i: int) -> list:
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
            d[col_idx(c.get("r"))] = val
        out.append([d.get(k, "") for k in range(max(d) + 1)] if d else [])
    return out


def main():
    z = zipfile.ZipFile(XLSX)
    shared = [
        "".join(t.text or "" for t in si.iter(NS + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
    ]
    th = read_sheet(z, shared, 1)
    cal = read_sheet(z, shared, 5)

    th_map = {x[0]: x for x in th[1:] if x and x[0]}
    hdr = cal[0]
    idx = {n: i for i, n in enumerate(hdr)}

    # 竞赛包自带的字段单位字典(独立于阈值表), 用于交叉核对单位是否一致
    dict_unit = {}
    if FIELD_DICT.exists():
        with open(FIELD_DICT, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                dict_unit[row["field"]] = row.get("unit", "")

    prof = {
        x["compound"]: x
        for x in csv.DictReader(open(PROFILE, encoding="utf-8-sig"))
    }
    out = []
    for name in sorted(prof):
        t = th_map[name]
        mean_thr = float(t[9])
        med_thr = float(t[10]) if t[10] not in ("", "NA") else float("nan")
        vals = []
        for row in cal[1:]:
            if len(row) > idx[name]:
                try:
                    vals.append(float(row[idx[name]]))
                except (ValueError, TypeError):
                    pass
        if not vals:
            continue
        p_mean = 100 * sum(1 for v in vals if v > mean_thr) / len(vals)
        p_med = (
            100 * sum(1 for v in vals if v > med_thr) / len(vals)
            if med_thr == med_thr
            else float("nan")
        )
        d_unit = dict_unit.get(name, "")
        out.append(
            dict(
                compound=name,
                chemical_name=t[2],
                unit_threshold_table=t[6],
                unit_field_dictionary=d_unit,
                unit_consistent=("yes" if d_unit and d_unit == t[6] else "UNVERIFIED"),
                n_samples=len(vals),
                threshold_mean=round(mean_thr, 6),
                threshold_median=round(med_thr, 6) if med_thr == med_thr else "",
                median_conc_over_threshold_mean=round(
                    sorted(v for v in vals if v > 0)[len([v for v in vals if v > 0]) // 2]
                    / mean_thr,
                    3,
                ),
                pct_above_mean=round(p_mean, 1),
                pct_above_median=round(p_med, 1) if p_med == p_med else "",
                sensitivity_pp=round(p_mean - p_med, 1) if p_med == p_med else "",
                flag=("UNSTABLE" if abs(p_mean - p_med) >= 5 else "stable"),
            )
        )

    dest = TABLES / "threshold_definition_check.csv"
    with open(dest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"saved {len(out)} rows -> {dest}")
    print("\n=== 口径敏感性 >= 5 pp 的化合物 ===")
    for x in out:
        if x["flag"] == "UNSTABLE":
            print(
                f"  {x['compound']:16s} mean={x['threshold_mean']:<10g} "
                f"median={x['threshold_median']:<10g} "
                f"{x['pct_above_mean']:5.1f}% -> {x['pct_above_median']:5.1f}% "
                f"(diff {x['sensitivity_pp']:+.1f} pp)"
            )
    print("\n=== 单位未核实 ===")
    for x in out:
        if x["unit_consistent"] != "yes":
            print(
                f"  {x['compound']:16s} threshold={x['unit_threshold_table']:8s} "
                f"dict={x['unit_field_dictionary']!r}"
            )
    print("\n=== 分母不足 250 ===")
    for x in out:
        if x["n_samples"] != 250:
            print(f"  {x['compound']:16s} n={x['n_samples']}")


if __name__ == "__main__":
    main()
