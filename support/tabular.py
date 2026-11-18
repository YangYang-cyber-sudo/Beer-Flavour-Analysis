"""全量结构化数据体检；所有输入均只读。大文件需预留内存。"""

import csv
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import resolve_path, utc_now


EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".parquet"}
MISSING_TOKENS = {"na", "n/a", "null", "none", "nan", "未知", "缺失", "?", "-"}
ID_PATTERN = re.compile(r"(^id$|_id$|^id_|编号|编码|序号)", re.I)
DATE_PATTERN = re.compile(r"(date|time|日期|时间)", re.I)
LEAK_PATTERN = re.compile(r"(target|label|prediction|predicted|outcome|答案|标签|预测结果)", re.I)


def discover(paths):
    found = []
    for value in paths:
        path = resolve_path(value)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            found.extend(p for p in sorted(path.rglob("*")) if p.suffix.lower() in EXTENSIONS and p.is_file())
        elif path.suffix.lower() in EXTENSIONS:
            found.append(path)
        else:
            raise ValueError("不支持的结构化文件: {}".format(path))
    return list(dict.fromkeys(found))


def read_tables(path, encoding=None, sep=None):
    """保留 CSV/TXT 的前导零、NA 字符和原始文本，另行推断字段类型。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        import openpyxl
        workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        try:
            for sheet in workbook:
                rows = sheet.iter_rows(values_only=True)
                header = list(next(rows, ()))
                # 空工作表仍出现在报告中。公式读取的是缓存值，不执行公式。
                frame = pd.DataFrame(rows, columns=range(len(header)))
                frame.columns = [str(x) if x is not None else "" for x in header]
                yield sheet.title, frame, {"original_header": header, "formula_values": "cached_only"}
        finally:
            workbook.close()
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
        header = list(frame.columns)
        frame.columns = [str(c) for c in header]
        yield "table", frame, {"original_header": header}
    else:
        chosen = encoding
        if chosen is None:
            for candidate in ("utf-8-sig", "gb18030"):
                try:
                    # 增量解码验证全文件，避免只检查文件头漏掉后部编码错误。
                    with path.open(encoding=candidate) as handle:
                        while handle.read(1024 * 1024):
                            pass
                    chosen = candidate
                    break
                except UnicodeDecodeError:
                    continue
            if chosen is None:
                raise ValueError("无法识别编码，请指定 --encoding")
        with path.open(encoding=chosen, newline="") as handle:
            sample = handle.read(65536)
        delimiter = sep
        if delimiter == r"\t":
            delimiter = "\t"
        if delimiter is None:
            if suffix == ".tsv":
                delimiter = "\t"
            else:
                try:
                    delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;| ").delimiter
                except csv.Error:
                    delimiter = ","  # 单列文本也合法，可显式指定分隔符。
        if delimiter != "whitespace" and len(delimiter) != 1:
            raise ValueError("sep 必须是单个字符或 whitespace")
        with path.open(encoding=chosen, newline="") as handle:
            rows = (line.split() for line in handle) if delimiter == "whitespace" else csv.reader(handle, delimiter=delimiter, strict=True)
            header = None
            for number, row in enumerate(rows, 1):
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                if header is None:
                    header = row
                elif len(row) != len(header):
                    raise ValueError("记录 {} 有 {} 列，表头有 {} 列；拒绝错位读取".format(number, len(row), len(header)))
        if header is None:
            raise ValueError("空文件没有表头")
        frame = pd.read_csv(path, sep=r"\s+" if delimiter == "whitespace" else delimiter,
                            encoding=chosen, dtype=object, keep_default_na=False, header=0)
        if len(frame.columns) != len(header):
            raise ValueError("表头与数据列数不一致")
        frame.columns = header
        yield "table", frame, {"original_header": header, "encoding": chosen, "delimiter": delimiter}


def missing_mask(series):
    return series.isna() | series.map(lambda x: isinstance(x, str) and not x.strip())


def value_counts(series, top):
    # 同时保留类型，避免数值 1 和字符串 '1' 在 JSON 键中碰撞。
    counts = series.map(lambda x: (type(x).__name__, str(x))).value_counts(dropna=False)
    return [{"type": key[0], "value": key[1], "count": int(count),
             "ratio_nonmissing": float(count / len(series))} for key, count in counts.head(top).items()]


def profile_column(series, name, top=20, date_format=None, date_requested=False):
    missing = missing_mask(series)
    valid = series[~missing]
    text = valid.astype(str)
    result = {"name": name, "storage_dtype": str(series.dtype), "missing_count": int(missing.sum()),
              "missing_ratio": float(missing.mean()) if len(series) else None,
              "unique_nonmissing": int(valid.nunique()), "inferred_type": "empty"}
    if not len(valid):
        return result
    result["top_values"] = value_counts(valid, top)
    result["top_values_truncated"] = valid.nunique() > top
    result["suspicious_encoding"] = {
        "replacement_character_count": int(text.str.contains("\ufffd", regex=False).sum()),
        "control_character_count": int(text.str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", regex=True).sum()),
        "surrounding_whitespace_count": int((text != text.str.strip()).sum()),
        "missing_token_counts": text[text.str.strip().str.lower().isin(MISSING_TOKENS)].value_counts().to_dict(),
        "leading_zero_numeric_count": int(text.str.match(r"^[+-]?0\d+(?:\.\d+)?$").sum())}
    numeric = pd.to_numeric(valid, errors="coerce")
    numeric_count = int(numeric.notna().sum())
    if numeric_count:
        numbers = numeric[numeric.notna()].astype(float).to_numpy()
        finite = numbers[np.isfinite(numbers)]
        result["numeric_parse_ratio"] = numeric_count / len(valid)
        if numeric_count == len(valid):
            result["inferred_type"] = "numeric"
        if numeric_count >= 0.9 * len(valid):
            stats = {"finite_count": len(finite), "positive_inf": int(np.isposinf(numbers).sum()),
                     "negative_inf": int(np.isneginf(numbers).sum()),
                     "unparseable_count": len(valid) - numeric_count}
            if len(finite):
                qs = np.quantile(finite, [0, .01, .05, .25, .5, .75, .95, .99, 1])
                lo, hi = qs[3] - 1.5 * (qs[5] - qs[3]), qs[5] + 1.5 * (qs[5] - qs[3])
                stats.update({"mean": float(np.mean(finite)), "std_population": float(np.std(finite)),
                              "quantiles": dict(zip(["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"], qs.tolist())),
                              "negative_count": int((finite < 0).sum()), "zero_count": int((finite == 0).sum()),
                              "iqr_outlier_count": int(((finite < lo) | (finite > hi)).sum()),
                              "iqr_bounds": [float(lo), float(hi)],
                              "possible_sentinel_counts": {str(x): int((finite == x).sum()) for x in [-9999, -999, -99, 9999, 99999] if (finite == x).any()}})
            result["numeric"] = stats
    if result["inferred_type"] == "empty":
        result["inferred_type"] = "categorical_or_text"
    native_date = pd.api.types.is_datetime64_any_dtype(series) or valid.map(lambda x: isinstance(x, (pd.Timestamp, datetime, date))).all()
    iso_mask = text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$")
    iso = iso_mask.any()
    candidate = native_date or iso or date_requested or bool(DATE_PATTERN.search(name))
    if candidate:
        # 不把普通数字或含糊的日/月字符串当成纳秒日期。
        parse = native_date or iso or date_format is not None
        if parse:
            date_values = valid if native_date or date_format else valid.where(iso_mask, None)
            dates = pd.to_datetime(date_values, format=date_format, errors="coerce", utc=True)
            ok = dates.dropna()
            result["datetime"] = {"parsed_count": len(ok), "invalid_count": int(dates.isna().sum()),
                                  "min_utc": ok.min().isoformat() if len(ok) else None,
                                  "max_utc": ok.max().isoformat() if len(ok) else None,
                                  "monotonic_in_file": bool(ok.is_monotonic_increasing),
                                  "duplicate_time_count": int(ok.duplicated().sum()),
                                  "format": date_format or "native_or_ISO", "timezone_note": "naive timestamps interpreted as UTC for comparison; confirm official timezone"}
            if len(ok) == len(valid):
                result["inferred_type"] = "datetime"
        else:
            result["datetime"] = {"status": "needs_explicit_format", "note": "疑似日期字段；未猜测日月顺序或整数时间单位"}
    return result


def profile_table(frame, id_cols=(), target=None, top=20, date_cols=(), date_format=None):
    names = list(frame.columns)
    duplicated_names = sorted({str(x) for x in names if names.count(x) > 1})
    result = {"rows": len(frame), "columns": len(names), "field_names": names,
              "duplicate_field_names": duplicated_names,
              "blank_field_positions": [i for i, c in enumerate(names) if not str(c).strip()],
              "duplicate_rows_excess": int(frame.duplicated().sum()),
              "duplicate_rows_all_members": int(frame.duplicated(keep=False).sum()),
              "fields": [], "business_ids": [], "leakage_signals": []}
    for i, name in enumerate(names):
        field = profile_column(frame.iloc[:, i], name, top, date_format, name in date_cols)
        field["position"] = i
        result["fields"].append(field)
    if duplicated_names:
        result["checks_skipped"] = "名称重复；业务键、标签及跨表比较须先确认列位置"
        return result
    groups = [(list(id_cols), "explicit_composite_key")] if id_cols else []
    groups += [([name], "name_heuristic") for name in names if ID_PATTERN.search(name) and name not in id_cols]
    for cols, origin in groups:
        absent = [c for c in cols if c not in names]
        if absent:
            result["business_ids"].append({"columns": cols, "origin": origin, "missing_columns": absent})
            continue
        complete = ~pd.concat([missing_mask(frame[c]) for c in cols], axis=1).any(axis=1)
        keys = frame.loc[complete, cols]
        result["business_ids"].append({"columns": cols, "origin": origin,
                                       "missing_key_rows": int((~complete).sum()), "unique_keys": len(keys.drop_duplicates()),
                                       "duplicate_excess": int(keys.duplicated().sum()),
                                       "duplicate_all_members": int(keys.duplicated(keep=False).sum())})
    signals = result["leakage_signals"]
    if target and target not in names:
        signals.append({"kind": "target_absent", "column": target})
    for name in names:
        if name != target and LEAK_PATTERN.search(name):
            signals.append({"kind": "suspicious_name", "column": name})
    if target and target in names:
        y = frame[target]
        for name in names:
            if name == target:
                continue
            x = frame[name]
            mask = ~missing_mask(x) & ~missing_mask(y)
            if mask.sum() < 3 or y[mask].nunique() < 2:
                continue
            if (x[mask].astype(str) == y[mask].astype(str)).all():
                signals.append({"kind": "target_copy_on_observed_rows", "column": name, "rows": int(mask.sum())})
                continue
            xn, yn = pd.to_numeric(x[mask], errors="coerce"), pd.to_numeric(y[mask], errors="coerce")
            finite = np.isfinite(xn) & np.isfinite(yn)
            if finite.sum() >= 3 and xn[finite].nunique() > 1 and yn[finite].nunique() > 1:
                corr = float(xn[finite].corr(yn[finite]))
                if abs(corr) >= .9999:
                    signals.append({"kind": "near_perfect_linear_association", "column": name, "correlation": corr,
                                    "rows": int(finite.sum())})
    return result


def compare_tables(left, right, id_cols=(), target=None):
    if not left.columns.is_unique or not right.columns.is_unique:
        return {"status": "skipped_duplicate_headers"}
    common = [c for c in left.columns if c in right.columns]
    result = {"train_only_fields": [c for c in left.columns if c not in right.columns],
              "test_only_fields": [c for c in right.columns if c not in left.columns],
              "common_field_order_equal": common == [c for c in right.columns if c in left.columns],
              "storage_dtype_differences": [{"field": c, "train": str(left[c].dtype), "test": str(right[c].dtype)}
                                             for c in common if left[c].dtype != right[c].dtype]}
    keys = list(id_cols)
    if keys and all(c in common for c in keys):
        def key_set(df):
            valid = ~pd.concat([missing_mask(df[c]) for c in keys], axis=1).any(axis=1)
            return set(df.loc[valid, keys].astype(str).itertuples(index=False, name=None))
        result["overlapping_business_keys"] = len(key_set(left) & key_set(right))
    features = [c for c in common if c != target and c not in id_cols]
    if features:
        # 合并去重键，避免多对多连接导致重复行数膨胀。
        a, b = left[features].copy(), right[features].copy()
        for c in features:
            a[c] = a[c].map(lambda x: ("missing", "") if pd.isna(x) or (isinstance(x, str) and not x.strip()) else ("value", str(x)))
            b[c] = b[c].map(lambda x: ("missing", "") if pd.isna(x) or (isinstance(x, str) and not x.strip()) else ("value", str(x)))
        aset = set(a.itertuples(index=False, name=None))
        result["test_rows_matching_train_features"] = sum(row in aset for row in b.itertuples(index=False, name=None))
        result["feature_overlap_fields"] = features
    result["interpretation"] = "字段差异、实体或特征重叠仅为线索；时间预测中实体重叠可能合理，须按赛题审核"
    return result


def audit(paths=(), train=(), test=(), templates=(), id_cols=(), target=None,
          encoding=None, sep=None, top=20, date_cols=(), date_format=None):
    if top < 1:
        raise ValueError("top 必须大于零")
    report = {"kind": "tabular_audit", "created_at": utc_now(), "status": "ok", "files": [],
              "comparisons": [], "limitations": ["全量读取，不修改输入；内存需求取决于数据体积。", "默认只把空白和原生空值记为缺失，NA/null 等原文单独列为可疑编码。", "IQR、字段名称及相关性均为启发式信号，不能证明异常或无泄漏。", "须人工检查每个字段在预测时是否可得、实体与时间关系，并设计验证划分。", "XLSX 只读取公式缓存值；公式未缓存时可能表现为缺失。"]}
    loaded = {"train": [], "test": [], "template": []}
    for role, inputs in (("data", paths), ("train", train), ("test", test), ("template", templates)):
        for path in discover(inputs):
            entry = {"path": str(path), "role": role, "size_bytes": path.stat().st_size, "sheets": []}
            try:
                for sheet, frame, metadata in read_tables(path, encoding, sep):
                    profile = profile_table(frame, id_cols, target if role != "template" else None, top, date_cols, date_format)
                    profile.update({"name": sheet, "reader": metadata})
                    entry["sheets"].append(profile)
                    if role in loaded:
                        loaded[role].append((str(path), sheet, frame, profile))
            except Exception as exc:
                entry["error"] = "{}: {}".format(type(exc).__name__, exc)
                report["status"] = "partial_failure"
            report["files"].append(entry)
    if not report["files"]:
        raise ValueError("未找到支持的数据文件")
    for lp, ls, lf, lprof in loaded["train"]:
        for rp, rs, rf, rprof in loaded["test"]:
            pair = {"train": lp, "train_sheet": ls, "test": rp, "test_sheet": rs}
            if ls != rs and not (len(loaded["train"]) == len(loaded["test"]) == 1):
                pair["status"] = "skipped_unmatched_sheet_names"
            else:
                pair.update(compare_tables(lf, rf, id_cols, target))
                pair["inferred_type_differences"] = [{"field": c, "train": a["inferred_type"], "test": b["inferred_type"]}
                    for a in lprof["fields"] for b in rprof["fields"] for c in [a["name"]]
                    if c == b["name"] and a["inferred_type"] != b["inferred_type"]]
            report["comparisons"].append(pair)
    for tp, ts, tf, _ in loaded["template"]:
        for sp, ss, sf, _ in loaded["test"]:
            result = {"template": tp, "template_sheet": ts, "test": sp, "test_sheet": ss,
                      "template_rows": len(tf), "test_rows": len(sf), "row_count_equal": len(tf) == len(sf),
                      "note": "模板行数/顺序要求须以官方规则为准"}
            if id_cols and all(c in tf.columns and c in sf.columns for c in id_cols) and tf.columns.is_unique and sf.columns.is_unique:
                result["id_order_equal"] = tf[list(id_cols)].astype(str).reset_index(drop=True).equals(sf[list(id_cols)].astype(str).reset_index(drop=True))
            report.setdefault("template_checks", []).append(result)
    return report
