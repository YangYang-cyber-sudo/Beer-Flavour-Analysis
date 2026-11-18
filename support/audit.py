"""只读检查正式附件，复用本包的读取、统计和输出工具。"""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import importlib
import json
from pathlib import Path
import platform
import re
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
import openpyxl

from .common import clean_json, protect_output, write_json
from .paths import configure_console
from .tabular import read_tables, profile_table


META = ['beer', 'beer_id', 'tasting_category_fine']
TOL_ABS = 5.1e-8
TOL_REL = 5e-9


def normalize_missing(frame):
    """本数据说明明确用 NA 表示缺失；仅转换副本，不更改官方文件。"""
    result = frame.copy()
    for col in result:
        result[col] = result[col].map(
            lambda v: np.nan if isinstance(v, str) and (v == 'NA' or not v.strip()) else v)
    return result


def numeric(frame):
    return normalize_missing(frame).apply(pd.to_numeric, errors='raise').astype(float)


def indexed(frame, key='beer'):
    data = frame.copy()
    if data[key].isna().any() or data[key].astype(str).str.strip().eq('').any():
        raise ValueError('样本键缺失: ' + key)
    data[key] = data[key].astype(str)
    if data[key].isna().any() or data[key].duplicated().any():
        raise ValueError('样本键不唯一或缺失: ' + key)
    return data.set_index(key)


def norm_name(value):
    # 附件 8 的希腊字母存在 UTF-8 被按西文解释的痕迹；只处理已知字符。
    text = str(value).replace('_x000D_', '\r').replace('Î±', 'α').replace('Î²', 'β')
    text = text.lower().replace('alpha', 'α').replace('beta', 'β')
    return re.sub(r'[^a-z0-9αβγδ]', '', text)


def unique_name_map(records, aliases, key, allowed):
    candidates = defaultdict(set)
    for _, row in records.iterrows():
        if row[key] not in allowed:
            continue
        for alias in aliases:
            if pd.notna(row[alias]) and str(row[alias]) != 'NA':
                candidates[norm_name(row[alias])].add(str(row[key]))
    ambiguous = {k: sorted(v) for k, v in candidates.items() if len(v) != 1}
    if ambiguous:
        raise ValueError('名称对应有歧义，拒绝按位置猜测: ' + str(ambiguous))
    return {k: next(iter(v)) for k, v in candidates.items()}


def write_csv(output, name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(output / name, index=False, encoding='utf-8-sig')


def workbook_inventory(path):
    book = openpyxl.load_workbook(path, read_only=False, data_only=False)
    details = []
    for sheet in book:
        cells = [cell for row in sheet for cell in row if cell.value is not None]
        details.append({
            'sheet': sheet.title, 'physical_rows': sheet.max_row,
            'physical_columns': sheet.max_column,
            'effective_rows': max((c.row for c in cells), default=0),
            'effective_columns': max((c.column for c in cells), default=0),
            'nonempty_cells': len(cells), 'formula_count': sum(c.data_type == 'f' for c in cells),
            'merged_ranges': [str(r) for r in sheet.merged_cells.ranges]})
    book.close()
    return details


def scorecard_audit(path):
    formulas = openpyxl.load_workbook(path, data_only=False)
    cached = openpyxl.load_workbook(path, data_only=True)
    summary = formulas['summary']
    schema, cards, raw_labels = [], [], []
    fields = [c.value for c in summary[1]]
    for pos, field in enumerate(fields, 1):
        if field in ('Beer', 'comments', None):
            continue
        raw_name = str(field)
        alias = {'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity',
                 '4vg': 'X4vg'}.get(raw_name, raw_name)
        schema.append({'summary_name': raw_name, 'panel_field': alias,
                       'summary_column': pos, 'example_formula': summary.cell(2, pos).value})
    primary_rows = {7, 20, 33, 45, 50, 54, 58, 62, 68, 72, 76, 82, 86, 90, 94}
    for sheet in formulas:
        if sheet.title == 'summary':
            continue
        mark_cells = set()
        for row in primary_rows:
            for col in (3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 29, 31):
                mark_cells.add((row, col))
        for row in (10, 12, 14, 16, 23, 25, 27, 29, 36, 38, 40, 42):
            mark_cells.update(((row, 3), (row, 19)))
        filled = [{'cell': sheet.cell(r, c).coordinate, 'value': sheet.cell(r, c).value}
                  for r, c in sorted(mark_cells) if sheet.cell(r, c).value is not None]
        references = defaultdict(list)
        zeros = 0
        for item in schema:
            col = item['summary_column']
            row = next(r for r in range(2, summary.max_row + 1)
                       if str(summary.cell(r, 1).value) == str(sheet['S3'].value))
            formula = summary.cell(row, col).value
            refs = re.findall(r'!\$?([A-Z]+)\$?(\d+)', str(formula))
            for ref in refs:
                references[''.join(ref)].append(item['summary_name'])
            value = cached['summary'].cell(row, col).value
            zeros += int(value == 0)
            # 空白表单公式返回 0，不视为人工标签。
            raw_labels.append({'card': sheet.title, 'beer': str(sheet['S3'].value),
                               'field': item['panel_field'], 'cached_value': value,
                               'has_any_card_input': bool(filled),
                               'usable_raw_label': False if not filled else 'requires_manual_review'})
        cards.append({'sheet': sheet.title, 'beer_number': sheet['S3'].value,
                      'filled_mark_cells': filled, 'summary_cached_zero_count': zeros,
                      'duplicate_summary_references': {k: v for k, v in references.items() if len(v) > 1},
                      'acetaldehyde_card_cells_not_in_summary': ['Q45', 'AG45'],
                      'barnyard_label_row': 89, 'barnyard_scoring_cell': 'AG90'})
    formulas.close()
    cached.close()
    return schema, cards, raw_labels


def alignment_audit(tables, book):
    base = indexed(tables['1.csv'])
    records, samples = [], []
    sources = {k: tables[k] for k in ('4.csv', '7.csv')}
    sources.update({k: book[k] for k in ('Calibrated chemical dataset', 'Beer Prices')})
    shap = book['GBM SHAP Values']
    for group, frame in shap.groupby('data', sort=False):
        sources['GBM SHAP Values/' + str(group)] = frame
    for name, frame in sources.items():
        other = indexed(frame)
        common = base.index.intersection(other.index)
        target = other.reindex(base.index)
        rounded = name in ('4.csv', '7.csv')
        id_compatible = []
        for beer in common:
            exact = Decimal(str(base.at[beer, 'beer_id']))
            observed = Decimal(str(other.at[beer, 'beer_id']))
            # 四/七号 ID 保留的十进制精度；仅核对兼容，不能用它恢复身份。
            half_unit = Decimal(10) ** observed.as_tuple().exponent / 2
            compatible = abs(exact-observed) <= half_unit if rounded else exact == observed
            id_compatible.append(compatible)
            style_col = 'tasting_category_fine' if 'tasting_category_fine' in other else 'tasting_category'
            samples.append({'source': name, 'beer': beer, 'canonical_beer_id': str(exact),
                            'source_beer_id': str(other.at[beer, 'beer_id']),
                            'source_style': other.at[beer, style_col],
                            'id_precision_compatible': compatible})
        style_equal = None
        if 'tasting_category_fine' in other:
            style_equal = int((base.loc[common, 'tasting_category_fine'] == other.loc[common, 'tasting_category_fine']).sum())
        records.append({'source': name, 'rows': len(frame), 'unique_beer': len(other),
                        'unique_beer_id': frame['beer_id'].astype(str).nunique(),
                        'beer_missing_from_source': sorted(set(base.index)-set(other.index)),
                        'extra_beer': sorted(set(other.index)-set(base.index)),
                        'row_order_matches_1': list(other.index) == list(base.index),
                        'style_exact_matches': style_equal, 'id_compatible_count': sum(id_compatible),
                        'join_key': 'beer; validate one_to_one; full ID/style cross-check'})
    collision = tables['4.csv'].groupby('beer_id')['beer'].agg(list)
    collisions = [{'rounded_beer_id': k, 'beers': v, 'size': len(v)} for k, v in collision.items() if len(v) > 1]
    return records, samples, collisions, int(sum(len(v)**2 for v in collision))


def matrix_audit(tables, x, y, chemical_names, sensory_names):
    details, maps = [], []
    for name in ('2.csv', '5.csv', '6.csv'):
        frame = tables[name]
        matrix = numeric(frame.set_index(frame.columns[0]))
        row_map = sensory_names if name == '5.csv' else chemical_names
        col_map = chemical_names if name == '2.csv' else sensory_names
        missing = [str(k) for k in matrix.index if norm_name(k) not in row_map]
        missing += [str(k) for k in matrix.columns if norm_name(k) not in col_map]
        if missing:
            raise ValueError(name + ' 名称无法可靠对齐: ' + str(missing))
        rn = [row_map[norm_name(k)] for k in matrix.index]
        cn = [col_map[norm_name(k)] for k in matrix.columns]
        if len(set(rn)) != len(rn) or len(set(cn)) != len(cn):
            raise ValueError(name + ' 名称映射不是一一对应')
        source = x if name == '2.csv' else (y if name == '5.csv' else pd.concat([x, y], axis=1))
        original = matrix.to_numpy()
        entry = {'file': name, 'matrix_rows': len(matrix), 'matrix_columns': len(matrix.columns),
                 'missing_cells': int(np.isnan(original).sum()), 'out_of_range_cells': int((np.abs(original)>1+1e-10).sum())}
        for method in ('pearson', 'spearman'):
            computed = source.corr(method=method).loc[rn, cn].to_numpy()
            delta = np.abs(computed-original)
            entry[method + '_max_abs_error'] = float(np.nanmax(delta))
            entry[method + '_mean_abs_error'] = float(np.nanmean(delta))
            entry[method + '_allclose_1e-10'] = bool(np.allclose(computed, original, atol=1e-10, equal_nan=True))
        if name != '6.csv':
            entry['symmetry_max_error'] = float(np.nanmax(np.abs(original-original.T)))
            entry['diagonal_max_error_from_one'] = float(np.max(np.abs(np.diag(original)-1)))
        details.append(entry)
        maps.extend({'file': name, 'axis': axis, 'display_name': display, 'field': field}
                    for axis, names, fields in [('row', matrix.index, rn), ('column', matrix.columns, cn)]
                    for display, field in zip(names, fields))
    return details, maps


def transformation_audit(x, raw, metadata):
    summaries, exceptions, unusual = [], [], []
    by_field = metadata.drop_duplicates('field_name_R').set_index('field_name_R')
    for f in x:
        md = by_field.loc[f] if f in by_field.index else None
        entry = {'field': f, 'unit_raw_metadata': md['unit'] if md is not None else None,
                 'calibrated_metadata': md['calibrated'] if md is not None else None,
                 'chemical_display_name': md['chemical_name_short'] if md is not None else None,
                 'origin_metadata': md['origin_paper'] if md is not None else None,
                 'threshold_mean_raw_unit': md['threshold_mean'] if md is not None else None,
                 'threshold_median_raw_unit': md['threshold_median'] if md is not None else None,
                 'raw8_available': f in raw, 'one_missing': int(x[f].isna().sum()),
                 'one_negative': int((x[f]<0).sum()), 'one_zero': int((x[f]==0).sum())}
        if f not in raw:
            entry['transform_status'] = 'unverified_no_raw_pair'
            summaries.append(entry)
            continue
        a, b = raw[f], x[f]
        both = a.notna() & b.notna()
        pos = both & (a>0)
        errors = np.abs(np.log10(a[pos])-b[pos])
        is_log = bool(len(errors) and (errors<1e-7).all())
        expected = a.copy()
        if is_log:
            positive_min = a[a>0].min()
            replacement = positive_min/2
            expected = np.log10(a.where(a>0, replacement))
            expected[a.isna()] = np.nan
            entry.update({'transform_status': 'log10_positive_half_min_for_nonpositive',
                          'positive_log10_max_abs_error': float(errors.max()),
                          'nonpositive_raw_count': int((a<=0).sum()),
                          'replacement_raw': replacement})
        else:
            entry['transform_status'] = 'mostly_identity_with_exceptions'
        same = np.isclose(expected, b, atol=TOL_ABS, rtol=TOL_REL, equal_nan=True)
        finite_disagree = both & ~same
        entry.update({'raw_missing': int(a.isna().sum()),
                      'comparison_tolerance': 'abs=5.1e-8 + rel=5e-9 (stored rounding)',
                      'finite_mismatch_count': int(finite_disagree.sum()),
                      'raw_missing_but_one_present': int((a.isna()&b.notna()).sum()),
                      'raw_present_but_one_missing': int((a.notna()&b.isna()).sum())})
        for beer in x.index[~same]:
            exceptions.append({'beer': beer, 'field': f, 'raw8': a[beer], 'input1': b[beer],
                               'expected_by_dominant_transform': expected[beer],
                               'type': 'finite_value_disagreement' if both[beer] else 'missingness_disagreement'})
        for beer in a.index[a<0]:
            unusual.append({'beer': beer, 'field': f, 'raw8': a[beer], 'input1': b[beer],
                            'action': 'retain; investigate measurement/derivation; do not silently clip'})
        summaries.append(entry)
    return summaries, exceptions, unusual


def shap_audit(book, base, x):
    source = book['GBM SHAP Values']
    groups = source['data'].value_counts().to_dict()
    concentration = indexed(source[source['data']=='concentration']).reindex(base.index)
    contributions = indexed(source[source['data']=='SHAP']).reindex(base.index)
    c = numeric(concentration[x.columns])
    s = numeric(contributions[x.columns])
    equal = np.isclose(c.to_numpy(), x.to_numpy(), atol=1e-12, rtol=1e-12, equal_nan=True)
    mdi = book['GBM importance MDI']
    fields = mdi.iloc[:, 1].astype(str)
    return {'row_types': groups, 'independent_beers': source['beer'].nunique(),
            'feature_columns': len(x.columns), 'concentration_equals_input1_including_missing': bool(equal.all()),
            'concentration_max_finite_difference': float(np.nanmax(np.abs(c.to_numpy()-x.to_numpy()))),
            'shap_missing_cells': int(s.isna().sum().sum()),
            'base_prediction_unique': numeric(source[['base_prediction']])['base_prediction'].unique().tolist(),
            'mdi_rows': len(mdi), 'mdi_header': list(mdi.columns),
            'mdi_importance_sum': float(pd.to_numeric(mdi.iloc[:, 0]).sum()),
            'mdi_unique_features': fields.nunique(), 'mdi_feature_set_matches_input1': set(fields)==set(x.columns),
            'target_identified': False, 'training_protocol_identified': False,
            'note': 'No target/model/split/link-function provenance; SHAP summation alone cannot identify target or validate accuracy.'}


def target_mapping(schema, cards, y, sensory_dictionary, forum):
    panel = sensory_dictionary[sensory_dictionary['dataset']=='Trained panel']
    forum_dict = sensory_dictionary[sensory_dictionary['dataset']!='Trained panel']
    panel = panel.set_index('R_name')
    records = []
    for item in schema:
        field = item['panel_field']
        md = panel.loc[field]
        same_name = norm_name(md['CleanName_PanelFeedback'])
        candidates = sorted({str(r.R_name) for r in forum_dict.itertuples()
                             if pd.notna(r.CleanName_PanelFeedback)
                             and norm_name(r.CleanName_PanelFeedback)==same_name})
        vals = y[field]
        records.append(dict(item, scope='Aroma' if field.startswith('A_') else ('Flavour' if field.startswith('F_') else 'Other'),
                            metadata_type=md['DataType'], display_name=md['CleanName_PanelFeedback'],
                            valid_standardized_panel=int(vals.notna().sum()), usable_raw_0_6=0 if all(not c['filled_mark_cells'] for c in cards) else 'manual_review',
                            min=float(vals.min()), max=float(vals.max()), mean=float(vals.mean()),
                            std_population=float(vals.std(ddof=0)),
                            zero_count=int((vals==0).sum()), unique_values=int(vals.nunique()),
                            candidate_forum_fields=';'.join(candidates),
                            candidate_forum_present=';'.join(c for c in candidates if c in forum.columns),
                            warning='Forum name correspondence is not equivalent supervision; raw inverse unavailable'))
    for side, cell in [('Aroma','Q45'),('Flavour','AG45')]:
        records.append({'summary_name': None, 'panel_field': 'acetaldehyde_card_only_'+side,
                        'scope': side, 'display_name': 'Acetaldehyde', 'example_formula': cell,
                        'valid_standardized_panel': 0, 'usable_raw_0_6': 0,
                        'warning': 'Card dimension absent from summary and 4.csv; no supervised target'})
    return records


def defect_audit(raw, metadata, base):
    records, per_beer = [], []
    for label, field in [('DMS','DMS'), ('Diacetyl','X2.3.butanedione'), ('4VG','X4VG'),
                         ('trans-2-nonenal','X.E..2.nonenal'), ('Stale hops',None)]:
        if field is None:
            records.append({'defect': label, 'threshold_available': False,
                            'note': 'Sensory descriptor stale_hops, no single concentration/threshold supplied'})
            continue
        md = metadata[metadata['field_name_R']==field].iloc[0]
        rec = {'defect': label, 'field': field, 'raw_unit': md['unit'],
               'raw_available': int(raw[field].notna().sum()),
               'note': 'Threshold sensitivity analysis; style context and source conflicts still require scientific review'}
        for kind in ['mean','median']:
            threshold = pd.to_numeric(pd.Series([md['threshold_'+kind]]), errors='coerce').iloc[0]
            available = pd.notna(threshold) and threshold>0
            rec['threshold_'+kind] = threshold
            rec['exceed_'+kind] = int((raw[field]>threshold).sum()) if available else None
            if available:
                for beer, value in raw[field].items():
                    per_beer.append({'beer': beer, 'style': base.at[beer,'tasting_category_fine'],
                                     'defect': label, 'raw8_concentration': value, 'unit': md['unit'],
                                     'threshold_type': kind, 'threshold': threshold,
                                     'OAV': value/threshold if pd.notna(value) else np.nan,
                                     'exceeds': bool(value>threshold) if pd.notna(value) else None})
        rec['threshold_available'] = rec['exceed_mean'] is not None
        records.append(rec)
    return records, per_beer


def run_audit(data_dir, output):
    csv_names = ['1.csv','2.csv','4.csv','5.csv','6.csv','7.csv']
    required = csv_names+['3.xlsx','8.xlsx','数据说明.txt']
    missing = [n for n in required if not (data_dir/n).is_file()]
    if missing:
        raise FileNotFoundError('缺少官方附件: '+', '.join(missing))
    tables, inventories, field_rows = {}, [], []
    profile_dir = output/'profiles'
    profile_dir.mkdir()
    for name in csv_names:
        _, frame, reader = next(iter(read_tables(data_dir/name)))
        tables[name] = frame
        normalized = normalize_missing(frame)
        prof = profile_table(normalized, id_cols=['beer'] if 'beer' in frame else (), top=8)
        write_json(profile_dir/(name+'.json'), dict(prof, reader=reader,
                   original_NA_token_count=int((frame=='NA').sum().sum())))
        inventories.append({'file': name, 'sheet': None, 'rows': len(frame), 'columns': len(frame.columns),
                            'complete_duplicate_excess': int(frame.duplicated().sum()),
                            'missing_cells_after_NA_normalization': int(normalized.isna().sum().sum())})
        for field in prof['fields']:
            field_rows.append({'file': name, 'sheet': None, 'field': field['name'],
                               'position': field['position'], 'type': field['inferred_type'],
                               'missing': field['missing_count'], 'missing_ratio': field['missing_ratio'],
                               'unique': field['unique_nonmissing'],
                               'constant': field['unique_nonmissing']==1,
                               **{k:field.get('numeric',{}).get(k) for k in ['negative_count','zero_count','iqr_outlier_count']}})
        print('READ', name, frame.shape, flush=True)
    book = {name:frame for name,frame,_ in read_tables(data_dir/'8.xlsx')}
    workbooks = {}
    for filename in ['3.xlsx','8.xlsx']:
        workbooks[filename] = workbook_inventory(data_dir/filename)
        for sheet in workbooks[filename]:
            inventories.append({'file': filename, **sheet})
    for number, (sheet, frame) in enumerate(book.items(), 1):
        if not frame.size:
            continue
        prof = profile_table(normalize_missing(frame), id_cols=['beer'] if 'beer' in frame and sheet!='GBM SHAP Values' else (), top=8)
        write_json(profile_dir/('8_sheet_%02d.json'%number), dict(prof, sheet=sheet))
        for field in prof['fields']:
            field_rows.append({'file':'8.xlsx','sheet':sheet,'field':field['name'],'position':field['position'],
                               'type':field['inferred_type'],'missing':field['missing_count'],
                               'missing_ratio':field['missing_ratio'],'unique':field['unique_nonmissing'],
                               'constant':field['unique_nonmissing']==1})
    base = indexed(tables['1.csv'])
    x = numeric(base.drop(columns=['beer_id','tasting_category_fine']))
    y = numeric(indexed(tables['4.csv']).reindex(base.index).drop(columns=['beer_id','tasting_category_fine']))
    raw = numeric(indexed(book['Calibrated chemical dataset']).reindex(base.index).drop(columns=['beer_id','tasting_category_fine']))
    metadata = normalize_missing(book['Sensory thresholds'])
    sensory = normalize_missing(book['Sensory names'])
    chemical_names = unique_name_map(metadata, ['chemical_name','chemical_name_short'], 'field_name_R', set(x.columns))
    panel_names = unique_name_map(sensory[sensory['dataset']=='Trained panel'], ['CleanName_PanelFeedback'], 'R_name', set(y.columns))
    align, sample_rows, collisions, badjoin = alignment_audit(tables, book)
    matrices, matrix_map = matrix_audit(tables, x, y, chemical_names, panel_names)
    transforms, exceptions, negative_raw = transformation_audit(x, raw, metadata)
    schema, cards, raw_labels = scorecard_audit(data_dir/'3.xlsx')
    mapping = target_mapping(schema, cards, y, sensory, tables['7.csv'])
    defect, oavs = defect_audit(raw, metadata, base)
    shap = shap_audit(book, base, x)
    style_counts = base['tasting_category_fine'].value_counts().rename_axis('style').reset_index(name='count')
    style_counts['ratio'] = style_counts['count']/len(base)
    feature_dups = x.duplicated(keep=False)
    duplicates = base.loc[feature_dups, ['beer_id','tasting_category_fine']].reset_index()
    # 相同向量也不能跨折；当前源表只有每款啤酒一行，不把跨文件复用当新样本。
    negative_features = x.min().sort_values().head(12).to_dict()
    forum_normalized = normalize_missing(tables['7.csv'])
    all_na_forum = [c for c in tables['7.csv'].columns[3:] if forum_normalized[c].isna().all()]
    floor = np.isclose(x.to_numpy(), -1.81781854, atol=5e-10, rtol=0)
    summary = {'team':'MCB2601731','status':'2026A_DATA_READ_PASS_LABEL_DEFINITION_PENDING',
               'formal_models_trained': False,'rows': len(base),'chemical_features': len(x.columns),
               'panel_targets': len(y.columns),'panel_missing_cells': int(y.isna().sum().sum()),
               'chemical_missing_cells': int(x.isna().sum().sum()),
               'chemical_constant_fields': x.columns[x.nunique(dropna=True)<=1].tolist(),
               'feature_duplicate_members': int(feature_dups.sum()),
               'panel_complete_feature_duplicate_excess': int(y.duplicated().sum()),
               'raw_0_6_usable': 0 if all(not c['filled_mark_cells'] for c in cards) else 'manual_review',
               'rounded_id_unique': tables['4.csv']['beer_id'].nunique(),
               'unsafe_id_join_4_7_rows': badjoin,'workbooks': workbooks,
               'forum_all_missing': all_na_forum,'alignment': align,'matrices': matrices,
               'shap': shap,'defects': defect,'raw_negative_records':negative_raw,
               'transform_counts': dict(Counter(t['transform_status'] for t in transforms)),
               'finite_raw1_conflicts': sum(t.get('finite_mismatch_count',0) for t in transforms),
               'raw1_missingness_disagreements': sum(t.get('raw_missing_but_one_present',0)+t.get('raw_present_but_one_missing',0) for t in transforms),
               'common_value_minus_1_81781854': {'cells':int(floor.sum()),'fields':int(floor.any(axis=0).sum()),
                                                'note':'Repeated value only; unverified censoring/imputation provenance'},
               'panel_mean_range':[float(y.mean().min()),float(y.mean().max())],
               'panel_std_population_range':[float(y.std(ddof=0).min()),float(y.std(ddof=0).max())],
               'minimum_feature_values_examples': negative_features,
               'unresolved': ['226 versus actual 250: no 226-subset membership supplied',
                              'No raw 0-6 observations or invertible standardization parameters',
                              'Aroma/Flavour target scope ~40 versus 30 summary+2 card-only',
                              'Raw8 and input1 differ in some cells; provenance is unspecified',
                              'SHAP/GBM target and training protocol are unknown',
                              'No threshold for trans-2-nonenal or concentration threshold for stale_hops']}
    write_json(output/'summary.json', summary)
    for filename, rows in [('file_sheet_inventory.csv',inventories),('field_inventory.csv',field_rows),
                           ('sample_alignment.csv',sample_rows),('alignment_summary.csv',align),
                           ('rounded_id_collisions.csv',collisions),('matrix_checks.csv',matrices),
                           ('matrix_field_map.csv',matrix_map),('chemical_units_transforms.csv',transforms),
                           ('chemical_dictionary.csv',metadata),('sensory_dictionary.csv',sensory),
                           ('raw1_conflicts.csv',exceptions),('raw_negative_records.csv',negative_raw),
                           ('target_map_1_3_4_7.csv',mapping),('card_raw_label_checks.csv',raw_labels),
                           ('style_counts.csv',style_counts),('defect_thresholds.csv',defect),
                           ('defect_oav_by_beer.csv',oavs),('feature_duplicate_members.csv',duplicates)]:
        write_csv(output, filename, rows)
    write_json(output/'scorecards.json', cards)
    write_json(output/'shap_provenance.json', shap)
    (output/'findings.md').write_text(render_findings(summary, style_counts), encoding='utf-8')
    print(summary['status'], flush=True)
    return summary


def render_findings(s, styles):
    lines = ['# 2026 A 题可复现检查结果', '',
             '本文件由只读数据检查生成；不是预测模型的实验成绩。', '',
             '- 样本：%d；化学特征：%d；感官连续目标：%d。'%(s['rows'],s['chemical_features'],s['panel_targets']),
             '- 1.csv 缺失：%d；4.csv 感官缺失：%d。'%(s['chemical_missing_cells'],s['panel_missing_cells']),
             '- 3.xlsx 可用原始 0–6 标签：%s；空白表单的公式默认值不是真实零分。'%s['raw_0_6_usable'],
             '- 4/7 的 beer_id 唯一值仅 %d，不能据此把样本删成 96 条；错误关联会产生 %d 行。'%(s['rounded_id_unique'],s['unsafe_id_join_4_7_rows']),
             '- 请用 beer 一对一关联，并核对完整 ID 和风格；不依赖行顺序。',
             '- 原始8/输入1有限值冲突 %d 格，缺失状态不一致 %d 格；详见 raw1_conflicts.csv。'%(s['finite_raw1_conflicts'],s['raw1_missingness_disagreements']),
             '- SHAP concentration 块与 1.csv 对应：%s；GBM 目标/训练划分仍无法确认。'%s['shap']['concentration_equals_input1_including_missing'],
             '', '## 相关矩阵来源核验', '', '|文件|实际矩阵|Spearman 最大误差|Pearson 最大误差|', '|---|---|---|---|']
    for m in s['matrices']:
        lines.append('|%s|%d×%d|%.3g|%.3g|'%(m['file'],m['matrix_rows'],m['matrix_columns'],m['spearman_max_abs_error'],m['pearson_max_abs_error']))
    lines += ['', '## 风格数量', '', '|风格|数量|', '|---|---|']
    lines += ['|%s|%d|'%(r['style'],r['count']) for _,r in styles.iterrows()]
    lines += ['', '## 尚未解决', '']+['- '+v for v in s['unresolved']]
    lines += ['', '完整科研解释、候选方案和验证设计见随包 REPORT_2026A.md。',
              '本地环境通过不能替代目标运行环境的实际执行。', '']
    return '\n'.join(lines)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent.parent/'data'/'raw')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-python')
    parser.add_argument('--check-runtime-only', action='store_true')
    args = parser.parse_args()
    output, data = args.output_dir.resolve(), args.data_dir.resolve()
    protect_output(output, [data])
    result_zip = output.with_name(output.name+'.zip')
    if output.exists() or result_zip.exists():
        parser.error('输出目录或结果 ZIP 已存在；请选择新名称，拒绝覆盖: '+str(output))
    output.mkdir(parents=True)
    log = []
    class Tee:
        def __init__(self, stream): self.stream = stream
        def write(self, text): self.stream.write(text); log.append(text)
        def flush(self): self.stream.flush()
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(original_out), Tee(original_err)
    code = 0
    try:
        versions = {m:importlib.import_module(m).__version__ for m in ['numpy','pandas','openpyxl']}
        runtime = {'python':platform.python_version(),'executable':sys.executable,'libraries':versions,
                   'platform':platform.platform(),'expected_python':args.expected_python,
                   'data_dir':str(data),'output_dir':str(output)}
        write_json(output/'runtime.json',runtime)
        print(json.dumps(runtime, ensure_ascii=False, indent=2), flush=True)
        if args.expected_python and platform.python_version()!=args.expected_python:
            raise RuntimeError('Python 版本不符；拒绝继续，请使用项目虚拟环境')
        if args.check_runtime_only:
            print('2026A_RUNTIME_PASS')
        else:
            run_audit(data, output)
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout, sys.stderr = original_out, original_err
        (output/'run.log').write_text(''.join(log),encoding='utf-8')
        with zipfile.ZipFile(result_zip,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file(): archive.write(path,str(path.relative_to(output)))
        print('RESULT_ZIP='+str(result_zip),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
