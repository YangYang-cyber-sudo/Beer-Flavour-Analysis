# -*- coding: utf-8 -*-
"""论文主结果 vs 常规预处理：同模型 / 同冻结划分 / 同指标的单因素对照。

只读预注册文件 prep_paper_vs_routine_preregistration.json，不在此脚本内挑臂、挑模型或删结果。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support.paths import (DATA, SPLITS, DERIVED, TABLES,  # noqa: E402
                           PREDICTIONS, REPORTS, PREREG as PREREG_DIR,
                           ensure_output_dirs, configure_console)
from modelling.data import load_data, load_splits, AF_FIELDS  # noqa: E402
from modelling.models import FoldTransform  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

configure_console()
ensure_output_dirs()
PERF = DERIVED / 'q1_final'
PREREG = json.loads((PREREG_DIR / 'prep_paper_vs_routine_preregistration.json').read_text(encoding='utf-8'))
ARMS = [a['id'] for a in PREREG['arms']]
SEED = 2602731
LOG = []


def say(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    LOG.append(line)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def frozen_iterations(d):
    """各外折冻结早停轮数，取自存档的 prep_pvr_meta.json。

    原始工作区把 0/1 折的轮数存在 ``analysis_2026a/q1_v3_runs/*.joblib``。那些
    开发运行目录没有随 S1 存档发布，但同一组数值已写入 ``results/tables/
    prep_pvr_meta.json``；这里读取该归档值，避免依赖已不存在的开发目录。
    """
    source = {}
    it = {}
    meta_path = TABLES / 'prep_pvr_meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        for key, value in meta.get('iterations', {}).items():
            it[int(key)] = int(value)
            source[int(key)] = 'prep_pvr_meta.json'
    fallback = {0: 1760, 1: 1737, 2: 1627, 3: 1182, 4: 1295}
    for fold in range(5):
        if fold not in it:
            it[fold] = fallback[fold]
            source[fold] = 'documented_fallback'
    say('冻结迭代数: ' + str({k: (it[k], source[k]) for k in range(5)}))
    return it, source


def transform(arm, X, Y, tr, te, d):
    """返回 (Z_train, Z_test)。R_global 故意在全量上拟合，其余臂只在训练折内拟合。"""
    if arm == 'R_global':
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isnan(X), med, X)
        center, scale = Xf.mean(axis=0), Xf.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        Z = (Xf - center) / scale
        return Z[tr], Z[te]
    cfg = next(a for a in PREREG['arms'] if a['id'] == arm)['pre']
    ft = FoldTransform(cfg, d['groups'], d['verified_log']).fit(X[tr], Y[tr])
    return ft.transform(X[tr]), ft.transform(X[te])


def fit_predict(Ztr, Ytr, Zte, iterations):
    mean = Ytr.mean(axis=0)
    scale = np.maximum(Ytr.std(axis=0), 1e-10)
    model = CatBoostRegressor(
        loss_function='MultiRMSE', eval_metric='MultiRMSE',
        iterations=int(iterations), depth=3, learning_rate=0.03, l2_leaf_reg=30,
        random_seed=SEED, thread_count=4, task_type='CPU',
        allow_writing_files=False, verbose=False,
    )
    model.fit(Ztr, (Ytr - mean) / scale)
    pred = np.asarray(model.predict(Zte))
    if pred.ndim == 1:
        pred = pred[:, None]
    out = pred * scale + mean
    if out.shape != (len(Zte), Ytr.shape[1]):
        raise ValueError(f'预测维数异常: {out.shape} != {(len(Zte), Ytr.shape[1])}')
    return out


def metric_frame(Y, P, targets):
    ss = ((Y - P) ** 2).sum(axis=0)
    st = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    rows = []
    for j, t in enumerate(targets):
        rows.append({
            'target': t,
            'r2': 1 - ss[j] / st[j],
            'rmse': np.sqrt(ss[j] / len(Y)),
            'mae': np.abs(Y - P).mean(axis=0)[j],
            'rho': spearmanr(Y[:, j], P[:, j]).statistic,
        })
    return pd.DataFrame(rows)


def main():
    d = load_data(DATA)
    outer, _ = load_splits(d['beers'], SPLITS)
    X, Y, targets = d['X'], d['Y'], list(d['targets'])
    say(f'数据 X{X.shape} Y{Y.shape}；外层折计数 {np.bincount(outer)}')

    iters, it_source = frozen_iterations(d)

    # 论文已发表 OOF，用于复现校验（存档于 data/derived/q1_final/）
    pub = pd.read_csv(DERIVED / 'q1_final' / 'OOF_depth3_l2_30.csv').set_index('beer')
    pub.index = pub.index.astype(str)
    pub = pub.loc[list(d['beers']), targets].to_numpy()

    oofs = {'PUBLISHED': pub}
    for arm in ARMS:
        t0 = time.perf_counter()
        pred = np.full_like(Y, np.nan)
        for f in range(5):
            tr, te = np.flatnonzero(outer != f), np.flatnonzero(outer == f)
            Ztr, Zte = transform(arm, X, Y, tr, te, d)
            pred[te] = fit_predict(Ztr, Y[tr], Zte, iters[f])
            say(f'  {arm} fold {f} done, Ztr {Ztr.shape}, iter {iters[f]}')
        if not np.isfinite(pred).all():
            raise ValueError(f'{arm} OOF 含非有限值')
        oofs[arm] = pred
        mf = metric_frame(Y, pred, targets)
        say(f'{arm} 完成 宏R2={mf.r2.mean():.6f} 宏RMSE={mf.rmse.mean():.6f} '
            f'宏rho={mf.rho.mean():.6f} 用时 {time.perf_counter()-t0:.1f}s')

    # 复现校验
    mf_pub = metric_frame(Y, pub, targets)
    mf_w = metric_frame(Y, oofs['W_paper'], targets)
    gap = abs(mf_pub.r2.mean() - mf_w.r2.mean())
    say(f'复现校验: 论文 OOF 宏R2={mf_pub.r2.mean():.6f}, W_paper 重跑={mf_w.r2.mean():.6f}, |gap|={gap:.6f}')
    if gap > 0.01:
        say('⚠ 偏差 > 0.01，报告中必须显式说明，不可把重跑值冒充论文值')

    # scope 清单
    per = pd.read_csv(PERF / 'Q1_FINAL_PER_TARGET.csv')
    per = per[(per.model == 'depth3_l2_30') & (per.fold == 'OOF5')].drop_duplicates('target')

    def flag(col):
        return per[col].astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

    main37 = set(per.loc[flag('main37'), 'target'])
    af30 = set(per.loc[flag('af30'), 'target'])
    if len(main37) != 37:
        say(f'⚠ main37 解析得到 {len(main37)} 个目标，检查清单来源')
    if len(af30) != 30:
        af30 = set(AF_FIELDS)
        say('af30 解析异常，回退到 q1.data.AF_FIELDS')
    scopes = {'All50': set(targets), 'Main37': main37 or set(), 'AF30': af30 or set(AF_FIELDS)}
    say('scope sizes: ' + str({k: len(v) for k, v in scopes.items()}))

    # 汇总
    summary, by_target = [], []
    for arm, P in oofs.items():
        mf = metric_frame(Y, P, targets)
        mf.insert(0, 'arm', arm)
        by_target.append(mf)
        for s, cols in scopes.items():
            sub = mf[mf.target.isin(cols)]
            if len(sub) == 0:
                continue
            summary.append({'arm': arm, 'scope': s, 'n_targets': len(sub),
                            'macro_r2': sub.r2.mean(), 'macro_rmse': sub.rmse.mean(),
                            'macro_mae': sub.mae.mean(), 'macro_rho': sub.rho.mean(),
                            'median_r2': sub.r2.median(), 'positive_r2': int((sub.r2 > 0).sum())})
    S = pd.DataFrame(summary)
    BT = pd.concat(by_target, ignore_index=True)
    S.to_csv(TABLES / 'prep_pvr_summary.csv', index=False, encoding='utf-8-sig')
    BT.to_csv(TABLES / 'prep_pvr_by_target.csv', index=False, encoding='utf-8-sig')

    # 主/次对比：a=W_paper, b=各常规臂；d = (Y-Pa)^2 - (Y-Pb)^2，d<0 表示 a 更优
    contrasts = [('W_paper', 'R_control'), ('W_paper', 'R_fold'),
                 ('W_paper', 'R_global'), ('R_control', 'R_fold'), ('R_fold', 'R_global')]
    paired_rows, consistency, strat = [], [], []
    try:
        mod = pd.read_csv(TABLES / 'sensory_monitorability.csv')
        t2m = dict(zip(mod.target, mod.modality))
    except Exception:
        t2m = {}

    for a, b in contrasts:
        Pa, Pb = oofs[a], oofs[b]
        dd = (Y - Pa) ** 2 - (Y - Pb) ** 2
        ps = []
        for j, t in enumerate(targets):
            try:
                p = wilcoxon(dd[:, j], alternative='two-sided').pvalue
            except Exception:
                p = 1.0
            ps.append(p)
        qs = bh_fdr(ps)
        for j, t in enumerate(targets):
            paired_rows.append({
                'contrast': f'{a}_vs_{b}', 'arm_a': a, 'arm_b': b, 'target': t,
                'mean_sq_error_diff': dd[:, j].mean(),
                'p_wilcoxon': ps[j], 'q_bh': qs[j],
                'direction': ('a_better' if dd[:, j].mean() < 0 else 'b_better'),
                'sig_q05_a_better': bool(qs[j] < .05 and dd[:, j].mean() < 0),
                'sig_q05_b_better': bool(qs[j] < .05 and dd[:, j].mean() > 0),
            })
        P = pd.DataFrame(paired_rows[-len(targets):])
        for s, cols in scopes.items():
            sub = P[P.target.isin(cols)]
            consistency.append({
                'contrast': f'{a}_vs_{b}', 'scope': s,
                'n_a_better': int((sub.mean_sq_error_diff < 0).sum()),
                'n_b_better': int((sub.mean_sq_error_diff > 0).sum()),
                'mean_rmse_a': np.sqrt(((Y - Pa) ** 2).mean(0))[[targets.index(t) for t in sub.target]].mean() if len(sub) else np.nan,
                'mean_rmse_b': np.sqrt(((Y - Pb) ** 2).mean(0))[[targets.index(t) for t in sub.target]].mean() if len(sub) else np.nan,
                'n_sig_q05_a_better': int(sub.sig_q05_a_better.sum()),
                'n_sig_q05_b_better': int(sub.sig_q05_b_better.sum()),
            })
        if t2m:
            P = P.copy()
            P['modality'] = P.target.map(t2m).fillna('Unknown')
            for m, g in P.groupby('modality'):
                strat.append({'contrast': f'{a}_vs_{b}', 'modality': m, 'n': len(g),
                              'mean_sq_error_diff': g.mean_sq_error_diff.mean(),
                              'n_a_better': int((g.mean_sq_error_diff < 0).sum()),
                              'n_sig_q05_a_better': int(g.sig_q05_a_better.sum()),
                              'n_sig_q05_b_better': int(g.sig_q05_b_better.sum())})
        # 折级方向一致性
        for f in range(5):
            te = np.flatnonzero(outer == f)
            r2a = 1 - ((Y[te] - Pa[te]) ** 2).sum(0) / ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
            r2b = 1 - ((Y[te] - Pb[te]) ** 2).sum(0) / ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
            consistency.append({'contrast': f'{a}_vs_{b}', 'scope': f'fold_{f}',
                                'n_a_better': int((r2a > r2b).sum()), 'n_b_better': int((r2a < r2b).sum()),
                                'mean_rmse_a': np.sqrt(((Y[te] - Pa[te]) ** 2).mean(0)).mean(),
                                'mean_rmse_b': np.sqrt(((Y[te] - Pb[te]) ** 2).mean(0)).mean(),
                                'n_sig_q05_a_better': np.nan, 'n_sig_q05_b_better': np.nan})

    PR = pd.DataFrame(paired_rows)
    C = pd.DataFrame(consistency)
    PR.to_csv(TABLES / 'prep_pvr_paired.csv', index=False, encoding='utf-8-sig')
    C.to_csv(TABLES / 'prep_pvr_consistency.csv', index=False, encoding='utf-8-sig')
    if strat:
        pd.DataFrame(strat).to_csv(TABLES / 'prep_pvr_stratified.csv', index=False, encoding='utf-8-sig')

    np.savez(PREDICTIONS / 'prep_pvr_oof.npz', y=Y, outer=outer, targets=np.array(targets),
             **{k: v for k, v in oofs.items()})
    (TABLES / 'prep_pvr_meta.json').write_text(json.dumps({
        'iterations': {str(k): int(v) for k, v in iters.items()},
        'iteration_source': {str(k): v for k, v in it_source.items()},
        'reproduction_gap_macro_r2': float(gap),
        'published_macro_r2': float(mf_pub.r2.mean()),
        'rerun_W_macro_r2': float(mf_w.r2.mean()),
        'seed': SEED,
        'arms': ARMS,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    # 报告（纯文本表格，避免额外绘图/制表依赖）
    def tbl(df):
        return '```\n' + df.round(6).to_string(index=False) + '\n```'

    lines = ['# 论文主结果 vs 常规预处理：同模型同划分对照', '',
             f'复现校验：论文 OOF 宏 R² = {mf_pub.r2.mean():.6f}，W_paper 重跑 = {mf_w.r2.mean():.6f}，|gap| = {gap:.6f}。', '',
             '## 宏指标（5 外折 OOF，250 款，All50）', '', tbl(S[S.scope == 'All50']), '',
             '## 三套目标范围', '', tbl(S), '',
             '## 配对检验（逐目标 Wilcoxon + BH-FDR）', '',
             tbl(C[C.scope.isin(['All50', 'Main37', 'AF30'])]), '',
             '## 折级方向', '', tbl(C[C.scope.str.startswith('fold_')]), '']
    if strat:
        lines += ['## 按感官模态分层', '', tbl(pd.DataFrame(strat)), '']
    (REPORTS / 'prep_pvr_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (REPORTS / 'prep_pvr_run.log').write_text('\n'.join(LOG), encoding='utf-8')
    say('全部完成。')


if __name__ == '__main__':
    main()
