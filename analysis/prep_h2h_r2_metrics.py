"""
post-hoc 指标补算（不是新实验）：
从已冻结的预注册对照 prep_h2h / prep_h2h_addendum 的 OOF 预测中，
补算 R^2（逐目标 + 宏平均）与 RMSE，并做逐目标配对 / 分层。

重要口径：
- 四臂（A_plain / A_clip / B_log / B_increment）× 8 模型共用
  同一 X(1.csv 231列)、同一 y(4.csv 50列)、同一 250 款样品、
  同一冻结 5 外折 outer.csv（seed 2601731）、同一固定超参、同一 OOF 拼接。
  验证方法完全对称，不存在「我们臂用严格验证、普通臂用简单划分」。
- inner 3 折在本次对照中不参与调参（超参冻结），以保证「只换预处理」是单因素；
  若需「每臂各自内层调参后比各自最优」，属于另一个新实验，另行预注册。
- R^2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)，逐目标计算后取 50 目标均值。
  同一 y 下 R^2 与 RMSE 严格同源，不是两个独立证据。
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.paths import TABLES, PREDICTIONS, ensure_output_dirs, configure_console  # noqa: E402

configure_console()
ensure_output_dirs()
z = np.load(PREDICTIONS / 'prep_h2h_addendum_oof.npz')
y = np.asarray(z['y'], dtype=float)
outer = np.asarray(z['outer'])
targets = [str(t) for t in z['targets']]
ARMS = ['A_plain', 'A_clip', 'B_log', 'B_increment']
MODELS = ['ridge', 'pls', 'krr', 'elastic', 'extratrees', 'randomforest', 'mlp_sklearn', 'mlp_torch']

# 感官模态映射（用于分层）
mod = pd.read_csv(TABLES / 'sensory_monitorability.csv')
tcol = 'target' if 'target' in mod.columns else mod.columns[0]
modmap = dict(zip(mod[tcol].astype(str), mod['modality'])) if 'modality' in mod.columns else {}

var = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
assert y.shape == (250, 50), y.shape
assert np.isfinite(y).all()

summary, by_target, oof_pred = [], [], {}
for arm in ARMS:
    for m in MODELS:
        key = f'{arm}__{m}'
        o = np.asarray(z[key], dtype=float)
        assert o.shape == y.shape and np.isfinite(o).all(), key
        oof_pred[key] = o
        se = (o - y) ** 2
        rmse = np.sqrt(se.mean(axis=0))
        r2 = 1 - se.sum(axis=0) / var
        rho = np.array([stats.spearmanr(o[:, j], y[:, j]).statistic for j in range(y.shape[1])])
        summary.append(dict(arm=arm, model=m, macro_r2=float(r2.mean()),
                            median_r2=float(np.median(r2)),
                            macro_rmse=float(rmse.mean()), macro_rho=float(np.nanmean(rho)),
                            n_targets=len(targets), n_samples=len(y),
                            n_folds=len(np.unique(outer)),
                            n_r2_positive=int((r2 > 0).sum())))
        for j, t in enumerate(targets):
            by_target.append(dict(arm=arm, model=m, target=t,
                                  modality=modmap.get(t, 'unknown'),
                                  r2=float(r2[j]), rmse=float(rmse[j]),
                                  rho=float(rho[j]), truth_sd=float(y[:, j].std())))

S = pd.DataFrame(summary)
T = pd.DataFrame(by_target)
S.to_csv(TABLES / 'prep_h2h_r2_summary.csv', index=False, encoding='utf-8-sig')
T.to_csv(TABLES / 'prep_h2h_r2_by_target.csv', index=False, encoding='utf-8-sig')


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


# 逐目标配对：平方误差差的 Wilcoxon（沿用上一轮口径），BH-FDR 在 (contrast, model) 内 50 目标校正
CONTRASTS = {
    'JIA_vs_no_protection': ('A_plain', 'B_log'),
    'YI_vs_no_protection': ('A_plain', 'B_increment'),
    'JIA_vs_fair_baseline': ('A_clip', 'B_log'),
    'YI_vs_fair_baseline': ('A_clip', 'B_increment'),
}
paired = []
for name, (a, b) in CONTRASTS.items():
    for m in MODELS:
        oa, ob = oof_pred[f'{a}__{m}'], oof_pred[f'{b}__{m}']
        d = (oa - y) ** 2 - (ob - y) ** 2  # >0 表示 B 更准
        ps = np.array([stats.wilcoxon(d[:, j], alternative='two-sided').pvalue
                       for j in range(y.shape[1])])
        qs = bh(ps)
        ta = T[(T.arm == a) & (T.model == m)].set_index('target')
        tb = T[(T.arm == b) & (T.model == m)].set_index('target')
        dr2 = (tb['r2'] - ta['r2']).reindex(targets).values
        drmse = (tb['rmse'] - ta['rmse']).reindex(targets).values
        for j, t in enumerate(targets):
            paired.append(dict(contrast=name, model=m, target=t,
                               modality=modmap.get(t, 'unknown'),
                               delta_r2=float(dr2[j]), delta_rmse=float(drmse[j]),
                               p_wilcoxon=float(ps[j]), q_bh=float(qs[j]),
                               direction='B_better' if dr2[j] > 0 else 'A_better',
                               sig_fdr05=bool(qs[j] < 0.05)))
Pa = pd.DataFrame(paired)
Pa.to_csv(TABLES / 'prep_h2h_r2_paired.csv', index=False, encoding='utf-8-sig')

# 分层汇总
strat = (Pa.groupby(['contrast', 'modality'])
         .agg(n_targets=('target', 'nunique'), mean_delta_r2=('delta_r2', 'mean'),
              mean_delta_rmse=('delta_rmse', 'mean'),
              n_B_better=('delta_r2', lambda s: int((s > 0).sum())),
              n_sig_B_better=('sig_fdr05', lambda s: int((s & (Pa.loc[s.index, 'direction'] == 'B_better')).sum())),
              n_sig_A_better=('sig_fdr05', lambda s: int((s & (Pa.loc[s.index, 'direction'] == 'A_better')).sum())))
         .reset_index())
strat.to_csv(TABLES / 'prep_h2h_r2_stratified.csv', index=False, encoding='utf-8-sig')

print('=== 验证对称性核实 ===')
print('y', y.shape, '| outer folds', sorted(np.unique(outer).tolist()),
      '| 每折样本', np.bincount(outer).tolist())
print('四臂 × 8 模型 OOF 全部 shape=(250,50) 且无 NaN: OK')
print('超参来源: prep_h2h_preregistration.json（固定，inner 3 折不参与调参）')
print()
print('=== 四臂 × 8 模型 宏 R² ===')
print(S.pivot(index='model', columns='arm', values='macro_r2').round(4).to_string())
print()
print('=== 四臂 × 8 模型 宏 RMSE ===')
print(S.pivot(index='model', columns='arm', values='macro_rmse').round(4).to_string())
print()
print('=== 8 模型平均 ===')
print(S.groupby('arm')[['macro_r2', 'macro_rmse', 'macro_rho', 'n_r2_positive']].mean().round(4).to_string())
print()
print('=== 同源性核查（同一 y，R² 与 RMSE 应为严格反向） ===')
print('Spearman(macro_r2, macro_rmse) =',
      round(float(stats.spearmanr(S.macro_r2, S.macro_rmse).statistic), 6))
print()
print('=== 相对「无防护普通预处理」与「公平基线」的宏 R² 差（8 模型平均） ===')
for name, (a, b) in CONTRASTS.items():
    d = Pa.groupby(['contrast', 'model'])['delta_r2'].mean()
    sub = d.loc[name]
    print('%-24s mean ΔR² = %+.4f | 模型数为正 %d/8 | 显著 B_better 目标(跨8模型) %d | 显著 A_better %d'
          % (name, sub.mean(), int((sub > 0).sum()),
             int(((Pa.contrast == name) & Pa.sig_fdr05 & (Pa.direction == 'B_better')).sum()),
             int(((Pa.contrast == name) & Pa.sig_fdr05 & (Pa.direction == 'A_better')).sum())))
print()
print('=== 按感官模态分层：ΔR²（相对公平基线 A_clip） ===')
for name in ['JIA_vs_fair_baseline', 'YI_vs_fair_baseline']:
    print('--', name)
    print(strat[strat.contrast == name][['modality', 'n_targets', 'mean_delta_r2',
                                         'n_B_better', 'n_sig_B_better', 'n_sig_A_better']]
          .round(4).to_string(index=False))
print()
print('输出: prep_h2h_r2_summary.csv / _by_target.csv / _paired.csv / _stratified.csv')
