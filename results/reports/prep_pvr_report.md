# 论文主结果 vs 常规预处理：同模型同划分对照

复现校验：论文 OOF 宏 R² = 0.260843，W_paper 重跑 = 0.260843，|gap| = 0.000000。

## 宏指标（5 外折 OOF，250 款，All50）

```
      arm scope  n_targets  macro_r2  macro_rmse  macro_mae  macro_rho  median_r2  positive_r2
PUBLISHED All50         50  0.260843    0.471020   0.336615   0.434554   0.210145           50
  W_paper All50         50  0.260843    0.471020   0.336615   0.434554   0.210145           50
R_control All50         50  0.254102    0.474167   0.338899   0.435505   0.212579           50
   R_fold All50         50  0.255304    0.473604   0.338627   0.438777   0.203954           50
 R_global All50         50  0.255078    0.473610   0.338548   0.437865   0.200456           50
```

## 三套目标范围

```
      arm  scope  n_targets  macro_r2  macro_rmse  macro_mae  macro_rho  median_r2  positive_r2
PUBLISHED  All50         50  0.260843    0.471020   0.336615   0.434554   0.210145           50
PUBLISHED Main37         37  0.282596    0.468672   0.346841   0.476926   0.254580           37
PUBLISHED   AF30         30  0.220501    0.473404   0.348283   0.429339   0.184931           30
  W_paper  All50         50  0.260843    0.471020   0.336615   0.434554   0.210145           50
  W_paper Main37         37  0.282596    0.468672   0.346841   0.476926   0.254580           37
  W_paper   AF30         30  0.220501    0.473404   0.348283   0.429339   0.184931           30
R_control  All50         50  0.254102    0.474167   0.338899   0.435505   0.212579           50
R_control Main37         37  0.274913    0.472345   0.349877   0.474505   0.262363           37
R_control   AF30         30  0.214929    0.476065   0.350627   0.428585   0.188647           30
   R_fold  All50         50  0.255304    0.473604   0.338627   0.438777   0.203954           50
   R_fold Main37         37  0.274880    0.472136   0.349439   0.479092   0.248538           37
   R_fold   AF30         30  0.214310    0.476183   0.350125   0.432707   0.184194           30
 R_global  All50         50  0.255078    0.473610   0.338548   0.437865   0.200456           50
 R_global Main37         37  0.275255    0.471949   0.349335   0.478558   0.251750           37
 R_global   AF30         30  0.214545    0.476080   0.350113   0.431940   0.186024           30
```

## 配对检验（逐目标 Wilcoxon + BH-FDR）

```
            contrast  scope  n_a_better  n_b_better  mean_rmse_a  mean_rmse_b  n_sig_q05_a_better  n_sig_q05_b_better
W_paper_vs_R_control  All50          32          18     0.471020     0.474167                 0.0                 0.0
W_paper_vs_R_control Main37          24          13     0.468672     0.472345                 0.0                 0.0
W_paper_vs_R_control   AF30          17          13     0.473404     0.476065                 0.0                 0.0
   W_paper_vs_R_fold  All50          36          14     0.471020     0.473604                 0.0                 0.0
   W_paper_vs_R_fold Main37          28           9     0.468672     0.472136                 0.0                 0.0
   W_paper_vs_R_fold   AF30          21           9     0.473404     0.476183                 0.0                 0.0
 W_paper_vs_R_global  All50          34          16     0.471020     0.473610                 2.0                 0.0
 W_paper_vs_R_global Main37          26          11     0.468672     0.471949                 2.0                 0.0
 W_paper_vs_R_global   AF30          20          10     0.473404     0.476080                 1.0                 0.0
 R_control_vs_R_fold  All50          21          29     0.474167     0.473604                 0.0                 0.0
 R_control_vs_R_fold Main37          18          19     0.472345     0.472136                 0.0                 0.0
 R_control_vs_R_fold   AF30          16          14     0.476065     0.476183                 0.0                 0.0
  R_fold_vs_R_global  All50          25          25     0.473604     0.473610                 0.0                 0.0
  R_fold_vs_R_global Main37          13          24     0.472136     0.471949                 0.0                 0.0
  R_fold_vs_R_global   AF30          12          18     0.476183     0.476080                 0.0                 0.0
```

## 折级方向

```
            contrast  scope  n_a_better  n_b_better  mean_rmse_a  mean_rmse_b  n_sig_q05_a_better  n_sig_q05_b_better
W_paper_vs_R_control fold_0          32          18     0.477314     0.482640                 NaN                 NaN
W_paper_vs_R_control fold_1          25          25     0.427688     0.429022                 NaN                 NaN
W_paper_vs_R_control fold_2          29          21     0.462723     0.464403                 NaN                 NaN
W_paper_vs_R_control fold_3          28          22     0.493082     0.493985                 NaN                 NaN
W_paper_vs_R_control fold_4          34          16     0.455598     0.460900                 NaN                 NaN
   W_paper_vs_R_fold fold_0          31          19     0.477314     0.482110                 NaN                 NaN
   W_paper_vs_R_fold fold_1          26          24     0.427688     0.426945                 NaN                 NaN
   W_paper_vs_R_fold fold_2          23          27     0.462723     0.463803                 NaN                 NaN
   W_paper_vs_R_fold fold_3          33          17     0.493082     0.495285                 NaN                 NaN
   W_paper_vs_R_fold fold_4          35          15     0.455598     0.460645                 NaN                 NaN
 W_paper_vs_R_global fold_0          32          18     0.477314     0.481971                 NaN                 NaN
 W_paper_vs_R_global fold_1          24          26     0.427688     0.426563                 NaN                 NaN
 W_paper_vs_R_global fold_2          24          26     0.462723     0.463807                 NaN                 NaN
 W_paper_vs_R_global fold_3          32          18     0.493082     0.495548                 NaN                 NaN
 W_paper_vs_R_global fold_4          37          13     0.455598     0.460808                 NaN                 NaN
 R_control_vs_R_fold fold_0          23          27     0.482640     0.482110                 NaN                 NaN
 R_control_vs_R_fold fold_1          15          35     0.429022     0.426945                 NaN                 NaN
 R_control_vs_R_fold fold_2          18          32     0.464403     0.463803                 NaN                 NaN
 R_control_vs_R_fold fold_3          31          19     0.493985     0.495285                 NaN                 NaN
 R_control_vs_R_fold fold_4          22          28     0.460900     0.460645                 NaN                 NaN
  R_fold_vs_R_global fold_0          20          30     0.482110     0.481971                 NaN                 NaN
  R_fold_vs_R_global fold_1          21          29     0.426945     0.426563                 NaN                 NaN
  R_fold_vs_R_global fold_2          25          25     0.463803     0.463807                 NaN                 NaN
  R_fold_vs_R_global fold_3          33          17     0.495285     0.495548                 NaN                 NaN
  R_fold_vs_R_global fold_4          30          20     0.460645     0.460808                 NaN                 NaN
```

## 按感官模态分层

```
            contrast modality  n  mean_sq_error_diff  n_a_better  n_sig_q05_a_better  n_sig_q05_b_better
W_paper_vs_R_control    Aroma 15           -0.002898           8                   0                   0
W_paper_vs_R_control  Overall  1           -0.004069           1                   0                   0
W_paper_vs_R_control   Palate  4           -0.007460           4                   0                   0
W_paper_vs_R_control    Taste 30           -0.002630          19                   0                   0
   W_paper_vs_R_fold    Aroma 15           -0.003108           9                   0                   0
   W_paper_vs_R_fold  Overall  1           -0.001835           1                   0                   0
   W_paper_vs_R_fold   Palate  4           -0.006442           4                   0                   0
   W_paper_vs_R_fold    Taste 30           -0.001956          22                   0                   0
 W_paper_vs_R_global    Aroma 15           -0.002937           8                   1                   0
 W_paper_vs_R_global  Overall  1           -0.003151           1                   0                   0
 W_paper_vs_R_global   Palate  4           -0.006327           4                   1                   0
 W_paper_vs_R_global    Taste 30           -0.002013          21                   0                   0
 R_control_vs_R_fold    Aroma 15           -0.000210           9                   0                   0
 R_control_vs_R_fold  Overall  1            0.002234           0                   0                   0
 R_control_vs_R_fold   Palate  4            0.001018           0                   0                   0
 R_control_vs_R_fold    Taste 30            0.000674          12                   0                   0
  R_fold_vs_R_global    Aroma 15            0.000171           6                   0                   0
  R_fold_vs_R_global  Overall  1           -0.001316           1                   0                   0
  R_fold_vs_R_global   Palate  4            0.000115           1                   0                   0
  R_fold_vs_R_global    Taste 30           -0.000057          17                   0                   0
```
