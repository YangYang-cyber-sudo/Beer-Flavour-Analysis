"""预处理三臂 head-to-head 对照（只读预注册协议，不做任何事后挑选）。

同一 X / y / 5 折外层划分 / 模型池 / 超参，唯一变化是 FoldTransform 的配置。
输出全部 arm x model x fold 的 OOF 预测与配对统计，供 JFS v4' §3.3 使用。
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support.paths import (DATA, SPLITS, TABLES, PREDICTIONS,  # noqa: E402
                           PREREG, ensure_output_dirs, configure_console)
from modelling.data import load_data, load_splits                # noqa: E402
from modelling.models import FoldTransform                      # noqa: E402
from modelling.increment import IncrementTransform              # noqa: E402

configure_console()
SEED = 2601731
warnings.filterwarnings("ignore")
ensure_output_dirs()

PRE = json.loads((PREREG / "prep_h2h_preregistration.json").read_text(encoding="utf-8"))


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------------------- models
class TorchMLP:
    """固定 seed、全批量、无早停、无验证集偷看的小型 MLP。"""

    def __init__(self, hidden, epochs, lr, weight_decay, n_in, n_out, seed):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.seed, self.epochs, self.lr, self.wd = seed, epochs, lr, weight_decay
        layers, prev = [], n_in
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers += [torch.nn.Linear(prev, n_out)]
        self.net = torch.nn.Sequential(*layers)

    def fit(self, z, t):
        import torch
        x = torch.tensor(np.asarray(z, dtype=np.float32))
        y = torch.tensor(np.asarray(t, dtype=np.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            ((self.net(x) - y) ** 2).mean().backward()
            opt.step()
        return self

    def predict(self, z):
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(np.asarray(z, dtype=np.float32))).numpy()


def build_model(family, params, z_train, n_targets):
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor

    p = dict(params)
    if family == "ridge":
        return Ridge(solver="svd", **p)
    if family == "pls":
        n = min(int(p["n_components"]), z_train.shape[0] - 1, z_train.shape[1])
        return PLSRegression(n_components=n, scale=False, max_iter=1000)
    if family == "krr":
        gf = p.pop("gamma_factor")
        sq = (z_train * z_train).sum(axis=1)
        dist = np.maximum(sq[:, None] + sq[None, :] - 2 * z_train.dot(z_train.T), 0)
        pairs = dist[np.triu_indices(len(dist), 1)]
        bw = np.median(pairs[pairs > 1e-12]) if (pairs > 1e-12).any() else 1.0
        return KernelRidge(kernel="rbf", gamma=gf / bw, **p)
    if family == "elastic":
        return MultiOutputRegressor(
            ElasticNet(max_iter=1000, tol=1e-4, random_state=SEED, **p), n_jobs=1)
    if family == "extratrees":
        return ExtraTreesRegressor(random_state=SEED, n_jobs=1, **p)
    if family == "randomforest":
        return RandomForestRegressor(random_state=SEED, n_jobs=1, **p)
    if family == "mlp_sklearn":
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPRegressor(random_state=SEED, early_stopping=False,
                            learning_rate_init=0.001, **p)
    if family == "mlp_torch":
        hidden, epochs = p["hidden"], p["epochs"]
        return TorchMLP(hidden=hidden, epochs=epochs, lr=p["lr"],
                        weight_decay=p["weight_decay"], n_in=z_train.shape[1],
                        n_out=n_targets, seed=SEED)
    raise ValueError("未注册的模型家族: " + family)


# ---------------------------------------------------------------------------- core
def oof_predict(cfg, family, params, X, y, outer, groups, logs, identity):
    """5 折 OOF；所有数据依赖的变换与统计量只在训练折内拟合。"""
    oof = np.full(y.shape, np.nan, dtype=float)
    for f in range(5):
        tr, te = np.flatnonzero(outer != f), np.flatnonzero(outer == f)
        tf = IncrementTransform(dict(cfg), groups, logs, identity).fit(X[tr], y[tr])
        ztr, zte = tf.transform(X[tr]), tf.transform(X[te])
        mean = y[tr].mean(axis=0)
        sd = y[tr].std(axis=0)
        sd = np.where(sd > 1e-10, sd, 1.0)
        model = build_model(family, params, ztr, y.shape[1])
        model.fit(ztr, (y[tr] - mean) / sd)
        pred = np.asarray(model.predict(zte), dtype=float)
        if pred.ndim == 1:
            pred = pred[:, None]
        oof[te] = pred * sd + mean
        if not np.isfinite(oof[te]).all():
            raise ValueError(f"非有限预测: {family} fold {f}")
    return oof


def metrics(oof, y):
    rmse = np.sqrt(np.nanmean((oof - y) ** 2, axis=0))
    mae = np.nanmean(np.abs(oof - y), axis=0)
    rho = np.array([stats.spearmanr(oof[:, j], y[:, j]).statistic
                    for j in range(y.shape[1])])
    return {"rmse": rmse, "mae": mae, "rho": rho,
            "macro_rmse": float(rmse.mean()), "macro_mae": float(mae.mean()),
            "macro_rho": float(np.nanmean(rho))}


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.empty(n)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(q, 0, 1)


def paired(truth, oof_a, oof_b, targets):
    """逐目标：250 个 OOF 样本的平方误差差，双侧 Wilcoxon + BH-FDR。"""
    da = (oof_a - truth) ** 2
    db = (oof_b - truth) ** 2
    diff = da - db                      # >0 表示 B 更准
    rows = []
    for j, name in enumerate(targets):
        d = diff[:, j]
        try:
            p = float(stats.wilcoxon(d, alternative="two-sided",
                                     zero_method="wilcox").pvalue)
        except ValueError:
            p = 1.0
        rows.append({"target": name, "delta_rmse": None, "mean_sq_error_diff": float(d.mean()),
                     "median_sq_error_diff": float(np.median(d)), "p_wilcoxon": p})
    qs = bh_fdr([r["p_wilcoxon"] for r in rows])
    for r, q in zip(rows, qs):
        r["q_bh"] = float(q)
        r["direction"] = ("B_better" if r["mean_sq_error_diff"] > 0 else
                          "A_better" if r["mean_sq_error_diff"] < 0 else "tie")
        r["significant_fdr05"] = bool(q < 0.05)
    return rows


def main():
    started = datetime.now()
    log("START", started.isoformat(timespec="seconds"))
    d = load_data(DATA)
    X, y = d["X"], d["Y"]
    targets = list(d["targets"])
    groups, logs = d["groups"], d["verified_log"]
    identity = d["metadata"]["transform_status"].eq(
        "mostly_identity_with_exceptions").to_numpy()
    outer, _ = load_splits(d["beers"], SPLITS)
    log("DATA", X.shape, y.shape, "verified_log", int(logs.sum()),
        "verified_identity", int(identity.sum()), "outer folds", np.bincount(outer))

    # 代码路径一致性断言：A_plain 必须与旧 FoldTransform(plain) 逐位相同
    cfg_a = PRE["arms"]["A_plain"]["config"]
    tr = np.flatnonzero(outer != 0)
    te = np.flatnonzero(outer == 0)
    inc = IncrementTransform(dict(cfg_a), groups, logs, identity).fit(X[tr], y[tr])
    old = FoldTransform(dict(cfg_a), groups, logs).fit(X[tr], y[tr])
    np.testing.assert_allclose(inc.transform(X[te]), old.transform(X[te]),
                               rtol=0, atol=1e-12)
    log("ASSERT A_plain == legacy FoldTransform(plain): PASS")

    arms = {k: v["config"] for k, v in PRE["arms"].items()}
    models = PRE["models"]
    oofs, summary_rows, per_target_rows = {}, [], []

    for arm, cfg in arms.items():
        for m in models:
            key = f"{arm}__{m['id']}"
            log("RUN", key)
            oof = oof_predict(cfg, m["family"], m["params"], X, y, outer,
                              groups, logs, identity)
            oofs[key] = oof
            mt = metrics(oof, y)
            summary_rows.append({"arm": arm, "model": m["id"], "tier": m["tier"],
                                 "macro_rmse": mt["macro_rmse"],
                                 "macro_mae": mt["macro_mae"],
                                 "macro_rho": mt["macro_rho"]})
            for j, name in enumerate(targets):
                per_target_rows.append({"arm": arm, "model": m["id"], "target": name,
                                        "rmse": float(mt["rmse"][j]),
                                        "mae": float(mt["mae"][j]),
                                        "rho": float(mt["rho"][j])})
            log("    macro_rmse=%.6f macro_rho=%.4f" % (mt["macro_rmse"], mt["macro_rho"]))

    # ladder：逐项加回，Ridge 单模型
    ladder_rows = []
    for step, cfg in PRE["ladder"]["steps"].items():
        log("LADDER", step)
        oof = oof_predict(cfg, "ridge", {"alpha": 100}, X, y, outer,
                          groups, logs, identity)
        mt = metrics(oof, y)
        ladder_rows.append({"step": step, "shape": cfg["shape"], "clip": cfg["clip"],
                            "scale": cfg["scale"], "special": cfg["special"],
                            "macro_rmse": mt["macro_rmse"], "macro_rho": mt["macro_rho"]})

    # 配对统计
    contrasts = [("B_log", "A_plain", "JIA_vs_routine"),
                 ("B_increment", "A_plain", "YI_vs_routine"),
                 ("B_increment", "B_log", "YI_vs_JIA")]
    paired_rows, consistency_rows = [], []
    for b, a, label in contrasts:
        for m in models:
            key_a, key_b = f"{a}__{m['id']}", f"{b}__{m['id']}"
            rows = paired(y, oofs[key_a], oofs[key_b], targets)
            ma = metrics(oofs[key_a], y)
            mb = metrics(oofs[key_b], y)
            for r in rows:
                r.update({"contrast": label, "model": m["id"], "tier": m["tier"],
                          "rmse_a": float(ma["rmse"][targets.index(r["target"])]),
                          "rmse_b": float(mb["rmse"][targets.index(r["target"])])})
                r["delta_rmse"] = r["rmse_b"] - r["rmse_a"]
                paired_rows.append(r)
            improved = int((mb["rmse"] < ma["rmse"]).sum())
            regressed = int((mb["rmse"] > ma["rmse"]).sum())
            sig_better = sum(1 for r in rows
                             if r["significant_fdr05"] and r["direction"] == "B_better")
            sig_worse = sum(1 for r in rows
                            if r["significant_fdr05"] and r["direction"] == "A_better")
            consistency_rows.append({
                "contrast": label, "model": m["id"], "tier": m["tier"],
                "macro_rmse_a": ma["macro_rmse"], "macro_rmse_b": mb["macro_rmse"],
                "delta_macro_rmse": mb["macro_rmse"] - ma["macro_rmse"],
                "delta_macro_rho": mb["macro_rho"] - ma["macro_rho"],
                "n_targets_improved": improved, "n_targets_regressed": regressed,
                "n_sig_fdr05_B_better": sig_better, "n_sig_fdr05_A_better": sig_worse,
                "overall_better": bool(mb["macro_rmse"] < ma["macro_rmse"])})

    # 按 modality 分层（沿用既有 sensory_monitorability.csv 口径）
    modality = {}
    with (TABLES / "sensory_monitorability.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            modality[row["target"]] = row["modality"]
    stratified = []
    contrast_arms = {"JIA_vs_routine": ("A_plain", "B_log"),
                     "YI_vs_routine": ("A_plain", "B_increment"),
                     "YI_vs_JIA": ("B_log", "B_increment")}
    for c in consistency_rows:
        arm_a, arm_b = contrast_arms[c["contrast"]]
        ma, mb = metrics(oofs[f"{arm_a}__{c['model']}"], y), metrics(oofs[f"{arm_b}__{c['model']}"], y)
        for mod in sorted(set(modality.values())):
            idx = [j for j, t in enumerate(targets) if modality.get(t) == mod]
            if idx:
                stratified.append({"contrast": c["contrast"], "model": c["model"],
                                   "modality": mod, "n": len(idx),
                                   "mean_rmse_a": float(ma["rmse"][idx].mean()),
                                   "mean_rmse_b": float(mb["rmse"][idx].mean()),
                                   "delta": float(mb["rmse"][idx].mean() - ma["rmse"][idx].mean())})

    # ------------------------------------------------------------------ write
    def write_csv(name, rows):
        if not rows:
            return
        with (TABLES / name).open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv("prep_h2h_summary.csv", summary_rows)
    write_csv("prep_h2h_per_target.csv", per_target_rows)
    write_csv("prep_h2h_paired.csv", paired_rows)
    write_csv("prep_h2h_model_consistency.csv", consistency_rows)
    write_csv("prep_h2h_ladder.csv", ladder_rows)
    write_csv("prep_h2h_stratified.csv", stratified)
    np.savez_compressed(PREDICTIONS / "prep_h2h_oof.npz", y=y, outer=outer,
                        targets=np.array(targets), **oofs)

    tier_summary = {}
    for c in consistency_rows:
        tier_summary.setdefault((c["contrast"], c["tier"]), []).append(c)
    tiers = [{"contrast": k[0], "tier": k[1],
              "n_models": len(v),
              "n_models_overall_better": sum(1 for x in v if x["overall_better"]),
              "mean_delta_macro_rmse": float(np.mean([x["delta_macro_rmse"] for x in v]))}
             for k, v in tier_summary.items()]

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preregistration": "prep_h2h_preregistration.json",
        "n_arms": len(arms), "n_models": len(models), "n_folds": 5,
        "n_targets": y.shape[1], "n_samples": X.shape[0],
        "summary": summary_rows,
        "consistency": consistency_rows,
        "tier_summary": tiers,
        "ladder": ladder_rows,
        "stratified": stratified,
        "cross_model_count": {
            label: {"n_models": len(models),
                    "n_overall_better": sum(1 for c in consistency_rows
                                            if c["contrast"] == label and c["overall_better"]),
                    "n_sig_targets_B_better": sum(c["n_sig_fdr05_B_better"] for c in consistency_rows
                                                  if c["contrast"] == label),
                    "n_sig_targets_A_better": sum(c["n_sig_fdr05_A_better"] for c in consistency_rows
                                                  if c["contrast"] == label)}
            for _, _, label in contrasts},
    }
    (TABLES / "prep_h2h_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log("DONE", datetime.now().isoformat(timespec="seconds"),
        "elapsed", str(datetime.now() - started))
    log("CROSS_MODEL", json.dumps(results["cross_model_count"], ensure_ascii=False))
    for t in tiers:
        log("TIER", t)


if __name__ == "__main__":
    main()
