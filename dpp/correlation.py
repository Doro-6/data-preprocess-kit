# -*- coding: utf-8 -*-
"""相关性分析：相关强度、显著性、关联度量与共线性诊断。

同时给出 Pearson / Spearman / Kendall 三种口径，因为它们对线性、单调、
秩次关系的敏感度不同；所有 p 值再做 Benjamini-Hochberg 校正以控制假发现率。
"""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 参与相关性分析的数值编码变量（分类变量先做有序/二值编码）
CORR_VARS = [
    "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "HasCabin", "AgeImputed",
]
VAR_LABEL_CN = {
    "Survived": "生还", "Pclass": "舱位等级", "Sex": "性别(女=0)", "Age": "年龄",
    "SibSp": "同乘兄弟姐妹/配偶", "Parch": "同乘父母/子女", "Fare": "票价",
    "FamilySize": "家庭规模", "IsAlone": "是否独行", "HasCabin": "有无客舱号",
    "AgeImputed": "年龄是否被填补",
}


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """把分析用变量统一成数值编码。"""
    out = pd.DataFrame(index=df.index)
    for col in CORR_VARS:
        if col not in df.columns:
            continue
        if col == "Sex":
            out[col] = (df[col] == "female").astype(int)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。"""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals, kind="stable")
    ranked = pvals[order]
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        q[i] = prev
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def numeric_correlation(clean: pd.DataFrame, transformed: pd.DataFrame | None = None) -> pd.DataFrame:
    """三种相关系数矩阵（以生还为基准排序，便于阅读）。"""
    data = _coerce(clean)
    rows = []
    for col in data.columns:
        if col == "Survived":
            continue
        a, b = data["Survived"], data[col]
        r, p_r = stats.pearsonr(a, b)
        rho, p_s = stats.spearmanr(a, b)
        tau, p_k = stats.kendalltau(a, b)
        n = int(b.notna().sum())
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3) if n > 3 else np.nan
        rows.append({
            "变量": col,
            "中文名": VAR_LABEL_CN.get(col, col),
            "Pearson r": round(float(r), 4),
            "Pearson 95%CI 下限": round(float(np.tanh(z - 1.96 * se)), 4) if n > 3 else None,
            "Pearson 95%CI 上限": round(float(np.tanh(z + 1.96 * se)), 4) if n > 3 else None,
            "Pearson r²(解释方差)": round(float(r ** 2), 4),
            "Pearson p": float(p_r),
            "Spearman ρ": round(float(rho), 4),
            "Spearman p": float(p_s),
            "Kendall τ": round(float(tau), 4),
            "Kendall p": float(p_k),
            "样本量": n,
        })
    out = pd.DataFrame(rows)
    out["q值(BH)"] = _bh(out["Pearson p"].to_numpy())
    out["显著(q<0.05)"] = np.where(out["q值(BH)"] < 0.05, "是", "否")
    out["|r| 强度"] = pd.cut(out["Pearson r"].abs(), bins=[-0.001, 0.1, 0.3, 0.5, 1.0],
                             labels=["弱(<0.1)", "弱-中(0.1-0.3)", "中(0.3-0.5)", "强(>0.5)"])
    return out.sort_values("Pearson r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def pairwise_table(clean: pd.DataFrame, vars_: list[str] | None = None) -> pd.DataFrame:
    """变量两两相关表：三种系数 + p 值 + BH 校正 + 置信区间。"""
    data = _coerce(clean)
    cols = [c for c in (vars_ or CORR_VARS) if c in data.columns]
    rows = []
    for a, b in combinations(cols, 2):
        pair = data[[a, b]].dropna()
        n = len(pair)
        if n < 10 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
            continue
        r, p_r = stats.pearsonr(pair[a], pair[b])
        rho, p_s = stats.spearmanr(pair[a], pair[b])
        tau, p_k = stats.kendalltau(pair[a], pair[b])
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        se = 1 / np.sqrt(n - 3)
        kind = "数值-数值"
        if pair[a].nunique() == 2 or pair[b].nunique() == 2:
            kind = "二值-数值"
        rows.append({
            "变量A": a,
            "变量B": b,
            "关系类型": kind,
            "样本量": n,
            "Pearson r": round(float(r), 4),
            "95%CI 下限": round(float(np.tanh(z - 1.96 * se)), 4),
            "95%CI 上限": round(float(np.tanh(z + 1.96 * se)), 4),
            "Spearman ρ": round(float(rho), 4),
            "Kendall τ": round(float(tau), 4),
            "p 值(Pearson)": float(p_r),
            "p 值(Spearman)": float(p_s),
            "|r|": round(abs(float(r)), 4),
            "方向": "正相关" if r > 0 else "负相关",
        })
    out = pd.DataFrame(rows)
    out["q值(BH)"] = _bh(out["p 值(Pearson)"].to_numpy())
    out["显著(q<0.05)"] = np.where(out["q值(BH)"] < 0.05, "是", "否")
    out["显著性标记"] = np.where(out["q值(BH)"] < 0.001, "***",
                                 np.where(out["q值(BH)"] < 0.01, "**",
                                          np.where(out["q值(BH)"] < 0.05, "*", "ns")))
    return out.sort_values("|r|", ascending=False).reset_index(drop=True)


def corr_matrix_pairwise(clean: pd.DataFrame, method: str = "pearson",
                         vars_: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """相关系数矩阵与对应 p 值矩阵。"""
    data = _coerce(clean)
    cols = [c for c in (vars_ or CORR_VARS) if c in data.columns]
    sub = data[cols].dropna()
    r = sub.corr(method=method)
    p = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    for a, b in combinations(cols, 2):
        if sub[a].nunique() < 2 or sub[b].nunique() < 2:
            p.loc[a, b] = p.loc[b, a] = np.nan
            continue
        stat = fn(sub[a], sub[b])[1]
        p.loc[a, b] = p.loc[b, a] = float(stat)
    return r, p


def vif_table(transformed: pd.DataFrame, max_cols: int = 14) -> pd.DataFrame:
    """方差膨胀因子：检验转换后特征之间的多重共线性。"""
    drop = {"PassengerId", "Survived"}
    cols = [c for c in transformed.columns if c not in drop]
    X = transformed[cols].astype(float)
    # 只保留方差非零的列
    X = X.loc[:, X.std(ddof=0) > 0]
    if X.shape[1] > max_cols:
        # 优先保留相关性最高的核心特征
        priority = ["Age_standardized", "Fare_log", "Pclass_ordinal", "Sex_female", "FamilySize",
                    "IsAlone", "HasCabin", "Embarked_S", "Title_Mr", "Deck_Unknown",
                    "AgeImputed", "SibSp", "Parch", "FareBin_ordinal"]
        keep = [c for c in priority if c in X.columns][:max_cols]
        X = X[keep]
    corr = X.corr().to_numpy()
    try:
        vif = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(len(X.columns), np.nan)
    rows = []
    for col, v in zip(X.columns, vif):
        rows.append({
            "特征": col,
            "VIF": round(float(v), 4),
            "容忍度 1/VIF": round(float(1 / v), 4) if v else None,
            "共线性判断": ("严重（VIF>10）" if v > 10 else "中等（5<VIF≤10）" if v > 5
                           else "轻微（VIF≤5）"),
        })
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def association_measures(clean: pd.DataFrame) -> pd.DataFrame:
    """分类型变量之间的关联度量：卡方、Cramér's V、Theil's U。"""
    cols = [c for c in ["Survived", "Pclass", "Sex", "Embarked", "Title", "Deck",
                        "HasCabin", "IsAlone"] if c in clean.columns]
    rows = []
    for a, b in combinations(cols, 2):
        tab = pd.crosstab(clean[a], clean[b])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        n = tab.to_numpy().sum()
        v = float(np.sqrt((chi2 / n) / (min(tab.shape) - 1)))
        u_a_b = _theils_u(clean[a], clean[b])
        u_b_a = _theils_u(clean[b], clean[a])
        rows.append({
            "变量A": a,
            "变量B": b,
            "卡方": round(float(chi2), 4),
            "自由度": int(dof),
            "p 值": float(p),
            "Cramér's V": round(v, 4),
            "关联强度": "强" if v > 0.5 else "中等" if v > 0.3 else "弱" if v > 0.1 else "几乎无",
            "Theil's U (A -> B)": round(u_a_b, 4),
            "Theil's U (B -> A)": round(u_b_a, 4),
            "方向性": "近似对称" if abs(u_a_b - u_b_a) < 0.05 else
                      (f"{a} 更能决定 {b}" if u_a_b > u_b_a else f"{b} 更能决定 {a}"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q值(BH)"] = _bh(out["p 值"].to_numpy())
        out["显著(q<0.05)"] = np.where(out["q值(BH)"] < 0.05, "是", "否")
        out = out.sort_values("Cramér's V", ascending=False).reset_index(drop=True)
    return out


def _theils_u(x: pd.Series, y: pd.Series) -> float:
    """Theil's U：已知 y 后对 x 的不确定性下降比例。"""
    x = x.astype(str)
    y = y.astype(str)
    h_x = stats.entropy(x.value_counts(normalize=True))
    if h_x == 0:
        return 0.0
    cond = 0.0
    for val, grp in x.groupby(y):
        w = len(grp) / len(x)
        cond += w * stats.entropy(grp.value_counts(normalize=True))
    return float((h_x - cond) / h_x)


def correlation_facts(pairwise: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """抽取报告需要引用的关键相关结论。"""
    strong = pairwise[pairwise["|r|"] >= 0.3].head(10)
    sig = pairwise[pairwise["显著(q<0.05)"] == "是"]
    with_target = pairwise[(pairwise["变量A"] == "Survived") | (pairwise["变量B"] == "Survived")]
    return {
        "total_pairs": int(len(pairwise)),
        "significant_pairs": int(len(sig)),
        "significant_ratio": round(float(len(sig) / max(len(pairwise), 1)), 4),
        "strong_pairs": [
            {"pair": f"{r['变量A']}  vs  {r['变量B']}", "r": r["Pearson r"],
             "rho": r["Spearman ρ"], "q": round(float(r["q值(BH)"]), 6)}
            for _, r in strong.iterrows()
        ],
        "target_correlations": [
            {"var": (r["变量A"] if r["变量B"] == "Survived" else r["变量B"]),
             "r": r["Pearson r"], "rho": r["Spearman ρ"],
             "q": round(float(r["q值(BH)"]), 6), "mark": r["显著性标记"]}
            for _, r in with_target.sort_values("|r|", ascending=False).iterrows()
        ],
        "method_agreement": {
            "pearson_vs_spearman_max_diff": round(float(
                (pairwise["Pearson r"] - pairwise["Spearman ρ"]).abs().max()), 4),
            "pearson_vs_kendall_max_diff": round(float(
                (pairwise["Pearson r"] - pairwise["Kendall τ"]).abs().max()), 4),
        },
    }


# ==========================================================================
# 插图
# ==========================================================================
def _annotate(ax, mat: pd.DataFrame, pmat: pd.DataFrame, fontsize: float = 5.2) -> None:
    for i, a in enumerate(mat.index):
        for j, b in enumerate(mat.columns):
            if i == j:
                continue
            val = mat.iloc[i, j]
            p = pmat.iloc[i, j]
            star = "" if pd.isna(p) else ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center", fontsize=fontsize,
                    color="white" if abs(val) > 0.55 else "black")


def fig_correlation_heatmaps(clean: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6))
    labels = [VAR_LABEL_CN.get(v, v) for v in CORR_VARS]
    for ax, (method, title) in zip(axes, [("pearson", "Pearson 线性相关"),
                                          ("spearman", "Spearman 秩相关"),
                                          ("kendall", "Kendall τ 秩相关")]):
        r, p = corr_matrix_pairwise(clean, method=method)
        im = ax.imshow(r.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(r)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(r)), labels, fontsize=6)
        ax.set_title(f"({'abc'[['pearson', 'spearman', 'kendall'].index(method)]}) {title}"
                     f"（* p<0.05, ** p<0.01, *** p<0.001）", fontsize=8.6)
        _annotate(ax, r, p)
        ax.set_xticks(np.arange(-0.5, len(r), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(r), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.5)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="相关系数")
    return fig


def fig_significant_correlations(pairwise: pd.DataFrame, top: int = 16):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    sub = pairwise.head(top).iloc[::-1]
    labels = [f"{a}  vs  {b}" for a, b in zip(sub["变量A"], sub["变量B"])]

    ax = axes[0]
    colors = ["#C44E52" if r < 0 else "#4C72B0" for r in sub["Pearson r"]]
    ax.barh(labels, sub["Pearson r"], color=colors, height=0.62)
    for i, (r, mark) in enumerate(zip(sub["Pearson r"], sub["显著性标记"])):
        ax.text(r + (0.012 if r > 0 else -0.012), i, f"{r:.3f}{mark}",
                va="center", ha="left" if r > 0 else "right", fontsize=6.4)
    ax.axvline(0, color="black", lw=0.8)
    for x in (-0.3, 0.3):
        ax.axvline(x, color="gray", ls=":", lw=0.8)
    ax.set_xlim(-0.95, 0.95)
    ax.set_xlabel("Pearson 相关系数 r")
    ax.set_title(f"(a) |r| 最大的 {len(sub)} 对变量", fontsize=9)
    ax.tick_params(axis="y", labelsize=6.6)

    ax = axes[1]
    ax.scatter(pairwise["Pearson r"], pairwise["Spearman ρ"], s=16, alpha=0.7, color="#55A868")
    lim = [-1, 1]
    ax.plot(lim, lim, "k--", lw=0.9, label="两法完全一致")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Spearman ρ")
    diff = (pairwise["Pearson r"] - pairwise["Spearman ρ"]).abs()
    ax.set_title(f"(b) 线性相关 vs 秩相关：最大差异 {diff.max():.3f}\n"
                 f"（点偏离对角线说明关系非线性或存在离群值）", fontsize=9)
    idx = diff.idxmax()
    ax.annotate(f"{pairwise.loc[idx, '变量A']}  vs  {pairwise.loc[idx, '变量B']}",
                xy=(pairwise.loc[idx, "Pearson r"], pairwise.loc[idx, "Spearman ρ"]),
                xytext=(pairwise.loc[idx, "Pearson r"] - 0.55, pairwise.loc[idx, "Spearman ρ"] + 0.12),
                fontsize=6.4, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_correlation_scatter(clean: pd.DataFrame):
    """关键变量对的散点矩阵，直观展示相关形态（线性/非线性/异方差）。"""
    pairs = [("Fare", "Pclass"), ("Age", "Pclass"), ("Fare", "FamilySize"), ("Age", "SibSp")]
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.2))
    for ax, (x, y) in zip(axes, pairs):
        sub = clean[[x, y, "Survived"]].dropna()
        for s, c, name in [(0, "#4C72B0", "遇难"), (1, "#DD8452", "生还")]:
            m = sub["Survived"] == s
            ax.scatter(sub.loc[m, x], sub.loc[m, y], s=9, alpha=0.6, color=c, label=name)
        r, p = stats.pearsonr(sub[x], sub[y])
        rho, ps = stats.spearmanr(sub[x], sub[y])
        # 拟合趋势线
        z = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="#333333", lw=1.1, ls="--")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}\nr={r:.3f} (p={p:.1e})  ρ={rho:.3f}", fontsize=7.8)
        ax.legend(fontsize=6.4, frameon=False, loc="best")
    fig.tight_layout()
    return fig
