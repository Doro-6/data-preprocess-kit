# -*- coding: utf-8 -*-
"""描述性统计：集中趋势、离散程度、分布形态、分组汇总与区间估计。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def numeric_description(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """数值型变量的完整描述性统计表（含四分位、偏度、峰度、变异系数、异常值边界）。"""
    cols = cols or ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    rows = []
    for col in cols:
        s = df[col].dropna().astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        rows.append({
            "变量": col,
            "样本量": int(s.size),
            "缺失数": int(df[col].isna().sum()),
            "均值": round(float(s.mean()), 4),
            "中位数": round(float(s.median()), 4),
            "众数": round(float(s.mode().iloc[0]), 4),
            "标准差": round(float(s.std(ddof=1)), 4),
            "方差": round(float(s.var(ddof=1)), 4),
            "变异系数": round(float(s.std(ddof=1) / s.mean()), 4) if s.mean() else None,
            "最小值": round(float(s.min()), 4),
            "P25": round(float(q1), 4),
            "P75": round(float(q3), 4),
            "最大值": round(float(s.max()), 4),
            "极差": round(float(s.max() - s.min()), 4),
            "四分位距": round(float(iqr), 4),
            "标准误": round(float(s.std(ddof=1) / np.sqrt(s.size)), 4),
            "偏度": round(float(stats.skew(s)), 4),
            "峰度": round(float(stats.kurtosis(s)), 4),
            "IQR 下界": round(float(q1 - 1.5 * iqr), 4),
            "IQR 上界": round(float(q3 + 1.5 * iqr), 4),
            "IQR 异常值数": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
        })
    return pd.DataFrame(rows)


def categorical_description(df: pd.DataFrame) -> pd.DataFrame:
    """分类型变量的频数分布表。"""
    cols = ["Survived", "Pclass", "Sex", "Embarked", "Title", "Deck",
            "IsAlone", "HasCabin", "FamilySize"]
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).sort_index()
        total = int(vc.sum())
        cum = 0.0
        for level, count in vc.items():
            freq = count / total
            cum += freq
            rows.append({
                "变量": col,
                "取值": "缺失" if pd.isna(level) else str(level),
                "频数": int(count),
                "频率": round(float(freq), 4),
                "频率百分比": f"{freq:.2%}",
                "累计频率": round(float(cum), 4),
            })
    return pd.DataFrame(rows)


def shape_statistics(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """分布形态检验：偏度、峰度及其显著性检验（D'Agostino、Shapiro、KS）。"""
    cols = cols or ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
    rows = []
    for col in cols:
        s = df[col].dropna().astype(float)
        skew = float(stats.skew(s))
        kurt = float(stats.kurtosis(s))
        skew_test = stats.skewtest(s) if s.size >= 8 else (np.nan, np.nan)
        kurt_test = stats.kurtosistest(s) if s.size >= 20 else (np.nan, np.nan)
        norm_test = stats.normaltest(s) if s.size >= 20 else (np.nan, np.nan)
        shapiro = stats.shapiro(s) if s.size <= 5000 else (np.nan, np.nan)
        z = (s - s.mean()) / s.std(ddof=1)
        rows.append({
            "变量": col,
            "样本量": int(s.size),
            "偏度": round(skew, 4),
            "偏度方向": "右偏（长尾在右）" if skew > 0.1 else ("左偏（长尾在左）" if skew < -0.1 else "近似对称"),
            "偏度检验 p": float(skew_test[1]),
            "峰度(超额)": round(kurt, 4),
            "峰度形态": "尖峰厚尾" if kurt > 0.5 else ("低峰" if kurt < -0.5 else "接近正态"),
            "峰度检验 p": float(kurt_test[1]),
            "D'Agostino p": float(norm_test[1]),
            "Shapiro p": float(shapiro[1]),
            "是否服从正态(p>0.05)": "否" if norm_test[1] < 0.05 else "是",
            "|Z|>3 的观测数": int((z.abs() > 3).sum()),
        })
    return pd.DataFrame(rows)


def grouped_survival(df: pd.DataFrame) -> pd.DataFrame:
    """单一分组变量的生存率汇总（含生存率标准误与卡方检验）。"""
    rows = []
    for col in ["Sex", "Pclass", "Embarked", "Title", "IsAlone", "HasCabin", "Deck"]:
        if col not in df.columns:
            continue
        for level, grp in df.groupby(col, observed=True):
            n = len(grp)
            k = int(grp["Survived"].sum())
            rate = k / n
            se = np.sqrt(rate * (1 - rate) / n)
            rows.append({
                "分组变量": col,
                "分组": str(level),
                "样本量": n,
                "生还人数": k,
                "遇难人数": n - k,
                "生存率": round(rate, 4),
                "生存率百分比": f"{rate:.2%}",
                "标准误": round(float(se), 4),
                "95%CI 下限": round(float(max(0.0, rate - 1.96 * se)), 4),
                "95%CI 上限": round(float(min(1.0, rate + 1.96 * se)), 4),
                "平均年龄": round(float(grp["Age"].mean()), 2),
                "票价中位数": round(float(grp["Fare"].median()), 2),
                "女性占比": round(float((grp["Sex"] == "female").mean()), 4),
            })
    return pd.DataFrame(rows)


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """多变量交叉分组汇总：每个分组给出生还率与显著性检验。"""
    specs = [
        ("Sex × Pclass", ["Sex", "Pclass"]),
        ("Pclass × 年龄段", ["Pclass", "AgeGroup"]),
        ("Sex × 家庭角色", ["Sex", "FamilyRole"]),
        ("Pclass × HasCabin", ["Pclass", "HasCabin"]),
    ]
    data = df.copy()
    data["FamilyRole"] = np.where(data["IsAlone"] == 1, "独行",
                                  np.where(data["FamilySize"] <= 4, "小家庭", "大家庭"))
    data["AgeGroup"] = pd.cut(data["Age"], bins=[-0.01, 12, 18, 35, 60, np.inf],
                              labels=["儿童", "少年", "青年", "中年", "老年"])
    rows = []
    for name, keys in specs:
        tab = data.groupby(keys, observed=True)["Survived"].agg(["count", "sum"])
        tab["生存率"] = tab["sum"] / tab["count"]
        # 该分组方案的整体卡方检验
        ct = pd.crosstab([data[k] for k in keys], data["Survived"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.to_numpy().sum()
        v = float(np.sqrt((chi2 / n) / (min(ct.shape) - 1)))
        for idx, r in tab.iterrows():
            label = " × ".join(str(i) for i in (idx if isinstance(idx, tuple) else (idx,)))
            rows.append({
                "分组方案": name,
                "分组": label,
                "样本量": int(r["count"]),
                "生还人数": int(r["sum"]),
                "生存率": round(float(r["生存率"]), 4),
                "生存率百分比": f"{r['生存率']:.2%}",
                "方案卡方": round(float(chi2), 4),
                "方案自由度": int(dof),
                "方案 p 值": float(p),
                "方案 Cramér's V": round(v, 4),
                "方案是否显著": "是" if p < 0.05 else "否",
            })
    return pd.DataFrame(rows)


def key_survival_rates(df: pd.DataFrame) -> pd.DataFrame:
    """关键群体生存率汇总表（含样本量、生还/遇难人数、置信区间与相关系数）。"""
    groups = [
        ("总体", df, "—"),
        ("女性（Sex = female）", df[df["Sex"] == "female"], "Sex"),
        ("男性（Sex = male）", df[df["Sex"] == "male"], "Sex"),
        ("一等舱（Pclass = 1）", df[df["Pclass"] == 1], "Pclass"),
        ("二等舱（Pclass = 2）", df[df["Pclass"] == 2], "Pclass"),
        ("三等舱（Pclass = 3）", df[df["Pclass"] == 3], "Pclass"),
        ("独行乘客（IsAlone = 1）", df[df["IsAlone"] == 1], "IsAlone"),
        ("有家人同行（IsAlone = 0）", df[df["IsAlone"] == 0], "IsAlone"),
        ("有客舱号记录（HasCabin = 1）", df[df["HasCabin"] == 1], "HasCabin"),
        ("无客舱号记录（HasCabin = 0）", df[df["HasCabin"] == 0], "HasCabin"),
        ("女性且一等舱", df[(df["Sex"] == "female") & (df["Pclass"] == 1)], "组合"),
    ]
    base = float(df["Survived"].mean())
    rows = []
    for label, grp, var in groups:
        n = len(grp)
        k = int(grp["Survived"].sum())
        rate = k / n
        se = (rate * (1 - rate) / n) ** 0.5
        rows.append({
            "群体": label,
            "样本量": n,
            "生还人数": k,
            "遇难人数": n - k,
            "生存率": round(rate, 4),
            "生存率百分比": f"{rate:.2%}",
            "95%CI 下限": round(max(0.0, rate - 1.96 * se), 4),
            "95%CI 上限": round(min(1.0, rate + 1.96 * se), 4),
            "与总体差值": f"{rate - base:+.2%}",
            "分组变量": var,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> pd.DataFrame:
    """用 Bootstrap 重采样给出生存率的 95% 置信区间（对比正态近似的差异）。"""
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("全样本", df)] + [(f"Sex = {k}", g) for k, g in df.groupby("Sex", observed=True)] + \
             [(f"Pclass = {k}", g) for k, g in df.groupby("Pclass", observed=True)]
    for name, grp in groups:
        y = grp["Survived"].to_numpy()
        n = len(y)
        rate = y.mean()
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = y[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        se_norm = np.sqrt(rate * (1 - rate) / n)
        rows.append({
            "群体": name,
            "样本量": n,
            "生存率": round(float(rate), 4),
            "Bootstrap 标准误": round(float(boot.std(ddof=1)), 4),
            "正态近似标准误": round(float(se_norm), 4),
            "Bootstrap 95%CI 下限": round(float(lo), 4),
            "Bootstrap 95%CI 上限": round(float(hi), 4),
            "正态近似 95%CI 下限": round(float(max(0.0, rate - 1.96 * se_norm)), 4),
            "正态近似 95%CI 上限": round(float(min(1.0, rate + 1.96 * se_norm)), 4),
            "重采样次数": n_boot,
        })
    return pd.DataFrame(rows)


def key_facts(df: pd.DataFrame, shape_stats: pd.DataFrame, grouped: pd.DataFrame) -> dict:
    """抽取报告中需要引用的关键数值（保证正文与结果文件一致）。"""
    facts: dict = {}
    facts["n"] = int(len(df))
    facts["survived"] = int(df["Survived"].sum())
    facts["survival_rate"] = round(float(df["Survived"].mean()), 4)
    facts["female_rate"] = round(float(df.loc[df["Sex"] == "female", "Survived"].mean()), 4)
    facts["male_rate"] = round(float(df.loc[df["Sex"] == "male", "Survived"].mean()), 4)
    for p in (1, 2, 3):
        facts[f"pclass{p}_rate"] = round(float(df.loc[df["Pclass"] == p, "Survived"].mean()), 4)
    facts["age_mean"] = round(float(df["Age"].mean()), 4)
    facts["age_median"] = round(float(df["Age"].median()), 4)
    facts["age_std"] = round(float(df["Age"].std(ddof=1)), 4)
    facts["fare_mean"] = round(float(df["Fare"].mean()), 4)
    facts["fare_median"] = round(float(df["Fare"].median()), 4)
    facts["fare_skew"] = round(float(stats.skew(df["Fare"].dropna())), 4)
    facts["age_skew"] = round(float(stats.skew(df["Age"].dropna())), 4)
    facts["fare_mean_by_class"] = {int(k): round(float(v), 2)
                                   for k, v in df.groupby("Pclass")["Fare"].mean().items()}
    facts["fare_median_by_class"] = {int(k): round(float(v), 2)
                                     for k, v in df.groupby("Pclass")["Fare"].median().items()}
    facts["age_mean_by_class"] = {int(k): round(float(v), 2)
                                  for k, v in df.groupby("Pclass")["Age"].mean().items()}
    facts["shape"] = shape_stats.set_index("变量")[["偏度", "峰度(超额)", "是否服从正态(p>0.05)"]].to_dict("index")
    facts["alone_rate"] = round(float(df.loc[df["IsAlone"] == 1, "Survived"].mean()), 4)
    facts["with_family_rate"] = round(float(df.loc[df["IsAlone"] == 0, "Survived"].mean()), 4)
    facts["has_cabin_rate"] = round(float(df.loc[df["HasCabin"] == 1, "Survived"].mean()), 4)
    facts["no_cabin_rate"] = round(float(df.loc[df["HasCabin"] == 0, "Survived"].mean()), 4)
    facts["max_survival_group"] = grouped.loc[grouped["生存率"].idxmax()][["分组变量", "分组", "生存率"]].to_dict()
    facts["min_survival_group"] = grouped.loc[grouped["生存率"].idxmin()][["分组变量", "分组", "生存率"]].to_dict()
    return facts


# ==========================================================================
# 插图
# ==========================================================================
def fig_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    specs = [
        ("Age", "年龄分布", "#4C72B0"),
        ("Fare", "票价分布", "#DD8452"),
        ("FamilySize", "家庭规模分布", "#55A868"),
    ]
    for i, (col, title, color) in enumerate(specs):
        s = df[col].dropna().astype(float)
        ax = axes[0, i]
        bins = 30 if col != "FamilySize" else np.arange(0.5, s.max() + 1.5, 1)
        ax.hist(s, bins=bins, color=color, alpha=0.82, edgecolor="white", lw=0.4, density=True)
        kde_x = np.linspace(s.min(), s.max(), 200)
        ax.plot(kde_x, stats.gaussian_kde(s)(kde_x), color="#C44E52", lw=1.3, label="核密度估计")
        ax.axvline(s.mean(), color="#333333", ls="--", lw=1.0, label=f"均值 {s.mean():.2f}")
        ax.axvline(s.median(), color="#333333", ls=":", lw=1.2, label=f"中位数 {s.median():.2f}")
        ax.set_title(f"({'abc'[i]}) {title}", fontsize=8.8)
        ax.set_xlabel(col)
        ax.set_ylabel("密度")
        ax.legend(fontsize=6.6, frameon=False)
        ax.text(0.97, 0.95, f"n={s.size}\n偏度 {stats.skew(s):.2f}\n变异系数 "
                            f"{s.std(ddof=1) / s.mean():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                bbox=dict(fc="white", ec="#CCCCCC", lw=0.6, pad=2.2))

        ax2 = axes[1, i]
        bp = ax2.boxplot([s], vert=False, patch_artist=True, widths=0.5,
                         flierprops=dict(marker="o", ms=2.8, markerfacecolor="#C44E52", alpha=0.6))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        ax2.set_yticks([])
        ax2.set_xlabel(col)
        ax2.set_title(f"({'def'[i]}) {title}——箱线图（Q1 {q1:.2f} / 中位 {s.median():.2f} / Q3 {q3:.2f}）",
                      fontsize=8.8)
    fig.tight_layout()
    return fig


def fig_grouped_survival(grouped: pd.DataFrame):
    vars_ = ["Sex", "Pclass", "Embarked", "HasCabin", "IsAlone", "Title"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))
    for ax, var in zip(axes.ravel(), vars_):
        sub = grouped[grouped["分组变量"] == var].sort_values("生存率", ascending=False)
        x = np.arange(len(sub))
        err = np.vstack([sub["生存率"] - sub["95%CI 下限"], sub["95%CI 上限"] - sub["生存率"]])
        bars = ax.bar(x, sub["生存率"] * 100, yerr=err * 100, capsize=3, width=0.6,
                      color=["#DD8452" if v > 0.3838 else "#4C72B0" for v in sub["生存率"]],
                      error_kw=dict(lw=0.8, ecolor="#555555"))
        ax.axhline(38.38, color="#333333", ls="--", lw=1.0)
        ax.text(len(sub) - 0.5, 40.5, "总体 38.38%", ha="right", fontsize=6.6)
        for b, (v, n) in zip(bars, zip(sub["生存率"], sub["样本量"])):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 6.5, f"{v:.1%}\nn={n}",
                    ha="center", fontsize=6.4)
        ax.set_xticks(x, [str(v)[:12] for v in sub["分组"]], fontsize=7)
        ax.set_ylim(0, 118)
        ax.set_ylabel("生存率 (%)")
        ax.set_title(f"{var}（误差线为 95% 置信区间）", fontsize=8.8)
    fig.tight_layout()
    return fig


def fig_shape_diagnostics(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    cols = ["Age", "Fare", "FamilySize"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # (a) 偏度/峰度对比
    ax = axes[0]
    skews = [stats.skew(df[c].dropna()) for c in cols]
    kurts = [stats.kurtosis(df[c].dropna()) for c in cols]
    x = np.arange(len(cols))
    ax.bar(x - 0.19, skews, 0.36, label="偏度", color="#4C72B0")
    ax.bar(x + 0.19, kurts, 0.36, label="超额峰度", color="#DD8452")
    ax.axhline(0, color="black", lw=0.7)
    for xi, (a, b) in enumerate(zip(skews, kurts)):
        ax.text(xi - 0.19, a + (0.5 if a > 0 else -1.4), f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.19, b + (0.5 if b > 0 else -1.4), f"{b:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, cols)
    ax.set_ylabel("取值")
    ax.set_title("(a) 分布形态指标（正态分布基准 = 0）", fontsize=8.8)
    ax.legend(fontsize=7, frameon=False)

    # (b) Q-Q 图
    ax = axes[1]
    for c, col in zip(colors, cols):
        s = df[col].dropna().astype(float)
        z = (s - s.mean()) / s.std(ddof=1)
        osm, osr = stats.probplot(z, dist="norm", fit=False)
        ax.plot(osm, osr, "o", ms=2.2, color=c, alpha=0.6, label=col)
    lim = [-3.6, 3.6]
    ax.plot(lim, lim, "k--", lw=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(-4, 12)
    ax.set_xlabel("理论分位数（标准正态）")
    ax.set_ylabel("样本分位数（标准化后）")
    ax.set_title("(b) 正态 Q-Q 图：点偏离直线即偏离正态", fontsize=8.8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # (c) 经验分布 vs 正态
    ax = axes[2]
    s = df["Fare"].dropna().astype(float).sort_values()
    ecdf = np.arange(1, len(s) + 1) / len(s)
    ax.plot(s, ecdf, lw=1.4, color="#DD8452", label="Fare 经验分布")
    grid = np.linspace(s.min(), s.max(), 300)
    ax.plot(grid, stats.norm.cdf(grid, s.mean(), s.std(ddof=1)), ls="--", lw=1.2,
            color="#4C72B0", label="同均值方差的正态分布")
    ax.set_xlim(-20, 320)
    ax.set_xlabel("票价")
    ax.set_ylabel("累计概率")
    ax.set_title("(c) 经验分布与正态分布的理论差距", fontsize=8.8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
