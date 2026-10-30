# -*- coding: utf-8 -*-
"""数据转换：无量纲化、非线性变换、离散化、编码。

对应实验任务的“掌握数据转换的常用方法”。转换流程被封装为可复用的
``FeatureBuilder``，训练集与测试集共用同一套变换参数。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

# 参与建模的字段定义
NUMERIC_BASE = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
NUMERIC_SKEWED = ["Fare", "FamilySize"]          # 右偏明显，需先做对数变换
CATEGORICAL_NOMINAL = ["Sex", "Embarked", "Deck", "Title"]
CATEGORICAL_ORDINAL = {"Pclass": {1: 1, 2: 2, 3: 3}}   # 顺序编码：1 等舱 < 2 等舱 < 3 等舱
BINARY_FLAGS = ["IsAlone", "HasCabin", "AgeImputed", "FareImputed", "EmbarkedFilled"]


# ==========================================================================
# 1. 偏度与非线性变换
# ==========================================================================
def skew_report(clean: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    """比较关键变量在变换前后的偏度与峰度。"""
    rows = []
    pairs = [
        ("Fare", "原始票价", "Fare_log", "对数变换 lg(1+Fare)"),
        ("FamilySize", "家庭规模原值", "FamilySize_log", "对数变换 lg(1+x)"),
        ("Age", "原始年龄", "Age_yeojohnson", "Yeo-Johnson 幂变换"),
        ("Age", "原始年龄", "Age_standardized", "Z 分数标准化"),
        ("Fare", "原始票价", "Fare_minmax", "极差归一化"),
    ]
    for src, src_name, dst, dst_name in pairs:
        if src not in clean.columns or dst not in transformed.columns:
            continue
        s = clean[src].dropna().astype(float)
        t = transformed[dst].dropna().astype(float)
        # Age 的幂变换结果与原始 Age 行索引一致
        rows.append({
            "变量": src,
            "变换方式": dst_name,
            "变换前偏度": round(float(stats.skew(s)), 4),
            "变换后偏度": round(float(stats.skew(t)), 4),
            "变换前峰度": round(float(stats.kurtosis(s)), 4),
            "变换后峰度": round(float(stats.kurtosis(t)), 4),
            "变换前极差": round(float(s.max() - s.min()), 4),
            "变换后极差": round(float(t.max() - t.min()), 4),
            "变换前标准差": round(float(s.std(ddof=1)), 4),
            "变换后标准差": round(float(t.std(ddof=1)), 4),
            "是否降低偏度": "是" if abs(stats.skew(t)) < abs(stats.skew(s)) else "否",
        })
    return pd.DataFrame(rows)


def build_feature_matrix(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """核心转换流程：派生 -> 对数变换 -> 幂变换 -> 标准化 -> 编码。

    返回 (转换后的特征矩阵, 过程摘要)。
    """
    out = pd.DataFrame(index=clean.index)
    facts: dict = {"steps": []}

    # ---- 步骤 1：ID 与标签单独保留 ----
    out["PassengerId"] = clean["PassengerId"]
    if "Survived" in clean.columns:
        out["Survived"] = clean["Survived"].astype(int)

    # ---- 步骤 2：对数变换压缩右偏 ----
    out["Fare_log"] = np.log1p(clean["Fare"].astype(float))
    out["FamilySize_log"] = np.log1p(clean["FamilySize"].astype(float))
    facts["steps"].append({
        "步骤": "对数变换", "字段": "Fare, FamilySize",
        "方法": "lg(1+x)", "目的": "压缩右偏长尾，减小极端票价对距离与梯度的支配",
    })

    # ---- 步骤 3：幂变换（Yeo-Johnson 可处理含零与负值的情形） ----
    pt = PowerTransformer(method="yeo-johnson")
    age_values = clean[["Age"]].astype(float)
    out["Age_yeojohnson"] = pt.fit_transform(age_values).ravel()
    facts["steps"].append({
        "步骤": "幂变换", "字段": "Age", "方法": "Yeo-Johnson（极大似然估计 λ）",
        "目的": "在保持单调性的前提下使分布更接近正态",
    })
    facts["yeojohnson_lambda"] = [round(float(v), 4) for v in pt.lambdas_]

    # ---- 步骤 4：标准化 / 归一化 ----
    out["Age_standardized"] = StandardScaler().fit_transform(age_values).ravel()
    out["Fare_minmax"] = MinMaxScaler().fit_transform(clean[["Fare"]].astype(float)).ravel()
    facts["steps"].append({
        "步骤": "无量纲化", "字段": "Age, Fare",
        "方法": "Z 分数标准化 / 极差归一化", "目的": "消除量纲差异，使距离度量可比",
    })

    # ---- 步骤 5：计数型与二值变量 ----
    for col in ["SibSp", "Parch", "FamilySize"]:
        out[col] = clean[col].astype(int)
    for col in BINARY_FLAGS:
        if col in clean.columns:
            out[col] = clean[col].astype(int)
    facts["steps"].append({
        "步骤": "二值化", "字段": ", ".join(BINARY_FLAGS),
        "方法": "0/1 指示变量", "目的": "把『是否独行』『是否填补』等状态显式编码",
    })

    # ---- 步骤 6：顺序编码 ----
    for col, mapping in CATEGORICAL_ORDINAL.items():
        out[f"{col}_ordinal"] = clean[col].map(mapping).astype(int)
    facts["ordinal_mapping"] = {k: {str(a): b for a, b in v.items()} for k, v in CATEGORICAL_ORDINAL.items()}

    # 有序分箱：年龄组按生命阶段编码
    age_bins = [-0.01, 12, 18, 35, 60, np.inf]
    age_labels = ["Child", "Teen", "Adult", "Middle", "Senior"]
    age_group = pd.cut(clean["Age"], bins=age_bins, labels=age_labels)
    out["AgeGroup_ordinal"] = age_group.cat.codes.astype(int)
    facts["age_group_definition"] = {
        "bins": ["≤12", "13–18", "19–35", "36–60", ">60"],
        "labels": age_labels,
        "counts": age_group.value_counts().reindex(age_labels).to_dict(),
    }

    # ---- 步骤 7：独热编码 ----
    dummies = pd.get_dummies(clean[CATEGORICAL_NOMINAL], columns=CATEGORICAL_NOMINAL,
                             prefix=CATEGORICAL_NOMINAL, dtype=int)
    out = pd.concat([out, dummies], axis=1)
    facts["one_hot_columns"] = list(dummies.columns)
    facts["steps"].append({
        "步骤": "独热编码", "字段": ", ".join(CATEGORICAL_NOMINAL),
        "方法": "get_dummies（k 个类别展开为 k 列 0/1）",
        "目的": "无序类别间不引入虚假的大小关系",
    })

    # ---- 步骤 8：分箱离散化 ----
    out["FareBin_ordinal"] = pd.qcut(clean["Fare"], 4, labels=False, duplicates="drop").astype(int)
    facts["steps"].append({
        "步骤": "等频分箱", "字段": "Fare",
        "方法": "四分位等频分箱", "目的": "降低票价噪声与极端值影响，便于列联表分析",
    })

    # ---- 步骤 9：统计口径 ----
    facts["output_shape"] = list(out.shape)
    facts["output_columns"] = list(out.columns)
    facts["numeric_columns"] = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    facts["missing_cells"] = int(out.isna().sum().sum())
    facts["continuous_columns"] = ["Fare", "Age", "Fare_log", "Age_yeojohnson",
                                   "Age_standardized", "Fare_minmax", "FamilySize_log", "FamilySize"]
    return out, facts


# ==========================================================================
# 2. 编码方案对照
# ==========================================================================
def encoding_demo(clean: pd.DataFrame) -> pd.DataFrame:
    """对同一批乘客展示四种编码结果，说明各自的适用场景。"""
    sample = clean.loc[clean["PassengerId"].isin([1, 2, 3, 4, 5]),
                       ["PassengerId", "Sex", "Embarked", "Pclass", "Deck", "Title"]].copy()
    sex_binary = sample["Sex"].map({"male": 1, "female": 0})
    sex_onehot = pd.get_dummies(sample["Sex"], prefix="Sex", dtype=int)
    title_dummies = pd.get_dummies(sample["Title"], prefix="Title", dtype=int).astype(int)

    out = pd.DataFrame({
        "PassengerId": sample["PassengerId"].to_numpy(),
        "Sex 原始": sample["Sex"].to_numpy(),
        "Sex 标签编码": sex_binary.to_numpy(),
        "Pclass 原值": sample["Pclass"].to_numpy(),
        "Pclass 顺序编码": sample["Pclass"].map(CATEGORICAL_ORDINAL["Pclass"]).to_numpy(),
        "Embarked 原值": sample["Embarked"].to_numpy(),
        "Embarked 独热(C)": sex_onehot.join(sample).pipe(
            lambda d: pd.get_dummies(d["Embarked"], dtype=int).get("C", pd.Series(0, index=d.index))).to_numpy(),
        "Title 原值": sample["Title"].to_numpy(),
    })
    for col in title_dummies.columns:
        out[col] = title_dummies[col].to_numpy()
    out["适用场景"] = ["——"] * len(out)
    return out


# ==========================================================================
# 3. 离散化与列联分析
# ==========================================================================
def discretize(clean: pd.DataFrame) -> pd.DataFrame:
    """构造等宽、等频与业务规则三种分箱结果。"""
    out = clean.copy()
    out["AgeGroup"] = pd.cut(out["Age"], bins=[-0.01, 12, 18, 35, 60, np.inf],
                             labels=["儿童(≤12)", "少年(13-18)", "青年(19-35)", "中年(36-60)", "老年(>60)"])
    out["FareQuartile"] = pd.qcut(out["Fare"], 4,
                                  labels=["Q1 最低", "Q2 偏低", "Q3 偏高", "Q4 最高"], duplicates="drop")
    out["FareEqualWidth"] = pd.cut(out["Fare"], bins=4,
                                   labels=["W1", "W2", "W3", "W4"], include_lowest=True)
    return out


def binning_analysis(binned: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """分箱后的生存率列联表与卡方检验。"""
    rows = []
    facts: dict = {}
    for col, name in [("AgeGroup", "年龄业务分箱"), ("FareQuartile", "票价等频四分箱"),
                      ("FareEqualWidth", "票价等宽分箱"), ("FamilySizeBin", "家庭规模分箱")]:
        if col not in binned.columns:
            if col == "FamilySizeBin":
                s = pd.cut(binned["FamilySize"], bins=[0, 1, 3, 5, 20],
                           labels=["独行(1)", "小家庭(2-3)", "中等(4-5)", "大家庭(6+)"])
            else:
                continue
        else:
            s = binned[col]
        tab = pd.crosstab(s, binned["Survived"])
        if tab.shape[0] < 2:
            continue
        tab.columns = ["遇难(0)", "生还(1)"]
        tab["合计"] = tab.sum(axis=1)
        tab["生存率"] = (tab["生还(1)"] / tab["合计"]).round(4)
        tab["生存率百分比"] = (tab["生存率"] * 100).round(2).astype(str) + "%"
        chi2, p, dof, _ = stats.chi2_contingency(tab[["遇难(0)", "生还(1)"]])
        n = tab["合计"].sum()
        cramers_v = float(np.sqrt((chi2 / n) / (min(tab.shape[0], 2) - 1)))
        tab = tab.reset_index().rename(columns={tab.index.name or "index": "分组"})
        tab.insert(0, "分箱方式", name)
        rows.append(tab)
        facts[name] = {
            "chi2": round(float(chi2), 4),
            "dof": int(dof),
            "p_value": float(p),
            "cramers_v": round(cramers_v, 4),
            "n_groups": int(tab.shape[0]),
            "rate_range": f"{tab['生存率'].min():.3f}–{tab['生存率'].max():.3f}",
            "significant": bool(p < 0.05),
        }
    return pd.concat(rows, ignore_index=True), facts


# ==========================================================================
# 4. 插图
# ==========================================================================
def fig_transformations(clean: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))

    specs = [
        ("Fare", "Fare_log", "票价：原始 vs 对数变换", "lg(1+Fare)"),
        ("Age", "Age_yeojohnson", "年龄：原始 vs Yeo-Johnson 变换", "Yeo-Johnson"),
        ("Fare", "Fare_minmax", "票价：原始 vs 极差归一化", "MinMax [0,1]"),
    ]
    pt = PowerTransformer(method="yeo-johnson")
    age_pow = pt.fit_transform(clean[["Age"]].astype(float)).ravel()
    log_fare = np.log1p(clean["Fare"].astype(float))
    minmax_fare = MinMaxScaler().fit_transform(clean[["Fare"]].astype(float)).ravel()
    series = {"Fare_log": log_fare, "Age_yeojohnson": age_pow, "Fare_minmax": minmax_fare}

    for i, (src, dst, title, dst_label) in enumerate(specs):
        ax = axes[0, i]
        raw = clean[src].dropna().astype(float)
        ax.hist(raw, bins=32, color="#B0B8C4", alpha=0.95, label=f"原始 {src}")
        ax.set_title(f"({'abc'[i]}) {title}", fontsize=8.8)
        ax.set_ylabel("人数")
        ax.text(0.97, 0.92, f"偏度 {stats.skew(raw):.2f}\n峰度 {stats.kurtosis(raw):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                bbox=dict(fc="white", ec="#CCCCCC", lw=0.6, pad=2.4))

        ax2 = axes[1, i]
        tr = pd.Series(series[dst])
        ax2.hist(tr, bins=32, color="#DD8452", alpha=0.95)
        ax2.set_title(f"({'def'[i]}) 变换后：{dst_label}", fontsize=8.8)
        ax2.set_xlabel(dst)
        ax2.set_ylabel("人数")
        ax2.text(0.97, 0.92, f"偏度 {stats.skew(tr):.2f}\n峰度 {stats.kurtosis(tr):.2f}",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=7.2,
                 bbox=dict(fc="white", ec="#CCCCCC", lw=0.6, pad=2.4))

    fig.tight_layout()
    return fig


def fig_encoding(clean: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.2]})

    ax = axes[0]
    vc = clean["Title"].value_counts()
    ax.bar(vc.index, vc.values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(vc)])
    for i, v in enumerate(vc.values):
        ax.text(i, v + 6, str(v), ha="center", fontsize=7.5)
    ax.set_title("(a) 称谓类别分布（独热编码列数 = 类别数）", fontsize=8.8)
    ax.set_ylabel("人数")
    ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[1]
    vc2 = clean["Deck"].value_counts()
    ax.bar(vc2.index.astype(str), vc2.values,
           color=["#C44E52" if d == "Unknown" else "#4C72B0" for d in vc2.index.astype(str)])
    ax.set_title("(b) 甲板层分布（Unknown 占多数，故不单独成列）", fontsize=8.8)
    ax.set_ylabel("人数")
    ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[2]
    enc = pd.DataFrame({
        "编码方案": ["标签编码\n(Sex)", "顺序编码\n(Pclass)", "独热编码\n(Embarked)", "独热编码\n(Title)", "二值指示\n(HasCabin)"],
        "产生列数": [1, 1, 3, 5, 1],
    })
    bars = ax.bar(enc["编码方案"], enc["产生列数"],
                  color=["#4C72B0", "#DD8452", "#55A868", "#55A868", "#8172B3"], width=0.6)
    for b, v in zip(bars, enc["产生列数"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, str(v), ha="center", fontsize=8)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("新增列数")
    ax.set_title("(c) 各字段编码后的维度代价", fontsize=8.8)
    ax.tick_params(axis="x", labelsize=6.8)

    fig.tight_layout()
    return fig


def fig_binning(bin_table: pd.DataFrame):
    groups = bin_table["分箱方式"].unique()
    fig, axes = plt.subplots(1, len(groups), figsize=(11.5, 3.5))
    for ax, g in zip(np.atleast_1d(axes), groups):
        sub = bin_table[bin_table["分箱方式"] == g]
        x = np.arange(len(sub))
        ax.bar(x - 0.19, sub["遇难(0)"], 0.36, label="遇难", color="#C44E52")
        ax.bar(x + 0.19, sub["生还(1)"], 0.36, label="生还", color="#55A868")
        ax2 = ax.twinx()
        ax2.plot(x, sub["生存率"] * 100, "o-", color="#4C72B0", lw=1.4, ms=4, label="生存率")
        for xi, v in zip(x, sub["生存率"] * 100):
            ax2.text(xi, v + 5, f"{v:.0f}%", ha="center", fontsize=7, color="#4C72B0")
        ax.set_xticks(x, [str(v).replace("(", "\n(") for v in sub["分组"]], fontsize=6.8)
        ax.set_ylabel("人数")
        ax2.set_ylabel("生存率 (%)", color="#4C72B0")
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="y", labelcolor="#4C72B0", labelsize=7)
        ax.set_title(g, fontsize=8.8)
        if ax is np.atleast_1d(axes)[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6.8, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig
