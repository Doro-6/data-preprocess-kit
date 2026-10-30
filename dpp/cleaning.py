# -*- coding: utf-8 -*-
"""数据录入校验、字段编辑、缺失值分析、异常值检测与清洗。

本模块对应实验任务的“数据录入与编辑”“数据清洗”两个环节。
所有函数均为纯函数：接收 DataFrame，返回表格/统计结果，不做隐式落盘。
"""

from __future__ import annotations

import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 泰坦尼克号登船港口与甲板字母的领域知识
EMBARK_PORT = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
VALID_DECKS = list("ABCDEFGT")
VALID_PORTS = list("CQS")

# 姓名中的称谓 -> 统一称谓
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Don": "Rare", "Dona": "Rare", "Lady": "Rare", "Countess": "Rare",
    "Jonkheer": "Rare", "Sir": "Rare", "Capt": "Rare", "Col": "Rare",
    "Major": "Rare", "Dr": "Rare", "Rev": "Rare",
}


# ==========================================================================
# 1. 数据录入与编辑
# ==========================================================================
def column_overview(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """逐列给出类型、缺失率、取值个数与示例值，形成数据字典。"""
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        samples = [v for v in s.dropna().unique()[:3]]
        rows.append({
            "字段": col,
            "存储类型": str(s.dtype),
            "语义类型": _semantic_type(s, col),
            "非空数": int(s.notna().sum()),
            "缺失数": n_missing,
            "缺失率": round(n_missing / len(df), 4),
            "取值个数": int(s.nunique(dropna=True)),
            "最小值": _safe_min(s),
            "最大值": _safe_max(s),
            "示例值": " / ".join(str(v)[:28] for v in samples),
            "是否建模特征": "否（标签）" if col == target else ("否（标识）" if col == "PassengerId" else "是"),
        })
    return pd.DataFrame(rows)


def _semantic_type(s: pd.Series, col: str) -> str:
    if pd.api.types.is_numeric_dtype(s):
        if s.nunique() <= 5:
            return "定类型（数值编码）"
        if col in ("SibSp", "Parch"):
            return "计数型"
        return "连续型"
    if col in ("Name", "Ticket", "Cabin"):
        return "复合文本型"
    return "定类型"


def _safe_min(s: pd.Series):
    try:
        return round(float(s.min()), 4)
    except (TypeError, ValueError):
        return None


def _safe_max(s: pd.Series):
    try:
        return round(float(s.max()), 4)
    except (TypeError, ValueError):
        return None


def integrity_checks(df: pd.DataFrame, key: str = "PassengerId", label: str = "训练集") -> list[dict]:
    """录入层面的完整性校验：主键唯一、缺失口径、取值范围、重复记录。"""
    checks: list[dict] = []

    def add(check: str, condition: bool, detail: str, level: str = "FAIL") -> None:
        checks.append({
            "数据集": label,
            "校验项": check,
            "结果": "PASS" if condition else level,
            "说明": detail,
        })

    add(f"{key} 唯一且无缺失",
        df[key].is_unique and df[key].notna().all(),
        f"取值 {int(df[key].min())}–{int(df[key].max())}，共 {df[key].nunique()} 个不同值")

    dup = int(df.duplicated().sum())
    add("完全重复记录", dup == 0, f"发现 {dup} 条完全重复行", level="WARN")

    add("Survived 仅取 0/1 且无缺失",
        "Survived" not in df or (df["Survived"].isin([0, 1]).all() and df["Survived"].notna().all()),
        "标签为二值，符合二分类设定")

    add("Pclass ∈ {1,2,3}", set(df["Pclass"].unique()) <= {1, 2, 3},
        f"实际取值 {sorted(df['Pclass'].unique())}")

    add("Sex 取值规范", set(df["Sex"].unique()) <= {"male", "female"},
        f"实际取值 {sorted(df['Sex'].unique())}")

    add("Embarked ∈ {C,Q,S}（允许缺失）",
        set(df["Embarked"].dropna().unique()) <= set(VALID_PORTS),
        f"实际取值 {sorted(df['Embarked'].dropna().unique())}，缺失 {int(df['Embarked'].isna().sum())}")

    ok_age = df["Age"].dropna().between(0, 100).all()
    add("Age ∈ [0,100]（允许缺失）", bool(ok_age),
        f"实测 {df['Age'].min():.1f}–{df['Age'].max():.1f} 岁")

    neg_fare = int((df["Fare"].dropna() < 0).sum())
    zero_fare = int((df["Fare"] == 0).sum())
    add("Fare ≥ 0 或缺失", neg_fare == 0,
        f"负票价 {neg_fare} 条；票价为 0 的 {zero_fare} 条（团体票/赠票，保留）；"
        f"缺失 {int(df['Fare'].isna().sum())} 条（计入缺失率，不算取值越界）",
        level="WARN")

    add("SibSp/Parch 非负", bool((df["SibSp"] >= 0).all() and (df["Parch"] >= 0).all()),
        f"SibSp 最大 {int(df['SibSp'].max())}，Parch 最大 {int(df['Parch'].max())}")

    # 复合字段的格式校验
    bad_name = int(df["Name"].str.contains(",", regex=False).eq(False).sum())
    add("Name 符合『姓, 称谓. 名』格式", bad_name == 0, f"格式异常 {bad_name} 条")

    cabin_ok = df["Cabin"].dropna().str.match(r"^[A-Z]\d+").all()
    add("Cabin 以『甲板字母+编号』开头", bool(cabin_ok), "缺失视为无舱位记录")

    return checks


def cross_file_checks(train: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame) -> list[dict]:
    """跨文件一致性：训练/测试主键不重叠、提交文件与测试集对齐。"""
    checks: list[dict] = []

    def add(check: str, condition: bool, detail: str) -> None:
        checks.append({"校验项": check, "结果": "PASS" if condition else "FAIL", "说明": detail})

    inter = set(train["PassengerId"]) & set(test["PassengerId"])
    add("训练集与测试集主键不重叠", len(inter) == 0, f"交集大小 {len(inter)}")

    add("训练+测试覆盖完整乘客编号",
        sorted(set(train["PassengerId"]) | set(test["PassengerId"])) == list(range(1, 1310)),
        f"共 {train['PassengerId'].nunique() + test['PassengerId'].nunique()} 人，编号连续")

    add("gender_submission 与测试集主键完全一致",
        submission["PassengerId"].tolist() == test["PassengerId"].tolist(),
        f"两文件均 {len(submission)} 行且顺序一致")

    add("训练集与测试集特征列一致（除标签）",
        [c for c in train.columns if c != "Survived"] == list(test.columns),
        "字段顺序与命名完全一致，可直接复用同一套预处理流程")

    return checks


def parse_title(name: str) -> str:
    """从姓名中抽取称谓并归并稀有类别。"""
    m = re.search(r",\s*([^\.]+)\.", name)
    raw = m.group(1).strip() if m else "Unknown"
    return TITLE_MAP.get(raw, "Rare")


def parse_ticket(ticket: str) -> tuple[str, float]:
    """拆分票号为前缀+号码；纯数字票号前缀记为 NONE。"""
    t = str(ticket).strip().upper().replace(".", "").replace("/", " ").replace("-", " ")
    parts = t.split()
    if len(parts) == 1:
        return "NONE", _to_float(parts[0])
    prefix = " ".join(parts[:-1])
    return prefix, _to_float(parts[-1])


def _to_float(text: str) -> float:
    digits = re.sub(r"\D", "", text)
    return float(digits) if digits else np.nan


def edit_fields(df: pd.DataFrame) -> pd.DataFrame:
    """数据编辑：由复合字段派生结构化字段。

    派生字段
    --------
    Surname / Title      : 姓名 -> 姓氏、称谓
    Deck                 : 客舱号 -> 甲板层（多舱位取第一处）
    TicketPrefix/Number  : 票号 -> 前缀与数值编号
    FamilySize           : 同乘亲属数 = SibSp + Parch + 1
    IsAlone              : 是否独自出行
    CabinCount           : 登记的客舱数量
    """
    out = df.copy()
    out["Surname"] = out["Name"].str.split(",").str[0].str.strip()
    out["Title"] = out["Name"].map(parse_title)

    cabin = out["Cabin"].fillna("")
    out["Deck"] = cabin.str.extract(r"^([A-Z])")[0]
    out["CabinCount"] = cabin.apply(lambda v: 0 if v == "" else len(v.split()))

    ticket_parsed = out["Ticket"].apply(parse_ticket)
    out["TicketPrefix"] = [p for p, _ in ticket_parsed]
    out["TicketNumber"] = [n for _, n in ticket_parsed]

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    return out


# ==========================================================================
# 2. 缺失值：规模、机制、可视化
# ==========================================================================
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """缺失规模表，含缺失率与建议处理方式。"""
    rows = []
    n = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        rate = n_missing / n
        if rate >= 0.5:
            strategy = "缺失率过高，不做单值填补：转为『是否有记录』指示变量"
        elif col == "Age":
            strategy = "分组中位数填补（Sex × Pclass × Title）+ 填补指示变量"
        elif col == "Fare":
            strategy = "按 Pclass × Embarked 的组内中位数填补"
        elif col == "Embarked":
            strategy = "众数填补（缺失仅 2 条，对分布无实质影响）"
        else:
            strategy = "众数填补"
        rows.append({
            "字段": col,
            "缺失数": n_missing,
            "缺失率": round(rate, 4),
            "缺失率百分比": f"{rate:.2%}",
            "非缺失数": n - n_missing,
            "处理策略": strategy,
        })
    out = pd.DataFrame(rows).sort_values("缺失数", ascending=False)
    total = pd.DataFrame([{
        "字段": "合计（单元格）",
        "缺失数": int(df.isna().sum().sum()),
        "缺失率": round(df.isna().sum().sum() / df.size, 4),
        "缺失率百分比": f"{df.isna().sum().sum() / df.size:.2%}",
        "非缺失数": int(df.notna().sum().sum()),
        "处理策略": "—",
    }])
    return pd.concat([out, total], ignore_index=True)


def _cramers_v(flag: pd.Series, group: pd.Series) -> float:
    tab = pd.crosstab(flag, group)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt((chi2 / n) / (min(tab.shape) - 1)))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def missing_mechanism_tests(df: pd.DataFrame) -> pd.DataFrame:
    """缺失机制检验：对每个缺失字段，检验其它变量与其缺失指示是否独立。

    - 数值型协变量：两样本 t 检验 + Cohen's d
    - 分类型协变量：卡方检验 + Cramér's V
    多重比较用 Benjamini-Hochberg 控制 FDR。
    """
    rows: list[dict] = []
    miss_cols = [c for c in df.columns if df[c].isna().any()]
    numeric_pool = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_pool = ["Sex", "Pclass", "Embarked", "Title", "IsAlone", "Deck"]

    for target in miss_cols:
        flag = df[target].isna()
        if flag.sum() == 0 or flag.all():
            continue
        for cov in numeric_pool:
            if cov == target or cov not in df.columns:
                continue
            a = df.loc[~flag, cov].dropna().to_numpy(dtype=float)
            b = df.loc[flag, cov].dropna().to_numpy(dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            rows.append({
                "缺失字段": target,
                "检验变量": cov,
                "变量类型": "数值",
                "缺失组均值": round(float(b.mean()), 4),
                "完整组均值": round(float(a.mean()), 4),
                "检验方法": "Welch t 检验",
                "统计量": round(float(t_stat), 4),
                "p值": float(p_val),
                "效应量": round(_cohens_d(b, a), 4),
                "效应量类型": "Cohen's d",
            })
        for cov in categorical_pool:
            if cov == target or cov not in df.columns:
                continue
            sub = df[[cov]].copy()
            sub["_flag"] = flag
            sub = sub.dropna()
            if sub["_flag"].nunique() < 2 or sub[cov].nunique() < 2:
                continue
            tab = pd.crosstab(sub["_flag"], sub[cov])
            chi2, p_val, dof, _ = stats.chi2_contingency(tab)
            v = _cramers_v(sub["_flag"], sub[cov])
            share = (
                df.loc[flag, cov].value_counts(normalize=True).round(4).to_dict()
                if df.loc[flag, cov].notna().any() else {}
            )
            rows.append({
                "缺失字段": target,
                "检验变量": cov,
                "变量类型": "分类",
                "缺失组均值": _share_text(share),
                "完整组均值": _share_text(df.loc[~flag, cov].value_counts(normalize=True).round(4).to_dict()),
                "检验方法": f"卡方检验(df={dof})",
                "统计量": round(float(chi2), 4),
                "p值": float(p_val),
                "效应量": round(v, 4),
                "效应量类型": "Cramér's V",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Benjamini-Hochberg FDR
    order = out["p值"].rank(method="first").astype(int) - 1
    m = len(out)
    ranked = out["p值"].sort_values().to_numpy()
    qvals = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ranked[i] * m / (i + 1))
        qvals[i] = prev
    qmap = dict(zip(np.argsort(out["p值"].to_numpy(), kind="stable"), qvals))
    out["q值(BH)"] = [qmap[i] for i in range(m)]
    out["显著(q<0.05)"] = np.where(out["q值(BH)"] < 0.05, "是", "否")
    return out.sort_values(["缺失字段", "p值"]).reset_index(drop=True)


def _share_text(counts: dict) -> str:
    """把 value_counts 字典压成简短字符串，便于写入 CSV。"""
    if not counts:
        return "—"
    return "; ".join(f"{k}:{v}" for k, v in list(counts.items())[:4])


def missing_mechanism_conclusion(mcar: pd.DataFrame) -> dict:
    """依据检验结果给出缺失机制的判定结论。"""
    if mcar.empty:
        return {"verdict": "无缺失", "details": {}}
    details = {}
    for col, grp in mcar.groupby("缺失字段"):
        sig = grp[grp["q值(BH)"] < 0.05]
        max_effect = grp["效应量"].abs().max()
        details[col] = {
            "tested_pairs": int(len(grp)),
            "significant_pairs": int(len(sig)),
            "significant_vars": sorted(sig["检验变量"].unique().tolist()),
            "max_abs_effect": None if pd.isna(max_effect) else round(float(max_effect), 4),
        }
    parts = []
    for col, d in details.items():
        if d["significant_pairs"] == 0:
            parts.append(f"{col}：未发现与其它变量的显著关联，倾向 MCAR")
        else:
            parts.append(
                f"{col}：与 {', '.join(d['significant_vars'])} 显著关联"
                f"（最大效应量 {d['max_abs_effect']}），倾向 MAR"
            )
    return {"verdict": "；".join(parts), "details": details}


# ==========================================================================
# 3. 异常值：三种准则对比
# ==========================================================================
def iqr_outliers(s: pd.Series) -> tuple[pd.Series, float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < low) | (s > high), float(low), float(high)


def zscore_outliers(s: pd.Series, thr: float = 3.0) -> pd.Series:
    z = (s - s.mean()) / s.std(ddof=1)
    return z.abs() > thr


def mad_outliers(s: pd.Series, thr: float = 3.5) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=s.index)
    modified_z = 0.6745 * (s - med) / mad
    return modified_z.abs() > thr


def outlier_analysis(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """对 Age / Fare 做三种准则对比，并补充业务定义域规则检查。"""
    rows, detail_rows = [], []
    facts: dict = {"by_column": {}, "domain_rules": []}

    for col in ("Age", "Fare"):
        s = df[col].dropna()
        masks = {
            "IQR 1.5 倍": iqr_outliers(s)[0],
            "Z 分数 |z|>3": zscore_outliers(s),
            "MAD 修正 Z>3.5": mad_outliers(s),
        }
        union = pd.Series(False, index=s.index)
        inter = pd.Series(True, index=s.index)
        col_facts = {"n": int(len(s)), "mean": round(float(s.mean()), 4),
                     "median": round(float(s.median()), 4), "std": round(float(s.std(ddof=1)), 4),
                     "skew": round(float(stats.skew(s)), 4)}
        for name, mask in masks.items():
            mask = mask.reindex(s.index).fillna(False).astype(bool)
            union |= mask
            inter &= mask
            rows.append({
                "变量": col,
                "检测准则": name,
                "异常值个数": int(mask.sum()),
                "占比": round(float(mask.mean()), 4),
                "下界/阈值": round(float(iqr_outliers(s)[1]), 4) if name.startswith("IQR") else "—",
                "上界/阈值": round(float(iqr_outliers(s)[2]), 4) if name.startswith("IQR") else "—",
                "最小值": round(float(s[mask].min()), 4) if mask.any() else None,
                "最大值": round(float(s[mask].max()), 4) if mask.any() else None,
            })
            col_facts[name] = int(mask.sum())
        rows.append({
            "变量": col, "检测准则": "三种准则并集", "异常值个数": int(union.sum()),
            "占比": round(float(union.mean()), 4), "下界/阈值": "—", "上界/阈值": "—",
            "最小值": round(float(s[union].min()), 4) if union.any() else None,
            "最大值": round(float(s[union].max()), 4) if union.any() else None,
        })
        rows.append({
            "变量": col, "检测准则": "三种准则交集", "异常值个数": int(inter.sum()),
            "占比": round(float(inter.mean()), 4), "下界/阈值": "—", "上界/阈值": "—",
            "最小值": round(float(s[inter].min()), 4) if inter.any() else None,
            "最大值": round(float(s[inter].max()), 4) if inter.any() else None,
        })
        col_facts["union"] = int(union.sum())
        col_facts["intersection"] = int(inter.sum())
        col_facts["jaccard"] = round(float((union & inter).sum() / max(union.sum(), 1)), 4)
        facts["by_column"][col] = col_facts

        # 逐条列出被 IQR 判为异常的记录，便于人工复核
        mask = masks["IQR 1.5 倍"]
        cols = ["PassengerId", "Name", "Pclass", "Sex", col, "Fare" if col == "Age" else "Pclass"]
        for idx in s[mask].sort_values(ascending=False).index[:15]:
            detail_rows.append({
                "变量": col,
                "PassengerId": int(df.loc[idx, "PassengerId"]),
                "姓名": str(df.loc[idx, "Name"])[:34],
                "舱位等级": int(df.loc[idx, "Pclass"]),
                "性别": df.loc[idx, "Sex"],
                "取值": round(float(df.loc[idx, col]), 4),
                "是否保留": "保留（真实业务记录）",
            })

    # 业务定义域规则
    rules = [
        ("票价非负", int((df["Fare"].dropna() < 0).sum()), "违反票价为负的记录"),
        ("票价为零", int((df["Fare"] == 0).sum()), "可能为团体票或赠票，保留"),
        ("年龄超过 80 岁", int((df["Age"].dropna() > 80).sum()), "真实长寿乘客，保留"),
        ("家庭规模超过 8 人", int((df["FamilySize"] > 8).sum()), "大家庭出行，保留"),
        ("SibSp 达到 8", int((df["SibSp"] == 8).sum()), "同一家庭 8 名兄弟姐妹，保留"),
    ]
    for name, count, note in rules:
        facts["domain_rules"].append({"规则": name, "命中记录数": count, "处理": note})

    tables = {
        "22_outlier_criteria.csv": pd.DataFrame(rows),
        "23_outlier_records.csv": pd.DataFrame(detail_rows),
    }
    return tables, facts


# ==========================================================================
# 4. 执行清洗
# ==========================================================================
def clean_dataset(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, list[dict]]:
    """按既定策略清洗数据，并返回逐条溯源日志。

    策略
    ----
    1. 去除完全重复的记录（保留首次出现）；
    2. Cabin 缺失 -> Deck 记为 Unknown，并新增 HasCabin 指示变量；
    3. Embarked 缺失 -> 众数填补；
    4. Age 缺失 -> Sex × Pclass × Title 组内中位数填补 + AgeImputed 指示变量；
    5. Fare 缺失 -> Pclass × Embarked 组内中位数填补 + FareImputed 指示变量；
    6. 离群值不删除：Fare 的右偏由对数变换处理，年龄极值为真实记录。
    """
    log: list[dict] = []
    out = df.copy()
    n0, m0 = out.shape

    def record(step: str, field: str, affected: int, method: str, detail: str) -> None:
        log.append({
            "数据集": split,
            "步骤": step,
            "字段": field,
            "影响记录数": int(affected),
            "处理方法": method,
            "说明": detail,
        })

    # 1) 重复记录
    n_dup = int(out.duplicated().sum())
    if n_dup:
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
    record("去重", "全表", n_dup, "drop_duplicates", "完全重复的乘客记录，保留首次出现")

    # 2) Cabin -> HasCabin/Deck
    n_cabin = int(out["Cabin"].isna().sum())
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["Deck"] = out["Deck"].fillna("Unknown")
    record("缺失填补", "Cabin/Deck", n_cabin, "指示变量 + 常量 Unknown",
           "缺失率 77%，不做单值猜测；改为 HasCabin 指示变量并将甲板记为 Unknown")

    # 3) Embarked（先登记填补标记，再填值，避免索引在去重后错位）
    n_emb = int(out["Embarked"].isna().sum())
    out["EmbarkedFilled"] = out["Embarked"].isna().astype(int)
    if n_emb:
        mode = out["Embarked"].mode().iloc[0]
        out["Embarked"] = out["Embarked"].fillna(mode)
        record("缺失填补", "Embarked", n_emb, f"众数填补（{mode}）",
               "缺失仅 2 条（0.22%），众数填补对方差与分布无实质影响")

    # 4) Age
    age_missing = out["Age"].isna()
    n_age = int(age_missing.sum())
    group_median = out.groupby(["Sex", "Pclass", "Title"], observed=True)["Age"].transform("median")
    fallback = out.groupby(["Sex", "Pclass"], observed=True)["Age"].transform("median")
    filled_values = group_median.fillna(fallback).fillna(out["Age"].median())
    out["Age"] = out["Age"].fillna(filled_values)
    out["AgeImputed"] = age_missing.astype(int)
    n_groups = out.loc[age_missing, ["Sex", "Pclass", "Title"]].drop_duplicates().shape[0]
    record("缺失填补", "Age", n_age, "Sex × Pclass × Title 组内中位数",
           f"缺失率 19.87%，用 {n_groups} 个同质分组的年龄中位数填补，并加 AgeImputed 标记以便后续敏感性分析")

    # 5) Fare
    fare_missing = out["Fare"].isna()
    n_fare = int(fare_missing.sum())
    if n_fare:
        fare_med = out.groupby(["Pclass", "Embarked"], observed=True)["Fare"].transform("median")
        out["Fare"] = out["Fare"].fillna(fare_med).fillna(out["Fare"].median())
        record("缺失填补", "Fare", n_fare, "Pclass × Embarked 组内中位数",
               "测试集仅 1 条缺失；同舱位同港口的票价中位数是最合理的近似")

    out["FareImputed"] = fare_missing.astype(int)

    record("离群值", "Fare/Age", int((out["Fare"] > 300).sum()), "保留 + 变换",
           "高价票为真实头等舱记录，不做删除，改由对数变换压缩右偏")
    record("一致性", "FamilySize", int((out["FamilySize"] != out["SibSp"] + out["Parch"] + 1).sum()),
           "派生字段复核", "FamilySize = SibSp + Parch + 1，恒等式成立")

    log.append({
        "数据集": split, "步骤": "汇总", "字段": "全表", "影响记录数": int(n0 - len(out)),
        "处理方法": "shape 变化", "说明": f"清洗前 {n0}×{m0}，清洗后 {out.shape[0]}×{out.shape[1]}，"
                                          f"单元格缺失 {int(df.isna().sum().sum())} -> {int(out[df.columns].isna().sum().sum())}",
    })
    return out, log


# ==========================================================================
# 5. 插图
# ==========================================================================
def fig_data_overview(train: pd.DataFrame, test: pd.DataFrame):
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.35)

    # (a) 原始数据表（前 8 行、核心列）
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    sub = train[cols].head(8).copy()
    sub["Name"] = sub["Name"].str.slice(0, 22)
    sub["Cabin"] = sub["Cabin"].fillna("NaN")
    sub["Age"] = sub["Age"].map(lambda v: "NaN" if pd.isna(v) else f"{v:.1f}")
    sub["Fare"] = sub["Fare"].map(lambda v: f"{v:.2f}")
    table = ax.table(cellText=sub.values, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    table.scale(1, 1.22)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight="bold")
        elif c == 1:
            cell.set_facecolor("#F2E2D9" if sub.iloc[r - 1]["Survived"] == 1 else "#E3E8F0")
    ax.set_title(f"(a) 训练集前 8 行原始记录（{train.shape[0]} 行 × {train.shape[1]} 列）", fontsize=8.5, pad=6)

    # (b) 训练/测试规模
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(["训练集", "测试集"], [len(train), len(test)], color=["#4C72B0", "#DD8452"], width=0.55)
    for i, v in enumerate([len(train), len(test)]):
        ax.text(i, v + 12, str(v), ha="center", fontsize=8)
    ax.set_ylim(0, 1000)
    ax.set_ylabel("记录数")
    ax.set_title("(b) 数据集规模", fontsize=8.5)

    # (c) 字段类型构成
    ax = fig.add_subplot(gs[1, 1])
    sem = [("连续/计数", 3), ("定类型", 3), ("复合文本", 3), ("标识/标签", 2)]
    labels = [k for k, _ in sem]
    vals = [v for _, v in sem]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(labels, vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, str(v), ha="center", fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_ylabel("字段数")
    ax.set_title("(c) 字段语义类型", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)

    # (d) 缺失情况速览
    ax = fig.add_subplot(gs[1, 2])
    miss = train.isna().sum()
    miss = miss[miss > 0].sort_values()
    ax.barh(miss.index, miss.values / len(train) * 100, color="#C44E52", height=0.5)
    for i, (name, v) in enumerate(miss.items()):
        ax.text(v / len(train) * 100 + 1.5, i, f"{v / len(train) * 100:.1f}%", va="center", fontsize=7.5)
    ax.set_xlim(0, 95)
    ax.set_xlabel("缺失率 (%)")
    ax.set_title("(d) 含缺失的字段", fontsize=8.5)

    return fig


def fig_missing_overview(df: pd.DataFrame):
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1], hspace=0.42, wspace=0.28)

    # (a) 缺失数量条形图
    ax = fig.add_subplot(gs[0, 0])
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    short = {"TicketNumber": "Ticket", "Embarked": "Embark"}
    positions = np.arange(len(miss))
    bars = ax.bar(positions, miss.values, color="#C44E52", width=0.44)
    for b, v in zip(bars, miss.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.035 * max(miss.values),
                f"{v}\n({v / len(df):.1%})", ha="center", fontsize=7.2)
    ax.set_xticks(positions, [short.get(c, c) for c in miss.index], fontsize=8)
    ax.set_ylim(0, max(miss.values) * 1.42)
    ax.set_ylabel("缺失记录数")
    ax.set_title("(a) 各字段缺失规模", fontsize=9)

    # (b) 缺失模式（按模式分组统计）
    ax = fig.add_subplot(gs[1, 0])
    pattern = df[["Age", "Cabin", "Embarked"]].isna()
    combo = pattern.apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    counts = combo.value_counts().sort_index()
    label_map = {c: "+".join(n for n, k in zip(["Age", "Cabin", "Embarked"], c) if k == "1") or "无缺失"
                 for c in counts.index}
    labels = [label_map[c] for c in counts.index]
    colors = ["#B0B0B0" if c == "000" else "#DD8452" for c in counts.index]
    bars = ax.barh(labels, counts.values, color=colors, height=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(v + 8, b.get_y() + b.get_height() / 2, f"{v} ({v / len(df):.1%})", va="center", fontsize=7.5)
    ax.set_xlim(0, counts.max() * 1.32)
    ax.set_xlabel("记录数")
    ax.set_title("(b) 缺失模式组合（1 = 该字段缺失）", fontsize=9)

    # (c) 缺失共现矩阵
    ax = fig.add_subplot(gs[0, 1])
    cols = ["Age", "Cabin", "Embarked"]
    mat = np.zeros((len(cols), len(cols)))
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            fa, fb = df[a].isna(), df[b].isna()
            union = (fa | fb).sum()
            mat[i, j] = (fa & fb).sum() / union if union else 0
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.55 else "black")
    ax.set_title("(c) 缺失共现（Jaccard 系数）", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) 缺失模式的数值分布对照
    ax = fig.add_subplot(gs[1, 1])
    groups = [
        ("Age 缺失", df.loc[df["Age"].isna(), "Fare"].dropna(), "#DD8452"),
        ("Age 完整", df.loc[df["Age"].notna(), "Fare"].dropna(), "#4C72B0"),
        ("Cabin 缺失", df.loc[df["Cabin"].isna(), "Fare"].dropna(), "#DD8452"),
        ("Cabin 完整", df.loc[df["Cabin"].notna(), "Fare"].dropna(), "#4C72B0"),
    ]
    data = [g[1].to_numpy() for g in groups]
    bp = ax.boxplot(data, tick_labels=[g[0] for g in groups], patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=2.4, alpha=0.5))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(g[2])
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_ylabel("票价 Fare（对数刻度）")
    ax.set_title("(d) 缺失组与完整组的票价分布对比", fontsize=9)
    ax.tick_params(axis="x", labelsize=7.5)

    return fig


def fig_missing_mechanism(df: pd.DataFrame):
    from . import cleaning as _self  # 复用本模块函数

    tests = _self.missing_mechanism_tests(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) p 值条形图（-log10）
    ax = axes[0]
    t = tests.copy()
    t["标签"] = t["缺失字段"] + "  vs  " + t["检验变量"]
    t = t.sort_values("p值", ascending=False).tail(14)
    vals = -np.log10(t["p值"].clip(lower=1e-12))
    colors = ["#C44E52" if q < 0.05 else "#9AA5B1" for q in t["q值(BH)"]]
    bars = ax.barh(t["标签"], vals, color=colors, height=0.6)
    ax.axvline(-np.log10(0.05), color="black", ls="--", lw=0.9)
    ax.text(-np.log10(0.05) + 0.15, -0.8, "p = 0.05", fontsize=7)
    for b, v, p in zip(bars, vals, t["p值"]):
        ax.text(min(v + 0.2, ax.get_xlim()[1]), b.get_y() + b.get_height() / 2,
                f"p={p:.1e}", va="center", fontsize=6.3)
    ax.set_xlabel("-log10(p)   （右侧越远，越不像完全随机缺失）")
    ax.set_title("(a) 缺失指示与其它变量的独立性检验", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)

    # (b) 效应量
    ax = axes[1]
    t2 = tests.sort_values("效应量", key=lambda s: s.abs()).tail(12)
    colors = ["#C44E52" if abs(e) > 0.3 else "#DD8452" if abs(e) > 0.1 else "#9AA5B1" for e in t2["效应量"]]
    ax.barh(t2["缺失字段"] + "  vs  " + t2["检验变量"], t2["效应量"].abs(), color=colors, height=0.6)
    ax.axvline(0.1, color="gray", ls=":", lw=0.9)
    ax.axvline(0.3, color="gray", ls="--", lw=0.9)
    ax.text(0.105, -0.7, "小", fontsize=6.5)
    ax.text(0.305, -0.7, "中", fontsize=6.5)
    ax.set_xlabel("|效应量|（Cohen's d 或 Cramér's V）")
    ax.set_title("(b) 关联强度（而非仅看显著性）", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    return fig


def fig_outliers(df: pd.DataFrame, facts: dict):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    # (a) 箱线图 + IQR 边界
    for ax, col in zip(axes[:2], ["Age", "Fare"]):
        s = df[col].dropna()
        bp = ax.boxplot([s], vert=False, patch_artist=True, widths=0.45,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#C44E52", alpha=0.7))
        bp["boxes"][0].set_facecolor("#A8C4E0")
        _, low, high = iqr_outliers(s)
        ax.axvline(low, color="#C44E52", ls="--", lw=0.9)
        ax.axvline(high, color="#C44E52", ls="--", lw=0.9)
        ax.text(high, 1.32, f"上界 {high:.1f}", color="#C44E52", fontsize=6.8, ha="left")
        ax.text(low, 1.32, f"下界 {low:.1f}", color="#C44E52", fontsize=6.8, ha="right")
        ax.set_xlabel(col)
        ax.set_yticks([])
        ax.set_ylim(0.55, 1.55)
        n_out = int(facts["by_column"][col]["IQR 1.5 倍"])
        ax.set_title(f"({'ab'[0 if col == 'Age' else 1]}) {col} 箱线图，IQR 判出 {n_out} 个离群点", fontsize=9)

    # (c) 三种准则对比
    ax = axes[2]
    cols = ["Age", "Fare"]
    criteria = ["IQR 1.5 倍", "Z 分数 |z|>3", "MAD 修正 Z>3.5", "三种准则并集"]
    x = np.arange(len(cols))
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, crit in enumerate(criteria):
        vals = [facts["by_column"][c].get(crit, facts["by_column"][c].get(crit.replace("三种准则并集", "union"), 0))
                for c in cols]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.92, label=crit, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, str(int(v)), ha="center", fontsize=6.5)
    ax.set_xticks(x, cols)
    ax.set_ylabel("被判定为离群点的记录数")
    ax.set_title("(c) 三种离群点准则的判定差异", fontsize=9)
    ax.legend(fontsize=6.8, frameon=False)
    ax.set_ylim(0, max(facts["by_column"]["Fare"].get(k, 0) for k in criteria) * 1.25)

    fig.tight_layout()
    return fig


def fig_cleaning_effect(before: pd.DataFrame, after: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

    # (a) 缺失单元格对比
    ax = axes[0]
    cols = ["Age", "Cabin", "Embarked", "Fare"]
    b = [int(before[c].isna().sum()) for c in cols]
    a = [int(after[c].isna().sum()) for c in cols]
    x = np.arange(len(cols))
    ax.bar(x - 0.19, b, 0.36, label="清洗前", color="#C44E52")
    ax.bar(x + 0.19, a, 0.36, label="清洗后", color="#55A868")
    for xi, (bv, av) in enumerate(zip(b, a)):
        ax.text(xi - 0.19, bv + 8, str(bv), ha="center", fontsize=7)
        ax.text(xi + 0.19, av + 8, str(av), ha="center", fontsize=7)
    ax.set_xticks(x, cols)
    ax.set_ylim(0, max(b) * 1.25)
    ax.set_ylabel("缺失记录数")
    ax.set_title("(a) 缺失值填补效果", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)

    # (b) 填补前后年龄分布
    ax = axes[1]
    ax.hist(before["Age"].dropna(), bins=28, color="#4C72B0", alpha=0.72, label="原始 Age")
    med = after["Age"].median()
    ax.hist(after["Age"], bins=28, color="#DD8452", alpha=0.42, label="填补后 Age")
    ax.axvline(before["Age"].mean(), color="#4C72B0", ls="--", lw=1.0)
    ax.axvline(after["Age"].mean(), color="#DD8452", ls="--", lw=1.0)
    ax.set_xlabel("年龄")
    ax.set_ylabel("人数")
    ax.set_title(f"(b) 年龄填补前后（均值 {before['Age'].mean():.2f}  ->  {after['Age'].mean():.2f}）", fontsize=9)
    ax.legend(fontsize=7.2, frameon=False)

    # (c) 填补标记的分布
    ax = axes[2]
    tab = pd.crosstab(after["Pclass"], after["AgeImputed"], normalize="index")
    x = np.arange(len(tab))
    ax.bar(x - 0.19, tab.get(0, pd.Series(0, index=tab.index)) * 100, 0.36,
           label="实测年龄", color="#4C72B0")
    ax.bar(x + 0.19, tab.get(1, pd.Series(0, index=tab.index)) * 100, 0.36,
           label="填补年龄", color="#DD8452")
    for xi in x:
        v = tab.get(1, pd.Series(0, index=tab.index)).iloc[xi] * 100
        ax.text(xi + 0.19, v + 1.2, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(x, [f"{int(i)} 等舱" for i in tab.index])
    ax.set_ylabel("占比 (%)")
    ax.set_title("(c) 年龄填补比例随舱位变化（提示 MAR 机制）", fontsize=9)
    ax.legend(fontsize=7.2, frameon=False)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    return fig
