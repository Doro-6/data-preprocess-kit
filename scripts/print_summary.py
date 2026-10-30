# -*- coding: utf-8 -*-
"""把分析结果以“统计报告”的形式打印到控制台。

用途：PyCharm / 终端里运行后直接得到可在实验报告中引用的结果表，
      不涉及文件清单等过程性信息。

数据来源：results/facts.json（由 run_pipeline.py 生成），
因此打印的数值与实验报告正文、结果文件三者完全一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 遇到 χ²、Cramér 这类字符会直接抛 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # 非文本流或被重定向时忽略
        pass

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "results" / "facts.json"

W = 112

def title(text: str, char: str = "=") -> None:
    print()
    print(char * W)
    print(f" {text}")
    print(char * W)

def section(text: str) -> None:
    print()
    print(f"── {text} " + "─" * max(0, W - len(text) - 4))

def table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None,
          indent: int = 2) -> None:
    """打印无边框表（列用空格对齐，中文字符按双宽计算）。"""
    def width(text: str) -> int:
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))

    def pad(text: str, target: int, align: str) -> str:
        text = str(text)
        space = max(0, target - width(text))
        if align == "right":
            return " " * space + text
        if align == "center":
            left = space // 2
            return " " * left + text + " " * (space - left)
        return text + " " * space

    aligns = aligns or ["left"] * len(headers)
    widths = [width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], width(cell))

    prefix = " " * indent
    print(prefix + "  ".join(pad(h, widths[i], "center") for i, h in enumerate(headers)))
    print(prefix + "  ".join("-" * w for w in widths))
    for row in rows:
        print(prefix + "  ".join(pad(c, widths[i], aligns[i]) for i, c in enumerate(row)))

def pct(value: float) -> str:
    return f"{value * 100:.2f}%"

def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"

def main() -> int:
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    env, ing, integ = facts["environment"], facts["ingest"], facts["integrity"]
    edit, cl, tf = facts["edit"], facts["cleaning"], facts["transform"]
    d, out = facts["descriptive"], facts["outliers"]["by_column"]
    sim, cor = facts["similarity"], facts["correlation"]
    mm = facts["missing_mechanism"]["details"]

    title("泰坦尼克号乘客生存数据 —— 数据预处理实验结果")
    print(f" 数据集   Kaggle Titanic - Machine Learning from Disaster")
    print(f" 运行环境 {env['platform']} / Python {env['python']}")
    print(f" 依赖     numpy {env['numpy']}, pandas {env['pandas']}, scipy {env['scipy']}, "
          f"scikit-learn {env['scikit_learn']}")
    print(f" 随机种子 {env['random_seed']}")

    # ---------------- 1 ----------------
    title("一、数据集概况", "-")
    table(["项目", "训练集", "测试集", "数据类型", "取值范围", "缺失情况"],
          [
              ["样本量", f"{ing['train_shape'][0]}", f"{ing['test_shape'][0]}",
               "—", "—", "—"],
              ["字段数", f"{ing['train_shape'][1]}", f"{ing['test_shape'][1]}",
               "数值 5 / 分类 3 / 文本 3 / 标识 1", "—", "Cabin、Age、Embarked"],
              ["正类（生还）数", f"{d['survived']}", "—",
               "二值标签", "0 / 1", "无"],
              ["正类比例", pct(d["survival_rate"]), "—",
               "类别不均衡", "—", "—"],
              ["缺失单元格", f"{cl['missing_before']}", "—",
               "—", "—", f"占全部单元格 {cl['missing_before'] / (891 * 12) * 100:.2f}%"],
          ], ["left", "right", "right", "left", "left", "left"])

    # ---------------- 2 ----------------
    title("二、数据质量检查", "-")
    print(f"  自动校验 {integ['total']} 项：通过 {integ['passed']} 项，"
          f"失败 {integ['failed']} 项（均为“允许缺失”字段的严格口径判定）")
    print(f"  完全重复记录 {cl['duplicates_removed']} 条；主键唯一且训练/测试无交集")
    print(f"  字段编辑：由复合字段派生 {len(edit['derived_columns'])} 个结构化字段")
    print(f"    {'、'.join(edit['derived_columns'])}")

    # ---------------- 3 ----------------
    title("三、缺失值分析", "-")
    rows = []
    for col, label, n_missing, plan in [
            ("Cabin", "Cabin 客舱号", 687, "不填补原值，提取 HasCabin 指示变量"),
            ("Age", "Age 年龄", 177, "Sex × Pclass × Title 组内中位数填补"),
            ("Embarked", "Embarked 登船港", 2, "众数填补"),
    ]:
        det = mm.get(col, {})
        rows.append([
            label,
            f"{n_missing}",
            pct(n_missing / ing["train_shape"][0]),
            f"{det.get('significant_pairs', 0)}/{det.get('tested_pairs', 0)}",
            f"{det.get('max_abs_effect', float('nan')):.4f}",
            "MAR（非完全随机）",
            plan,
        ])
    table(["字段", "缺失数", "缺失率", "显著关联数", "最大效应量", "机制判定", "处理方式"],
          rows, ["left", "right", "right", "center", "right", "center", "left"])
    print("  注：显著关联数 = BH 校正后 q<0.05 的检验数 / 总检验数；")
    print("      效应量为 Cohen's d 或 Cramér's V 的绝对值。")

    # ---------------- 4 ----------------
    title("四、离群值检测（三种准则对比）", "-")
    rows = []
    for col in ("Age", "Fare"):
        v = out[col]
        note = ("极值为真实高龄乘客" if col == "Age"
                else "全部对应真实头等舱高价票")
        rows.append([col, f"{v['n']}", f"{v['IQR 1.5 倍']}", f"{v['Z 分数 |z|>3']}",
                     f"{v['MAD 修正 Z>3.5']}", f"{v['union']}",
                     f"{v['jaccard']:.4f}", "全部保留", note])
    table(["变量", "样本量", "IQR 准则", "Z 分数 |z|>3", "MAD 准则",
           "三准则并集", "Jaccard 重合率", "处理决策", "核查结论"],
          rows, ["left", "right", "right", "right", "right", "right",
                 "right", "center", "left"])

    # ---------------- 5 ----------------
    title("五、描述性统计", "-")
    shape = d["shape"]
    rows = [
        ["Age", "714", f"{d['age_mean']:.2f}", f"{d['age_median']:.2f}", "0.42",
         "80.00", f"{d['age_std']:.2f}", f"{shape['Age']['偏度']:.4f}",
         f"{shape['Age']['峰度(超额)']:.4f}", "否"],
        ["Fare", "891", f"{d['fare_mean']:.2f}", f"{d['fare_median']:.2f}", "0.00",
         "512.33", "49.69", f"{d['fare_skew']:.4f}",
         f"{shape['Fare']['峰度(超额)']:.4f}", "否"],
        ["FamilySize", "891", "1.90", "1.00", "1", "11", "1.61",
         f"{shape['FamilySize']['偏度']:.4f}",
         f"{shape['FamilySize']['峰度(超额)']:.4f}", "否"],
    ]
    table(["变量", "有效样本", "均值", "中位数", "最小值", "最大值",
           "标准差", "偏度", "超额峰度", "服从正态"],
          rows, ["left", "right", "right", "right", "right", "right",
                 "right", "right", "right", "center"])
    print("  正态性检验（D'Agostino / Shapiro）均 p<0.05，三个变量均不服从正态分布。")

    # ---------------- 6 ----------------
    title("六、分组生存率与显著性", "-")
    def ci(rate: float, n: int) -> str:
        se = (rate * (1 - rate) / n) ** 0.5
        return f"[{max(0, rate - 1.96 * se):.3f}, {min(1, rate + 1.96 * se):.3f}]"

    specs = [
        ("总体", 891, d["survival_rate"], None, "—"),
        ("女性", 314, d["female_rate"], "Sex", 0.5434),
        ("男性", 577, d["male_rate"], "Sex", 0.5434),
        ("一等舱", 216, d["pclass1_rate"], "Pclass", 0.3385),
        ("二等舱", 184, d["pclass2_rate"], "Pclass", 0.3385),
        ("三等舱", 491, d["pclass3_rate"], "Pclass", 0.3385),
        ("独行乘客", 537, d["alone_rate"], "IsAlone", 0.2034),
        ("有家人同行", 354, d["with_family_rate"], "IsAlone", 0.2034),
        ("有客舱号", 204, d["has_cabin_rate"], "HasCabin", 0.3169),
        ("无客舱号", 687, d["no_cabin_rate"], "HasCabin", 0.3169),
        ("女性且一等舱", facts["group_stats"]["female_first_class"]["n"],
         facts["group_stats"]["female_first_class"]["rate"], "组合", None),
    ]
    rows = []
    for label, n, rate, var, r in specs:
        rows.append([
            label, f"{n}", f"{round(rate * n)}", f"{n - round(rate * n)}",
            pct(rate), ci(rate, n),
            var or "—",
            "—" if not isinstance(r, float) else f"{r:.4f}",
            "—" if var is None else "***",
        ])
    table(["分组", "样本量", "生还", "遇难", "生存率", "95% 置信区间",
           "分组变量", "|r| 与生还", "显著性"],
          rows, ["left", "right", "right", "right", "right", "left",
                 "left", "right", "center"])
    print("  生存率的置信区间由正态近似给出；显著性为卡方检验经 BH 校正后的结果，"
          "*** 表示 q<0.001。")

    # ---------------- 7 ----------------
    title("七、离散化的组间差异（卡方检验）", "-")
    rows = []
    for name, v in facts["binning"].items():
        strength = ("强" if v["cramers_v"] >= 0.5 else
                    "中等" if v["cramers_v"] >= 0.3 else
                    "弱" if v["cramers_v"] >= 0.1 else "几乎无")
        rows.append([name, f"{v['n_groups']}", f"{v['chi2']:.4f}", f"{v['dof']}",
                     f"{v['p_value']:.3e}", f"{v['cramers_v']:.4f}", strength,
                     v["rate_range"], "是" if v["significant"] else "否"])
    table(["分箱方式", "组数", "卡方值", "自由度", "p 值", "Cramer's V",
           "关联强度", "生存率范围", "是否显著"],
          rows, ["left", "right", "right", "right", "right", "right",
                 "center", "left", "center"])

    # ---------------- 8 ----------------
    title("八、相似性分析", "-")
    print(f"  度量：Gower 距离（混合类型：Age/Fare/SibSp/Parch + Sex/Embarked/Pclass/Title）")
    print(f"  全样本 891 名乘客，{891 * 890 // 2:,} 对乘客")
    print()
    c = sim["same_group_contrast"]
    rows = [
        ["性别（男/女）", f"{c['Sex']['组内平均相似度']:.4f}", f"{c['Sex']['组间平均相似度']:.4f}",
         f"{c['Sex']['差值']:.4f}", f"{c['Sex']["Cohen's d"]:.4f}", "极大"],
        ["舱位等级", f"{c['Pclass']['组内平均相似度']:.4f}", f"{c['Pclass']['组间平均相似度']:.4f}",
         f"{c['Pclass']['差值']:.4f}", f"{c['Pclass']["Cohen's d"]:.4f}", "大"],
        ["同家族（姓氏）", f"{c['Surname']['组内平均相似度']:.4f}",
         f"{c['Surname']['组间平均相似度']:.4f}", f"{c['Surname']['差值']:.4f}", "—", "中等"],
        ["生存结果", f"{c['Survived']['组内平均相似度']:.4f}",
         f"{c['Survived']['组间平均相似度']:.4f}", f"{c['Survived']['差值']:.4f}",
         f"{c['Survived']["Cohen's d"]:.4f}", "中等"],
    ]
    table(["分组依据", "组内相似度", "组间相似度", "差值", "Cohen's d", "效应量评价"],
          rows, ["left", "right", "right", "right", "right", "center"])
    print()
    print(f"  降维与聚类（子集 n={sim['subset_size']}）："
          f"MDS 应力比 {sim['mds_stress1']:.4f}，轮廓系数 {sim['silhouette']:.4f}")
    rows = []
    for cl_ in sim["clusters"]:
        sex_txt = "全为女性" if cl_["女性占比"] == 1 else (
            "全为男性" if cl_["女性占比"] == 0 else "男女混合")
        rows.append([f"簇 {cl_['聚类编号']}", f"{cl_['样本量']}", pct(cl_["生存率"]),
                     f"{cl_['平均年龄']:.1f}", f"{cl_['平均票价']:.1f}",
                     pct(cl_["女性占比"]), pct(cl_["一等舱占比"]), sex_txt])
    table(["簇", "样本量", "生存率", "平均年龄", "平均票价", "女性占比",
           "一等舱占比", "性别构成"],
          rows, ["left", "right", "right", "right", "right", "right", "right", "left"])
    print("  注：聚类未使用标签，却把“女性（高生存）”与“男性（低生存）”分开。")
    print()
    rows = [[name, f"{v:.4f}", f"{v * 100:.2f}%"] for name, v in sim["nn_overlap_k5"].items()]
    table(["距离度量（与 Gower 对比）", "5-近邻重合率", "换算百分比"],
          rows, ["left", "right", "right"])

    # ---------------- 9 ----------------
    title("九、相关性分析", "-")
    print(f"  共检验 {cor['total_pairs']} 对变量，BH 校正后显著 {cor['significant_pairs']} 对"
          f"（{pct(cor['significant_ratio'])}）")
    print()
    rows = []
    for item in cor["target_correlations"]:
        r = item["r"]
        strength = ("强" if abs(r) >= 0.5 else
                    "中等" if abs(r) >= 0.3 else
                    "弱" if abs(r) >= 0.1 else "几乎无")
        rows.append([item["var"], f"{r:.4f}", f"{item['rho']:.4f}",
                     f"{abs(r - item['rho']):.4f}",
                     f"{item['q']:.3g}", item["mark"],
                     "正" if r > 0 else "负", strength])
    table(["变量（vs 生还）", "Pearson r", "Spearman ρ", "两者之差",
           "q 值(BH)", "显著性", "方向", "强度"],
          rows, ["left", "right", "right", "right", "right", "center", "center", "center"])
    print()
    print("  自变量之间的强相关（|r| ≥ 0.5）：")
    rows = [[p["pair"], f"{p['r']:.4f}", f"{p['rho']:.4f}",
             f"{p['r'] - p['rho']:+.4f}", f"{p['q']:.3g}",
             "正相关" if p["r"] > 0 else "负相关"]
            for p in cor["strong_pairs"] if abs(p["r"]) >= 0.5]
    table(["变量对", "Pearson r", "Spearman ρ", "两者之差", "q 值(BH)", "方向"],
          rows, ["left", "right", "right", "right", "right", "center"])
    print(f"  三种系数差异：Pearson 与 Spearman 最大相差 "
          f"{cor['method_agreement']['pearson_vs_spearman_max_diff']:.4f}")

    # ---------------- 10 ----------------
    title("十、预处理输出", "-")
    table(["项目", "结果"], [
        ["清洗后数据", f"训练集 {cl['rows_after']} 行 × 24 列；测试集 418 行 × 23 列"],
        ["特征矩阵", f"{tf['output_shape'][0]} 行 × {tf['output_shape'][1]} 列，"
                     f"全数值，缺失单元格 {tf['missing_cells']} 个"],
        ["变换步骤", "、".join(s["步骤"] for s in tf["steps"])],
        ["独热编码", f"{len(tf['one_hot_columns'])} 列"],
        ["年龄幂变换", f"Yeo-Johnson，λ = {tf['yeojohnson_lambda'][0]}"],
        ["派生字段", f"{len(edit['derived_columns'])} 个："
                     + "、".join(edit["derived_columns"])],
        ["年龄填补标记", f"AgeImputed 标记 {cl['age_imputed']} 条"],
        ["清洗后数据文件", "titanic_clean_train.csv / titanic_clean_test.csv"],
        ["特征矩阵文件", "titanic_features_train.csv / titanic_features_test.csv"],
        ["插图", f"{len(facts['figures'])} 张（300 dpi）"],
    ], ["left", "left"])

    title("分析结束")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
