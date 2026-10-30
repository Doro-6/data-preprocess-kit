# -*- coding: utf-8 -*-
"""Titanic 数据预处理实验 —— 完整分析流水线。

运行：
    python scripts/run_pipeline.py

产出：
    results/*.csv, results/*.json   —— 报告中的全部数值证据
    reports/figures/*.png            —— 报告中的全部插图（300 dpi）
    data/processed/*.csv             —— 清洗与转换后的数据集
    logs/cleaning_log.csv            —— 逐条数据溯源日志
"""

from __future__ import annotations

import json
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 保证 scripts/ 直接运行时也能导入 dpp 包
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
_LOG_HANDLE = open(LOGS / "run.log", "w", encoding="utf-8", buffering=1)

class _Tee:
    """同时输出到控制台与日志文件，便于在受限终端环境下取证。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = _Tee(sys.__stdout__, _LOG_HANDLE)
sys.stderr = _Tee(sys.__stderr__, _LOG_HANDLE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import font_manager

try:
    from dpp import cleaning, correlation, descriptive, similarity, transform
except Exception:  # noqa: BLE001
    traceback.print_exc()
    raise

# dpp 导入后就绪，后续再在 main() 内部做重活

# --------------------------------------------------------------------------
# 全局配置
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
LOGS = ROOT / "logs"

for d in (PROCESSED, RESULTS, FIGURES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# 中文字体（Windows 自带的黑体/雅黑，缺失时回退）
_available = {f.name for f in font_manager.fontManager.ttflist}
for _cand in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    if _cand in _available:
        plt.rcParams["font.sans-serif"] = [_cand]
        FONT_USED = _cand
        break
else:  # pragma: no cover
    FONT_USED = "default"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 9

PALETTE = {0: "#4C72B0", 1: "#DD8452"}
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

_STAGE_BUFFER: list[str] = []

def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)
    _STAGE_BUFFER.append(str(msg))

QUIET = "--quiet" in sys.argv or "-q" in sys.argv

# 安静模式下值得保留的关键行
_QUIET_KEEP = (
    "step ",
    "train ",
    "cleaned ",
    "feature matrix ",
    "done,",
    "wrote facts.json",
)

def log_stage(title: str) -> None:
    """阶段开始：安静模式下先收尾上一阶段，再打印一行阶段名。

    摘要写到 stderr：安静模式下 stdout 会被调用方（run_all.py）捕获，
    写 stdout 会被一起吞掉，看不到进度。
    """
    _flush_stage()
    if QUIET:
        print(f"[{title}]", file=sys.stderr, flush=True)
    else:
        print(f"[pipeline] {'=' * 72}", flush=True)
        print(f"[pipeline] step {title}", flush=True)

def _flush_stage() -> None:
    """把当前阶段收集到的关键行合并为一行输出（安静模式）。"""
    if not QUIET:
        _STAGE_BUFFER.clear()
        return
    kept = [m for m in _STAGE_BUFFER if m.startswith(_QUIET_KEEP)]
    _STAGE_BUFFER.clear()
    if not kept:
        return
    main_line = next((m for m in kept if not m.startswith("step ")), kept[-1])
    print(f"  {main_line}", file=sys.stderr, flush=True)

def log_detail(msg: str) -> None:
    if not QUIET:
        print(f"[pipeline] {msg}", flush=True)

def pipeline_summary(f: dict, clean: pd.DataFrame) -> pd.DataFrame:
    """数据处理流程的输入输出与规模变化。"""
    ing, cl, tf = f["ingest"], f["cleaning"], f["transform"]
    rows = [
        {"步骤": "1. 数据读取", "对象": "train.csv / test.csv / gender_submission.csv",
         "输入": "Kaggle 原始 CSV", "输出": f"训练集 {ing['train_shape'][0]}×{ing['train_shape'][1]}，"
                                        f"测试集 {ing['test_shape'][0]}×{ing['test_shape'][1]}",
         "关键操作": "读取并建立数据字典", "缺失单元格": "—"},
        {"步骤": "2. 录入校验", "对象": "train + test",
         "输入": "两个数据集", "输出": f"{f['integrity']['total']} 项校验",
         "关键操作": "主键唯一性、取值域、字段格式、跨文件一致性",
         "缺失单元格": f"通过 {f['integrity']['passed']} 项"},
        {"步骤": "3. 字段编辑", "对象": "train + test",
         "输入": "Name / Ticket / Cabin 复合字段",
         "输出": f"新增 {len(f['edit']['derived_columns'])} 个派生字段",
         "关键操作": "Surname、Title、Deck、票号拆分、FamilySize、IsAlone",
         "缺失单元格": "—"},
        {"步骤": "4. 数据清洗", "对象": "train",
         "输入": f"{ing['train_shape'][0]}×{ing['train_shape'][1]}",
         "输出": f"{cl['rows_after']}×24",
         "关键操作": f"年龄分组中位数填补 {cl['age_imputed']} 条、Embarked 众数填补、"
                     f"Cabin 转 HasCabin 指示变量",
         "缺失单元格": f"{cl['missing_before']} -> {cl['missing_after']}"},
        {"步骤": "5. 数据清洗", "对象": "test",
         "输入": f"{ing['test_shape'][0]}×{ing['test_shape'][1]}", "输出": "418×23",
         "关键操作": "同训练集规则；票价缺失 1 条按 Pclass × Embarked 中位数填补",
         "缺失单元格": "-> 见 04_missing_summary_total.csv"},
        {"步骤": "6. 数据转换", "对象": "train",
         "输入": f"{cl['rows_after']}×24", "输出": f"{tf['output_shape'][0]}×{tf['output_shape'][1]}",
         "关键操作": "对数变换、幂变换、标准化、二值化、独热编码、顺序编码、等频分箱",
         "缺失单元格": f"{tf['missing_cells']}（全数值）"},
        {"步骤": "7. 数据转换", "对象": "test", "输入": "418×23", "输出": "418×35",
         "关键操作": "复用训练集同一套变换定义", "缺失单元格": "0"},
        {"步骤": "8. 结果输出", "对象": "全部产物",
         "输入": "各环节输出",
         "输出": f"{len(f['figures'])} 张插图 + 结果文件",
         "关键操作": "结果落盘、插图 300 dpi 导出、清洗溯源日志",
         "缺失单元格": "—"},
    ]
    return pd.DataFrame(rows)


def save_fig(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    log_detail(f"figure -> {path.name}")

def write_table(df: pd.DataFrame, name: str, index: bool = False, float_format: str = "%.4f") -> None:
    path = RESULTS / name
    df.to_csv(path, index=index, encoding="utf-8-sig", float_format=float_format)
    log_detail(f"table  -> {path.name}  ({df.shape[0]}x{df.shape[1]})")

def write_json(obj, name: str) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log_detail(f"json   -> {path.name}")

def main() -> int:
    if QUIET:
        # 安静模式下屏蔽 numpy/scipy 的数值告警（如极端分布下缺少样本量
        # 近似导致的 Precision loss），它们会打断结果的阅读，且已在
        # results/ 结果文件中以有效数字形式体现。
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)
    facts: dict = {}
    facts["environment"] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "chinese_font": FONT_USED,
        "random_seed": SEED,
    }
    log_detail(f"environment: {facts['environment']}")

    # ======================================================================
    # 一、数据准备与录入
    log_stage("1 / data ingestion & field editing")
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    submission = pd.read_csv(RAW / "gender_submission.csv")

    facts["ingest"] = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "submission_shape": list(submission.shape),
        "train_memory_kb": round(train.memory_usage(deep=True).sum() / 1024, 1),
        "columns": list(train.columns),
    }
    log(f"train {train.shape}, test {test.shape}, submission {submission.shape}")

    # 1.1 数据质量总览（每列的类型、缺失、取值数、示例）
    overview = cleaning.column_overview(train, "Survived")
    write_table(overview, "01_column_overview.csv")

    # 1.2 录入校验：主键唯一、取值范围、字段一致性
    checks = cleaning.integrity_checks(train, label="训练集")
    checks += cleaning.integrity_checks(test, label="测试集")
    checks += cleaning.cross_file_checks(train, test, submission)
    checks_df = pd.DataFrame(checks)
    write_table(checks_df, "02_integrity_checks.csv")
    facts["integrity"] = {
        "total": len(checks_df),
        "passed": int((checks_df["结果"] == "PASS").sum()),
        "failed": int((checks_df["结果"] == "FAIL").sum()),
        "warned": int((checks_df["结果"] == "WARN").sum()),
    }
    log(f"integrity checks: {facts['integrity']}")

    # 1.3 数据编辑：从复合字段中解析出结构化信息
    train_edit = cleaning.edit_fields(train)
    test_edit = cleaning.edit_fields(test)
    derived_cols = [c for c in train_edit.columns if c not in train.columns]
    facts["edit"] = {
        "derived_columns": derived_cols,
        "title_counts": train_edit["Title"].value_counts().to_dict(),
        "deck_counts": train_edit["Deck"].fillna("Unknown").value_counts().to_dict(),
        "ticket_prefix_top": train_edit["TicketPrefix"].replace("", "NONE").value_counts().head(8).to_dict(),
    }
    log_detail(f"derived columns: {derived_cols}")

    # 图 1：原始数据概览
    save_fig(cleaning.fig_data_overview(train, test), "fig01_数据概览.png")

    # ======================================================================
    # 二、数据清洗
    log_stage("2 / data cleaning")
    cleaning_log: list[dict] = []

    # 2.1 缺失值：规模、机制与统计检验
    missing_summary = cleaning.missing_summary(train_edit)
    write_table(missing_summary, "03_missing_summary.csv")

    mcar = cleaning.missing_mechanism_tests(train_edit)
    write_table(mcar, "04_missing_mechanism.csv")
    facts["missing_mechanism"] = cleaning.missing_mechanism_conclusion(mcar)
    log(f"missing mechanism -> {facts['missing_mechanism']['verdict']}")

    save_fig(cleaning.fig_missing_overview(train_edit), "fig02_缺失值总览.png")
    save_fig(cleaning.fig_missing_mechanism(train_edit), "fig03_缺失机制检验.png")

    # 2.2 异常值：三种准则对比 + 定义域规则
    outlier_tables, outlier_facts = cleaning.outlier_analysis(train_edit)
    for name, tbl in outlier_tables.items():
        write_table(tbl, name)
    facts["outliers"] = outlier_facts
    save_fig(cleaning.fig_outliers(train_edit, outlier_facts), "fig04_异常值检测.png")

    # 2.3 执行清洗
    clean_train, train_log = cleaning.clean_dataset(train_edit, "train")
    clean_test, test_log = cleaning.clean_dataset(test_edit, "test")
    cleaning_log.extend(train_log)
    cleaning_log.extend(test_log)

    clean_train.to_csv(PROCESSED / "titanic_clean_train.csv", index=False, encoding="utf-8-sig")
    clean_test.to_csv(PROCESSED / "titanic_clean_test.csv", index=False, encoding="utf-8-sig")
    log(f"cleaned train {clean_train.shape}, test {clean_test.shape}")

    clean_missing = cleaning.missing_summary(clean_train)
    write_table(clean_missing, "05_missing_after_clean.csv")
    facts["cleaning"] = {
        "missing_before": int(train_edit.isna().sum().sum()),
        "missing_after": int(clean_train.isna().sum().sum()),
        "rows_before": int(len(train_edit)),
        "rows_after": int(len(clean_train)),
        "duplicates_removed": int(len(train_edit) - len(clean_train)),
        "age_imputed": int(train_edit["Age"].isna().sum()),
        "cabin_filled": int((train_edit["Cabin"].isna() & clean_train["Cabin"].notna()).sum()),
    }
    save_fig(cleaning.fig_cleaning_effect(train_edit, clean_train), "fig05_清洗前后对比.png")

    # ======================================================================
    # 三、数据转换
    log_stage("3 / data transformation")
    transformed, transform_facts = transform.build_feature_matrix(clean_train)
    transformed_test, _ = transform.build_feature_matrix(clean_test)

    write_json(transform_facts, "06_transform_summary.json")
    facts["transform"] = transform_facts

    shape_report = transform.skew_report(clean_train, transformed)
    write_table(shape_report, "07_skew_before_after.csv")
    save_fig(transform.fig_transformations(clean_train), "fig06_数据变换效果.png")

    encoding_tbl = transform.encoding_demo(clean_train)
    write_table(encoding_tbl, "08_encoding_schemes.csv")
    save_fig(transform.fig_encoding(clean_train), "fig07_编码方案.png")

    binned = transform.discretize(clean_train)
    bin_table, bin_test = transform.binning_analysis(binned)
    write_table(bin_table, "09_binning_survival.csv")
    facts["binning"] = bin_test
    save_fig(transform.fig_binning(bin_table), "fig08_分箱分析.png")

    transformed.to_csv(PROCESSED / "titanic_features_train.csv", index=False, encoding="utf-8-sig")
    transformed_test.to_csv(PROCESSED / "titanic_features_test.csv", index=False, encoding="utf-8-sig")
    log(f"feature matrix train {transformed.shape}, test {transformed_test.shape}")

    # ======================================================================
    # 四、描述性统计
    log_stage("4 / descriptive statistics")
    num_desc = descriptive.numeric_description(clean_train)
    write_table(num_desc, "10_descriptive_numeric.csv")

    cat_desc = descriptive.categorical_description(clean_train)
    write_table(cat_desc, "11_descriptive_categorical.csv")

    shape_stats = descriptive.shape_statistics(clean_train)
    write_table(shape_stats, "12_shape_statistics.csv")

    grouped = descriptive.grouped_survival(clean_train)
    write_table(grouped, "13_grouped_survival.csv")

    cross_grouped = descriptive.group_summary(clean_train)
    write_table(cross_grouped, "13b_cross_grouped_survival.csv")
    facts["descriptive"] = descriptive.key_facts(clean_train, shape_stats, grouped)

    ci_tbl = descriptive.bootstrap_ci(clean_train, n_boot=5000, seed=SEED)
    write_table(ci_tbl, "14_rate_confidence_intervals.csv")

    save_fig(descriptive.fig_distributions(clean_train), "fig09_变量分布.png")
    save_fig(descriptive.fig_grouped_survival(grouped), "fig10_分组生存率.png")
    save_fig(descriptive.fig_shape_diagnostics(clean_train), "fig11_分布形态诊断.png")

    # ======================================================================
    # 五、相似性分析
    log_stage("5 / similarity analysis")
    sim_matrix, sim_df = similarity.gower_similarity(clean_train)
    sim_facts = {}
    sim_facts["gower_top_pairs"] = similarity.top_pairs(sim_df, k=10)
    sim_facts["neighbors"] = similarity.neighbor_examples(sim_df, clean_train, k=5)
    sim_facts["same_group_contrast"] = similarity.group_similarity_contrast(clean_train, sim_matrix)

    # 五种距离在同一子集上的对比 + 投影 + 聚类
    dist_cmp, ordination, dist_facts, dist_mats, linkage_Z = similarity.distance_comparison(
        clean_train, sim_matrix)
    write_table(dist_cmp, "15_distance_metrics.csv")
    write_table(ordination, "16_ordination_coordinates.csv")
    sim_facts.update(dist_facts)
    facts["similarity"] = sim_facts

    save_fig(similarity.fig_similarity_heatmap(sim_matrix, clean_train), "fig12_相似度矩阵.png")
    save_fig(similarity.fig_ordination(ordination, linkage_Z,
                                       silhouette=dist_facts.get("silhouette")), "fig13_降维投影.png")
    save_fig(similarity.fig_neighbors(sim_df, clean_train), "fig14_近邻画像.png")

    # ======================================================================
    # 六、相关性分析
    log_stage("6 / correlation analysis")
    corr_num = correlation.numeric_correlation(clean_train, transformed)
    write_table(corr_num, "17_correlation_matrix.csv", index=True)

    pairwise = correlation.pairwise_table(clean_train)
    write_table(pairwise, "18_correlation_tests.csv")
    facts["correlation"] = correlation.correlation_facts(pairwise, clean_train)

    vif = correlation.vif_table(transformed)
    write_table(vif, "19_vif.csv")

    assoc = correlation.association_measures(clean_train)
    write_table(assoc, "20_association_measures.csv")

    save_fig(correlation.fig_correlation_heatmaps(clean_train), "fig15_相关性热力图.png")
    save_fig(correlation.fig_significant_correlations(pairwise), "fig16_显著相关条形图.png")
    save_fig(correlation.fig_correlation_scatter(clean_train), "fig17_相关散点矩阵.png")

    # ======================================================================
    # 七、溯源日志与汇总
    log_stage("7 / lineage & summary")
    log_df = pd.DataFrame(cleaning_log)
    write_table(log_df, "21_cleaning_log.csv")

    # 相似性对比表（组内 vs 组间）
    contrast = pd.DataFrame([
        {"分组依据": name, **vals}
        for name, vals in sim_facts["same_group_contrast"].items()
    ])
    write_table(contrast, "24_similarity_contrast.csv")

    # 最近邻画像
    write_table(pd.DataFrame(sim_facts["neighbors"]), "25_neighbor_examples.csv")

    # 各分组的样本量与"女性且一等舱"组合的生存率
    combo = clean_train.loc[(clean_train["Sex"] == "female") & (clean_train["Pclass"] == 1), "Survived"]
    facts["group_stats"] = {
        "sizes": {
            "female": int((clean_train["Sex"] == "female").sum()),
            "male": int((clean_train["Sex"] == "male").sum()),
            "pclass1": int((clean_train["Pclass"] == 1).sum()),
            "pclass2": int((clean_train["Pclass"] == 2).sum()),
            "pclass3": int((clean_train["Pclass"] == 3).sum()),
            "alone": int((clean_train["IsAlone"] == 1).sum()),
            "with_family": int((clean_train["IsAlone"] == 0).sum()),
            "has_cabin": int((clean_train["HasCabin"] == 1).sum()),
            "no_cabin": int((clean_train["HasCabin"] == 0).sum()),
        },
        "female_first_class": {
            "n": int(len(combo)),
            "rate": round(float(combo.mean()), 4),
        },
    }

    facts["figures"] = sorted(p.name for p in FIGURES.glob("*.png"))

    # 关键群体生存率汇总
    write_table(descriptive.key_survival_rates(clean_train), "26_key_survival_rates.csv")

    # 各变量与生还的相关强度排序
    write_table(
        pd.DataFrame([
            {
                "变量": item["var"],
                "Pearson r": item["r"],
                "Spearman rho": item["rho"],
                "q 值(BH)": item["q"],
                "显著性": item["mark"],
                "方向": "正" if item["r"] > 0 else "负",
                "强度": ("强" if abs(item["r"]) >= 0.5 else
                         "中等" if abs(item["r"]) >= 0.3 else
                         "弱" if abs(item["r"]) >= 0.1 else "几乎无"),
            }
            for item in facts["correlation"]["target_correlations"]
        ]),
        "27_target_correlations.csv",
    )

    # 处理流程摘要
    write_table(pipeline_summary(facts, clean_train), "28_pipeline_summary.csv")

    facts["tables"] = sorted(p.name for p in RESULTS.glob("*.csv"))
    write_json(facts, "facts.json")
    log(f"wrote facts.json with {len(facts)} top-level keys")
    _flush_stage()
    return 0

if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    print(f"[pipeline] done, exit={code}", flush=True)
    raise SystemExit(code)
