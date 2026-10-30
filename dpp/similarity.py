# -*- coding: utf-8 -*-
"""相似性与距离分析。

混合类型数据的相似性度量以 Gower 距离为核心，并与欧氏、曼哈顿、余弦、
Jaccard 等度量在同一子集上做一致性对比；最后用 MDS/PCA 投影与层次聚类
把“相似性”落到可视化的分组结构上。
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# 参与相似性计算的特征（数值型 + 分类型混合）
GOWER_NUMERIC = ["Age", "Fare", "SibSp", "Parch"]
GOWER_CATEGORICAL = ["Sex", "Embarked", "Pclass", "Title"]


# ==========================================================================
# 1. Gower 距离
# ==========================================================================
def gower_matrix(df: pd.DataFrame,
                 numeric: list[str] | None = None,
                 categorical: list[str] | None = None) -> np.ndarray:
    """计算 Gower 距离矩阵。

    数值型：|x_i - x_j| / (max - min)；分类型：相同取 0，不同取 1。
    最终距离为各特征贡献的等权平均，因此量纲不同也可直接比较。
    """
    numeric = numeric or GOWER_NUMERIC
    categorical = categorical or GOWER_CATEGORICAL
    n = len(df)
    total = np.zeros((n, n))
    n_features = 0

    for col in numeric:
        v = df[col].to_numpy(dtype=float)
        rng = np.nanmax(v) - np.nanmin(v)
        if rng == 0:
            continue
        total += np.abs(v[:, None] - v[None, :]) / rng
        n_features += 1

    for col in categorical:
        codes = pd.Categorical(df[col]).codes.astype(float)
        total += (codes[:, None] != codes[None, :]).astype(float)
        n_features += 1

    dist = total / max(n_features, 1)
    np.fill_diagonal(dist, 0.0)
    return dist


def gower_similarity(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """返回相似度矩阵 S = 1 - D，以及带 PassengerId 标签的相似度表。"""
    dist = gower_matrix(df)
    sim = 1.0 - dist
    sim_df = pd.DataFrame(sim, index=df["PassengerId"].to_numpy(), columns=df["PassengerId"].to_numpy())
    sim_df.index.name = "PassengerId"
    return sim, sim_df


def top_pairs(sim_df: pd.DataFrame, k: int = 10) -> list[dict]:
    """相似度最高的若干乘客对（去除重复对与自身配对）。"""
    m = sim_df.to_numpy().copy()
    np.fill_diagonal(m, -np.inf)
    idx = np.dstack(np.unravel_index(np.argsort(-m, axis=None)[:k * 3], m.shape))[0]
    seen, rows = set(), []
    ids = sim_df.index.to_numpy()
    for i, j in idx:
        key = tuple(sorted((int(ids[i]), int(ids[j]))))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"乘客A": key[0], "乘客B": key[1], "Gower 相似度": round(float(m[i, j]), 4)})
        if len(rows) == k:
            break
    return rows


def neighbor_examples(sim_df: pd.DataFrame, df: pd.DataFrame, k: int = 5,
                      anchors: list[int] | None = None) -> list[dict]:
    """给定若干“锚点”乘客，列出其最相似的 k 位乘客及其生存情况。"""
    anchors = anchors or [1, 2, 3, 5, 10, 100]
    id_to_row = {int(pid): i for i, pid in enumerate(df["PassengerId"].to_numpy())}
    rows = []
    for pid in anchors:
        if pid not in sim_df.index:
            continue
        sims = sim_df.loc[pid].drop(index=pid).sort_values(ascending=False)
        anchor = df.iloc[id_to_row[pid]]
        rec = {
            "锚点 PassengerId": pid,
            "锚点姓名": str(anchor["Name"])[:26],
            "锚点特征": f"{anchor['Sex']}/{int(anchor['Pclass'])}等舱/{anchor['Age']:.0f}岁",
            "锚点是否生还": int(anchor["Survived"]),
        }
        for rank, (nid, s) in enumerate(sims.head(k).items(), start=1):
            nb = df.iloc[id_to_row[int(nid)]]
            rec[f"近邻{rank}"] = (f"#{int(nid)} {str(nb['Name'])[:16]} "
                                  f"({nb['Sex'][0]}/{int(nb['Pclass'])}等/{nb['Age']:.0f}岁)")
            rec[f"相似度{rank}"] = round(float(s), 4)
            rec[f"近邻{rank}生还"] = int(nb["Survived"])
        rec["近邻生还比例"] = round(float(np.mean([rec[f"近邻{i}生还"] for i in range(1, k + 1)])), 4)
        rows.append(rec)
    return rows


def group_similarity_contrast(df: pd.DataFrame, sim: np.ndarray) -> dict:
    """组内 vs 组间相似度：同性别、同舱位、同姓氏的相似度是否更高。"""
    facts: dict = {}
    iu = np.triu_indices(len(df), k=1)
    for col in ["Sex", "Pclass", "Survived", "Surname"]:
        codes = pd.Categorical(df[col]).codes
        same = codes[:, None] == codes[None, :]
        same_v = sim[iu][same[iu]]
        diff_v = sim[iu][~same[iu]]
        if len(same_v) < 2 or len(diff_v) < 2:
            continue
        t, p = stats.ttest_ind(same_v, diff_v, equal_var=False)
        pooled = np.sqrt((same_v.var(ddof=1) + diff_v.var(ddof=1)) / 2)
        facts[col] = {
            "组内平均相似度": round(float(same_v.mean()), 4),
            "组间平均相似度": round(float(diff_v.mean()), 4),
            "差值": round(float(same_v.mean() - diff_v.mean()), 4),
            "Cohen's d": round(float((same_v.mean() - diff_v.mean()) / pooled), 4) if pooled else None,
            "t 检验 p 值": float(p),
            "显著": bool(p < 0.05),
        }
    return facts


# ==========================================================================
# 2. 多种距离度量的一致性对比
# ==========================================================================
def distance_comparison(df: pd.DataFrame, sim: np.ndarray, n_subset: int = 220, seed: int = 42):
    """在同一子集上比较 5 种距离度量，并给出 MDS 投影与聚类结构。"""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=min(n_subset, len(df)), replace=False))
    sub = df.iloc[idx].reset_index(drop=True)

    # 数值特征标准化后计算欧氏/曼哈顿/余弦距离
    X = StandardScaler().fit_transform(sub[["Age", "Fare", "SibSp", "Parch", "FamilySize"]].astype(float))
    d_euclid = squareform(pdist(X, metric="euclidean"))
    d_manhattan = squareform(pdist(X, metric="cityblock"))
    d_cosine = squareform(pdist(X, metric="cosine"))

    # 混合类型：Gower 与 Jaccard（特征二值化后）
    d_gower = gower_matrix(sub)
    B = pd.concat([
        pd.get_dummies(sub[["Sex", "Embarked", "Pclass", "Title"]], dtype=int),
        pd.cut(sub["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False).rename("AgeBin"),
        pd.qcut(sub["Fare"], 4, labels=False, duplicates="drop").rename("FareBin"),
    ], axis=1).astype(bool).to_numpy()
    d_jaccard = squareform(pdist(B.astype(float), metric="jaccard"))

    dists = {
        "Gower（混合类型）": d_gower,
        "欧氏（标准化数值）": d_euclid / d_euclid.max(),
        "曼哈顿（标准化数值）": d_manhattan / d_manhattan.max(),
        "余弦（标准化数值）": d_cosine / d_cosine.max(),
        "Jaccard（二值化）": d_jaccard,
    }

    iu = np.triu_indices(len(sub), k=1)
    base = dists["Gower（混合类型）"][iu]
    rows = []
    for name, d in dists.items():
        vals = d[iu]
        rho, p = stats.spearmanr(base, vals)
        overlap = None if name.startswith("Gower") else _nn_overlap(d_gower, d, k=5)
        rows.append({
            "距离度量": name,
            "平均距离": round(float(vals.mean()), 4),
            "标准差": round(float(vals.std(ddof=1)), 4),
            "与 Gower 的 Spearman ρ": round(float(rho), 4),
            "ρ 的 p 值": float(p),
            "5-近邻与 Gower 的重合率": None if overlap is None else round(overlap, 4),
            "一致性评价": "基准度量" if overlap is None else
                          ("高度一致" if rho > 0.8 else "中等一致" if rho > 0.5 else "排序差异明显"),
        })

    # ---- 投影与聚类 ----
    x_pca = StandardScaler().fit_transform(
        sub[["Age", "Fare", "SibSp", "Parch", "FamilySize"]].astype(float))
    pca = PCA(n_components=2, random_state=seed).fit(x_pca)
    coords_pca = pca.transform(x_pca)

    mds = _fit_mds(d_gower, seed)
    with warnings.catch_warnings():
        # scikit-learn 1.10 起 init 的默认值会变化，此处显式依赖当前默认行为
        warnings.simplefilter("ignore", FutureWarning)
        coords_mds = mds.fit_transform(d_gower)
    # Kruskal 应力比：把原始应力归一化，便于判断二维投影的保真度
    stress1 = float(np.sqrt(mds.stress_ / (np.sum(d_gower ** 2) / 2)))

    Z = linkage(squareform(d_gower, checks=False), method="average")
    labels = fcluster(Z, t=4, criterion="maxclust")
    sil = _silhouette(d_gower, labels)

    ordination = pd.DataFrame({
        "PassengerId": sub["PassengerId"].to_numpy(),
        "MDS1": coords_mds[:, 0],
        "MDS2": coords_mds[:, 1],
        "PCA1": coords_pca[:, 0],
        "PCA2": coords_pca[:, 1],
        "聚类编号": labels,
        "Survived": sub["Survived"].to_numpy(),
        "Pclass": sub["Pclass"].to_numpy(),
        "Sex": sub["Sex"].to_numpy(),
        "Age": sub["Age"].to_numpy(),
        "Fare": sub["Fare"].to_numpy(),
    })

    cluster_profile = ordination.groupby("聚类编号").agg(
        样本量=("PassengerId", "size"),
        生存率=("Survived", "mean"),
        平均年龄=("Age", "mean"),
        平均票价=("Fare", "mean"),
        一等舱占比=("Pclass", lambda s: float((s == 1).mean())),
        女性占比=("Sex", lambda s: float((s == "female").mean())),
    ).round(4).reset_index()

    facts = {
        "subset_size": int(len(sub)),
        "mds_stress": round(float(mds.stress_), 4),
        "mds_stress1": round(stress1, 4),
        "silhouette": None if sil != sil else round(float(sil), 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "clusters": cluster_profile.to_dict("records"),
        "nn_overlap_k5": {name: round(_nn_overlap(d_gower, d, k=5), 4)
                          for name, d in dists.items() if not name.startswith("Gower")},
    }
    return pd.DataFrame(rows), ordination, facts, dists, Z


def _fit_mds(dist: np.ndarray, seed: int):
    """构造 MDS 对象；屏蔽 scikit-learn 1.10 前后的参数改名告警。"""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:  # scikit-learn >= 1.10 用 metric
            return MDS(n_components=2, metric="precomputed", random_state=seed,
                       n_init=4, max_iter=300, normalized_stress="auto")
        except TypeError:  # scikit-learn 1.4–1.9 用 dissimilarity
            return MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
                       n_init=4, max_iter=300, normalized_stress="auto")


def _silhouette(dist: np.ndarray, labels: np.ndarray):
    """在预计算距离上计算轮廓系数；失败时返回 NaN。"""
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:  # noqa: BLE001
        return float("nan")


def _nn_overlap(d1: np.ndarray, d2: np.ndarray, k: int = 5) -> float:
    """两种距离下 k 近邻集合的平均 Jaccard 重合率。"""
    a = np.argsort(d1, axis=1)[:, 1:k + 1]
    b = np.argsort(d2, axis=1)[:, 1:k + 1]
    scores = []
    for ra, rb in zip(a, b):
        sa, sb = set(ra.tolist()), set(rb.tolist())
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


# ==========================================================================
# 3. 插图
# ==========================================================================
def fig_similarity_heatmap(sim: np.ndarray, df: pd.DataFrame, n: int = 40):
    """相似度热力图（按舱位/性别排序）与全样本距离分布。"""
    order = df.sort_values(["Pclass", "Sex", "Age"]).index.to_numpy()
    sel = order[:n]
    sub = sim[np.ix_(sel, sel)]
    labels = [f"#{int(df.loc[i, 'PassengerId'])}" for i in sel]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    im = ax.imshow(sub, cmap="viridis", vmin=sub.min(), vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n), labels, fontsize=5)
    ax.set_title(f"(a) 前 {n} 名乘客的 Gower 相似度矩阵（按舱位 -> 性别排序）", fontsize=9)
    bounds = np.cumsum(df.loc[sel, "Pclass"].value_counts().sort_index().to_numpy())
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="white", lw=1.0, ls="--")
        ax.axvline(b - 0.5, color="white", lw=1.0, ls="--")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="相似度")

    ax = axes[1]
    dist = 1 - sim
    iu = np.triu_indices(len(df), k=1)
    vals = dist[iu]
    ax.hist(vals, bins=60, color="#4C72B0", alpha=0.85)
    for q, c, name in [(0.25, "#55A868", "P25"), (0.5, "#DD8452", "中位数"), (0.75, "#C44E52", "P75")]:
        v = np.quantile(vals, q)
        ax.axvline(v, color=c, ls="--", lw=1.2, label=f"{name} {v:.3f}")
    ax.set_xlabel("Gower 距离")
    ax.set_ylabel("乘客对数")
    ax.set_title(f"(b) 全样本 {len(vals):,} 对乘客的距离分布", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_ordination(ordination: pd.DataFrame, Z: np.ndarray, n_dendro: int = 60, silhouette=None):
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    for p, c in zip([1, 2, 3], ["#4C72B0", "#DD8452", "#55A868"]):
        m = ordination["Pclass"] == p
        ax.scatter(ordination.loc[m, "MDS1"], ordination.loc[m, "MDS2"], s=9, alpha=0.75,
                   color=c, label=f"{p} 等舱")
    ax.set_title("(a) Gower 距离的 MDS 二维投影（按舱位着色）", fontsize=9)
    ax.set_xlabel("MDS 维度 1")
    ax.set_ylabel("MDS 维度 2")
    ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    for s, c, name in [(0, "#4C72B0", "遇难"), (1, "#DD8452", "生还")]:
        m = ordination["Survived"] == s
        ax.scatter(ordination.loc[m, "MDS1"], ordination.loc[m, "MDS2"], s=9, alpha=0.75,
                   color=c, label=name)
    ax.set_title("(b) 同一投影按生存结果着色", fontsize=9)
    ax.set_xlabel("MDS 维度 1")
    ax.set_ylabel("MDS 维度 2")
    ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    prof = ordination.groupby("聚类编号").agg(
        生存率=("Survived", "mean"), 平均年龄=("Age", "mean"),
        女性占比=("Sex", lambda s: float((s == "female").mean())),
        一等舱占比=("Pclass", lambda s: float((s == 1).mean())),
        样本量=("PassengerId", "size"))
    metrics = ["生存率", "女性占比", "一等舱占比"]
    x = np.arange(len(prof))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * 0.26, prof[m] * 100, 0.24, label=m,
                      color=["#DD8452", "#4C72B0", "#55A868"][i])
        for b, v in zip(bars, prof[m] * 100):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.2)
    ax.set_xticks(x, [f"簇 {int(i)}\n(n={int(v)})" for i, v in zip(prof.index, prof["样本量"])], fontsize=7.5)
    ax.set_ylabel("比例 (%)")
    ax.set_ylim(0, 112)
    sil_txt = f"，轮廓系数 {silhouette:.3f}" if silhouette is not None and silhouette == silhouette else ""
    ax.set_title(f"(c) 层次聚类（平均连接）所得 4 个簇的画像{sil_txt}", fontsize=9)
    ax.legend(fontsize=6.8, frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color="#888888", truncate_mode="lastp", p=n_dendro)
    ax.set_title("(d) Gower 距离上的层次聚类树（截断显示）", fontsize=9)
    ax.set_xlabel("簇 / 乘客")
    ax.set_ylabel("合并距离")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    return fig


def fig_neighbors(sim_df: pd.DataFrame, df: pd.DataFrame, anchors: list[int] | None = None, k: int = 5):
    """近邻画像：平行坐标图对比锚点与其最近邻的特征。"""
    anchors = anchors or [1, 2, 5]
    dims = ["Pclass", "Age", "Fare", "SibSp", "Parch"]
    id_to_row = {int(pid): i for i, pid in enumerate(df["PassengerId"].to_numpy())}
    fig, axes = plt.subplots(1, len(anchors), figsize=(11.5, 3.6), sharey=True)
    for ax, pid in zip(np.atleast_1d(axes), anchors):
        sims = sim_df.loc[pid].drop(index=pid).sort_values(ascending=False)
        neighbors = [pid] + [int(i) for i in sims.head(k).index]
        norm = df[dims].astype(float)
        norm = (norm - norm.min()) / (norm.max() - norm.min())
        surv_flags = []
        for rank, nid in enumerate(neighbors):
            row = norm.iloc[id_to_row[nid]].to_numpy()
            surv = int(df.iloc[id_to_row[nid]]["Survived"])
            if rank > 0:
                surv_flags.append(surv)
            if rank == 0:
                ax.plot(range(len(dims)), row, "-o", color="#111111", lw=2.0, ms=4,
                        label="锚点", zorder=5)
            else:
                ax.plot(range(len(dims)), row, "-o", lw=1.0, ms=3,
                        color="#DD8452" if surv == 1 else "#4C72B0", alpha=0.85,
                        label=("近邻·生还" if surv == 1 else "近邻·遇难") if rank == 1 else None)
        ax.set_xticks(range(len(dims)), dims, fontsize=7)
        anchor_surv = int(df.iloc[id_to_row[pid]]["Survived"])
        ax.set_title(f"锚点 #{pid}（{'生还' if anchor_surv == 1 else '遇难'}）及其 {k} 个最近邻\n"
                     f"近邻中生还 {int(np.sum(surv_flags))}/{k}", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("归一化取值")
            ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig
