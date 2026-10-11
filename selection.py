# -*- coding: utf-8 -*-
"""v6 选标：无元数据簇覆盖选标，一次性生成冻结标注池 `outputs/v6_round1/selection/selection.json`。

协议见 `docs/v6协议草案.md` §2 与 `../agents.md`。五步规则，全程只用无标签信号：
1. 目标信号 → 12 kHz → 512 点不重叠切窗 → `logspec512`（256 维）→ 文件内窗口均值 →
   DE 与 FE 拼接 512 维；
2. 短记录过滤：窗口数 < 0.5 × 全体中位数的文件不进标注池（照常进测试集）；
3. 分簇：候选集上按维 z-score → k-means 搜 `k∈[6,20]`，取最小的满足
   `silhouette(k) ≥ 0.95 × max silhouette` 的 `k`（`n_init=25`、`random_state=0`）；
4. 每簇取 medoid（到本簇簇心欧氏距离最小的真实文件），单例簇排在名额序列最后；
5. 预算 `K = Σ_j max(1, round(p·n_j))`，本轮 `p=0` 即每非单例簇 1 条、`K=k`。

`类/尺寸/位置` 只在池落定之后进入 `diagnostics`，任何判断都不读它，
下面的函数签名也不接受元数据入参。选标不加载任何权重、不写第二个产物。
"""
from pathlib import Path
import argparse
import datetime
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import dataset as ds
import preprocessing as pp

#: 冻结的选标参数（训练前写死，见协议 §9）。
CHANNEL_ORDER = ("DE", "FE")
K_SEARCH = tuple(range(6, 21))
SILHOUETTE_TOLERANCE = 0.95
SHORT_RECORD_RATIO = 0.5
KMEANS_N_INIT = 25
KMEANS_RANDOM_STATE = 0
LABEL_RATIO_P = 0.0
EPS = 1e-12


def file_level_features(records, file_keys, config=None):
    """按协议第 1 步构造文件级特征矩阵：返回 `(keys, matrix)`，matrix 形状 (n, 256×通道数)。

    每个文件每个通道：切窗 → `extract_logspec` → 沿窗口取均值；DE 段在前、FE 段在后。
    只读信号，不读文件名以外的标签信息。
    """
    cfg = config or pp.build_repr_config()
    by_key = {record["file_key"]: record for record in records}
    keys = [str(key) for key in file_keys]
    parts = []
    for channel in CHANNEL_ORDER:
        rows = []
        for key in keys:
            # 每个文件单独切窗：候选集内窗口数不同，不能整批 reshape 后按文件求均值。
            segments = ds.build_segments([by_key[key]], [key], channel, segment_length=int(cfg["seg_len"]))
            features = pp.extract_logspec(segments.signals, cfg)
            rows.append(features.reshape(features.shape[0], -1).mean(axis=0).astype(np.float64))
        parts.append(np.stack(rows))
    return keys, np.concatenate(parts, axis=1)


def short_record_exclusion(records, file_keys):
    """按协议第 2 步给出短记录过滤结果：`(候选键列表, 过滤明细, 中位窗口数)`。

    窗口数取 DE 通道在 12 kHz / 512 点口径下的完整窗数，与特征构建的窗口数一致。
    """
    by_key = {record["file_key"]: record for record in records}
    counts = {str(key): ds.window_count(by_key[key], "DE") for key in file_keys}
    median = float(np.median(np.asarray(list(counts.values()), dtype=np.float64)))
    threshold = SHORT_RECORD_RATIO * median
    candidates = [key for key in sorted(counts) if counts[key] >= threshold]
    excluded = [{"file_key": key, "windows": counts[key]}
                for key in sorted(counts) if counts[key] < threshold]
    return candidates, excluded, median, threshold


def zscore(matrix):
    """候选集上按维 z-score（`ddof=0`，标准差加 EPS 防零方差维）。"""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    return (matrix - mean) / (std + EPS)


def silhouette_curve(z_matrix):
    """k∈[6,20] 的 silhouette 曲线与 k-means 结果缓存：返回 `(曲线, {k: (labels, centers)})`。"""
    curve, fitted = {}, {}
    for k in K_SEARCH:
        model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
        labels = model.fit_predict(z_matrix)
        curve[int(k)] = float(silhouette_score(z_matrix, labels))
        fitted[int(k)] = (labels, model.cluster_centers_)
    return curve, fitted


def select_k(curve, tolerance=SILHOUETTE_TOLERANCE):
    """取最小的满足 `silhouette(k) ≥ tolerance × max silhouette` 的 `k`。"""
    k_max = max(curve, key=lambda k: (curve[k], -k))
    target = tolerance * curve[k_max]
    k_selected = min(k for k in sorted(curve) if curve[k] >= target)
    return k_selected, k_max, target


def cluster_tables(z_matrix, keys, labels, centers):
    """每簇的 medoid、到簇心距离与孤立度；返回普通字典列表，供名额规则使用。

    medoid = 本簇内到簇心欧氏距离最小的真实文件；同分时按 `file_key` 升序取第一个
    （`keys` 已按 `file_key` 排序，故 `argmin` 天然给出该 tie-break）。
    孤立度 = 该文件到最近邻候选文件的欧氏距离。
    """
    centroids = np.linalg.norm(z_matrix - centers[labels], axis=1)
    neighbors = NearestNeighbors(n_neighbors=min(2, len(keys))).fit(z_matrix)
    distances = neighbors.kneighbors(z_matrix, return_distance=True)[0]
    isolation = distances[:, 1] if distances.shape[1] > 1 else np.zeros(len(keys))
    rows = []
    for index, key in enumerate(keys):
        rows.append({
            "file_key": key,
            "cluster": int(labels[index]),
            "distance_to_centroid": float(centroids[index]),
            "isolation": float(isolation[index]),
        })
    medoids = {}
    for cluster in sorted({row["cluster"] for row in rows}):
        members = [row for row in rows if row["cluster"] == cluster]
        medoids[cluster] = min(members, key=lambda row: row["distance_to_centroid"])
    return rows, medoids


def slot_order(medoids, sizes):
    """名额序列：非单例簇按簇号在前，单例簇一律排在最后（协议第 4 步）。"""
    return sorted(medoids, key=lambda cluster: (sizes[cluster] > 1, cluster))


def budget(sizes, p=LABEL_RATIO_P):
    """`K = Σ_j max(1, round(p·n_j))`，单例簇只在名额有余时回填（本轮 `p=0` → 非单例簇各 1）。"""
    targets = {cluster: max(0 if size == 1 else 1, int(round(p * size))) for cluster, size in sizes.items()}
    return sum(targets.values()), targets


def run_source(records, target_source):
    """跑完一个来源的五步规则，返回可直接 JSON 序列化的结果块。"""
    source_records = [record for record in records if record["dataset"] == target_source]
    all_keys = sorted(record["file_key"] for record in source_records)
    candidates, excluded, median_windows, threshold = short_record_exclusion(source_records, all_keys)
    keys, matrix = file_level_features(source_records, candidates)
    z_matrix = zscore(matrix)
    curve, fitted = silhouette_curve(z_matrix)
    k_selected, k_argmax, target_value = select_k(curve)
    labels, centers = fitted[k_selected]
    rows, medoids = cluster_tables(z_matrix, keys, labels, centers)
    sizes = {cluster: int(np.sum(labels == cluster)) for cluster in sorted(medoids)}
    K, per_cluster_slots = budget(sizes)
    order = slot_order(medoids, sizes)
    pool_rows = [medoids[cluster] for cluster in order[:K]]
    pool = [row["file_key"] for row in pool_rows]
    test = sorted(set(all_keys) - set(pool))
    return {
        "target_source": target_source,
        "n_target_files": len(all_keys),
        "short_record_filter": {
            "rule": "DE 窗口数 < 0.5 × 全体中位数（只约束进池，不约束评估）",
            "median_windows": median_windows,
            "threshold": threshold,
            "excluded": excluded,
            "n_candidates": len(candidates),
        },
        "silhouette_curve": curve,
        "k_selected": k_selected,
        "k_at_max_silhouette": k_argmax,
        "silhouette_at_k_selected": curve[k_selected],
        "silhouette_max": curve[k_argmax],
        "silhouette_target": target_value,
        "tolerance": SILHOUETTE_TOLERANCE,
        "k_search": list(K_SEARCH),
        "kmeans": {"n_init": KMEANS_N_INIT, "random_state": KMEANS_RANDOM_STATE},
        "cluster_sizes": sizes,
        "singleton_clusters": int(sum(1 for size in sizes.values() if size == 1)),
        "label_ratio_p": LABEL_RATIO_P,
        "K": K,
        "cluster_slots": per_cluster_slots,
        "slot_sequence": [
            {"cluster": cluster,
             "size": sizes[cluster],
             "slots": per_cluster_slots[cluster],
             "medoid": medoids[cluster]["file_key"],
             "medoid_distance_to_centroid": medoids[cluster]["distance_to_centroid"],
             "medoid_isolation": medoids[cluster]["isolation"]}
            for cluster in order
        ],
        "pool": pool,
        "cluster_labels": {row["file_key"]: row["cluster"] for row in rows},
        "isolation": {row["file_key"]: row["isolation"] for row in rows},
        "distance_to_centroid": {row["file_key"]: row["distance_to_centroid"] for row in rows},
        "test": test,
        "diagnostics": diagnostics(source_records, pool, test),
    }


def diagnostics(source_records, pool, test):
    """池落定之后才计算元数据（类/尺寸/位置、覆盖计数），不参与任何选标判断。"""
    by_key = {record["file_key"]: record for record in source_records}
    groups = {ds.fault_group(key): key for key in sorted(by_key)}
    layer_names = {}
    for group in groups:
        label, size, position = group
        layer_names[f"{label}/{size}/{position}"] = ds.LABEL_NAMES[label] + f"@{size}" + (
            f"@{position}" if position else "")
    pool_layers = sorted({ds.fault_group(key)[:2] for key in pool})
    all_layers = sorted({ds.fault_group(key)[:2] for key in by_key})
    pool_groups = sorted({ds.fault_group(key) for key in pool})
    all_groups = sorted(groups)
    test_groups = sorted({ds.fault_group(key) for key in test})
    return {
        "note": "以下元数据仅用于诊断与事后分析，选标路径未读取",
        "pool_metadata": {key: {"class_name": ds.LABEL_NAMES[int(by_key[key]["label"])],
                                "label": int(by_key[key]["label"]),
                                "size": ds.fault_group(key)[1],
                                "position": ds.fault_group(key)[2]}
                          for key in pool},
        "layer_names": layer_names,
        "pool_class_counts": {ds.LABEL_NAMES[i]: int(sum(ds.fault_group(key)[0] == i for key in pool))
                              for i in (1, 2, 3)},
        "coverage_class_size": {"covered": len(pool_layers), "total": len(all_layers),
                                "cells": [f"{ds.LABEL_NAMES[l]}@{s}" for l, s in pool_layers]},
        "coverage_class_size_position": {"covered": len(pool_groups), "total": len(all_groups),
                                         "cells": [f"{l}/{s}/{p}" for l, s, p in pool_groups]},
        "test_class_counts": {ds.LABEL_NAMES[i]: int(sum(by_key[key]["label"] == i for key in test))
                              for i in range(len(ds.LABEL_NAMES))},
        "test_group_counts": {f"{l}/{s}/{p}": int(sum(ds.fault_group(key) == (l, s, p) for key in test))
                              for (l, s, p) in all_groups},
        "empty_test_groups": sorted(f"{l}/{s}/{p}" for (l, s, p) in all_groups
                                    if (l, s, p) not in test_groups),
        "thin_test_groups": sorted(f"{l}/{s}/{p}" for (l, s, p) in all_groups
                                   if sum(ds.fault_group(key) == (l, s, p) for key in test) <= 1),
    }


def build_payload(data_root=None):
    """两个目标来源各跑一遍五步规则，组装 selection.json 内容。"""
    records = ds.enumerate_target_files(data_root)
    return {
        "protocol": "v6_round1",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "repr_config": pp.build_repr_config(),
        "rule": {
            "feature": "logspec512 逐窗口特征 → 文件内窗口均值 → DE+FE 拼接 512 维 → 候选集按维 z-score(ddof=0)",
            "channels": list(CHANNEL_ORDER),
            "short_record": f"DE 窗口数 < {SHORT_RECORD_RATIO} × 全体中位数，只约束进池",
            "k_selection": f"k∈[{K_SEARCH[0]},{K_SEARCH[-1]}] 中最小的满足 silhouette(k) ≥ "
                            f"{SILHOUETTE_TOLERANCE} × max silhouette 的 k",
            "representative": "每簇 medoid（到簇心欧氏距离最小），同分按 file_key 升序；单例簇排名额序列最后",
            "budget": "K = Σ_j max(1, round(p·n_j))，单例簇只在名额有余时按孤立度回填；本轮 p=0",
            "silhouette_tolerance": SILHOUETTE_TOLERANCE,
            "label_ratio_p": LABEL_RATIO_P,
            "random_state": KMEANS_RANDOM_STATE,
            "n_init": KMEANS_N_INIT,
            "metadata_used_in_selection": False,
        },
        "sources": {source: run_source(records, source) for source in ds.TARGET_DATASETS},
    }


def main():
    parser = argparse.ArgumentParser(description="v6 簇覆盖选标（只读信号，不加载权重）")
    parser.add_argument("--data-root", default=None, help="数据根，默认 ../rbop_diagnosis_v4/Data")
    parser.add_argument("--out", default=str(ds.V6_OUTPUT_ROOT / "selection" / "selection.json"))
    args = parser.parse_args()
    payload = build_payload(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for source, block in payload["sources"].items():
        cov = block["diagnostics"]["coverage_class_size"]
        print(f"{source}: 候选 {block['short_record_filter']['n_candidates']}/"
              f"{block['n_target_files']} 文件 → k={block['k_selected']} "
              f"(sil {block['silhouette_at_k_selected']:.3f}, max {block['silhouette_max']:.3f} "
              f"@k={block['k_at_max_silhouette']}), 单例簇 {block['singleton_clusters']}, "
              f"K={block['K']}, 类×尺寸覆盖 {cov['covered']}/{cov['total']}, "
              f"test {len(block['test'])} 文件 / {2 * len(block['test'])} 记录, "
              f"空层 {len(block['diagnostics']['empty_test_groups'])}")
        print("  池:", " ".join(Path(key).stem for key in block["pool"]))
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
