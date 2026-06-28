import csv
from typing import Dict, Iterable, List, Tuple

import numpy as np


"""
检索评估工具（教学注释版）。

功能清单：
1) 连续哈希码 -> 二值哈希码（{-1, +1}）。
2) 计算 query 与数据库样本的汉明距离。
3) 计算 mAP、P@K、R@K。
4) 统计平均 PR 曲线并导出 CSV。
5) 可选按 head/middle/tail 三组输出分组指标。

给 C++ 背景同学的提示：
- NumPy 的大部分操作是“向量化”计算，尽量避免 Python 层 for 循环。
- 字典 result 用来统一承载多种评估输出，类似一个轻量 struct/json。
"""


def sign_hash(codes: np.ndarray) -> np.ndarray:
    # 连续哈希码二值化到 {-1,+1}，用于汉明距离检索。
    # np.where(条件, 真值, 假值) 是逐元素选择。
    binary = np.where(codes >= 0, 1, -1)
    # int8 足够表达 -1/+1，内存占用更小。
    return binary.astype(np.int8)


def hamming_distance(query_code: np.ndarray, db_codes: np.ndarray) -> np.ndarray:
    # d_H = 0.5 * (b - <q, d>)，其中 b 为哈希位数。
    # 当 q, d ∈ {-1,+1}^b 时，这个公式与逐位比较完全等价。
    bits = query_code.shape[0]
    # np.dot(db_codes, query_code): 对数据库每个向量与 query 做点积。
    return 0.5 * (bits - np.dot(db_codes, query_code))


def _safe_div(num: float, den: float) -> float:
    # 防止除零，小工具函数。
    return float(num) / float(den) if den > 0 else 0.0


def _compute_ap(sorted_matches: np.ndarray, topk: int = 0) -> float:
    # sorted_matches 是按距离升序后的相关性序列：相关=1，不相关=0。
    if topk > 0:
        # 若设置 topk，只在前 k 个结果上计算 AP。
        sorted_matches = sorted_matches[:topk]

    positive = np.sum(sorted_matches)
    if positive == 0:
        # 该 query 没有任何正样本时，AP 定义为 0。
        return 0.0

    hit_count = 0.0
    precision_sum = 0.0
    # enumerate(..., start=1) 从 1 开始计 rank，更贴合检索公式。
    for idx, is_rel in enumerate(sorted_matches, start=1):
        if is_rel:
            hit_count += 1.0
            precision_sum += hit_count / idx

    # AP = 所有正样本命中位置处 precision 的平均值。
    return precision_sum / positive


def _precision_recall_at_k(sorted_matches: np.ndarray, total_pos: int, k: int) -> Tuple[float, float]:
    # K 不能超过候选数量。
    k = min(k, len(sorted_matches))
    if k == 0:
        return 0.0, 0.0

    # 前 k 个中的命中数。
    hits = float(np.sum(sorted_matches[:k]))
    precision = hits / float(k)
    recall = _safe_div(hits, total_pos)
    return precision, recall


def _group_classes_by_frequency(class_counts: List[int]) -> Dict[str, List[int]]:
    # 按类别频次从高到低切分 head / middle / tail。
    # 先构造 [(cls_id, count), ...]
    cls_with_count = list(enumerate(class_counts))
    # reverse=True 表示降序。
    cls_with_count.sort(key=lambda x: x[1], reverse=True)

    n_cls = len(cls_with_count)
    # 至少保证每组有机会分到类别。
    one_third = max(1, n_cls // 3)

    # Python 切片是“左闭右开”。
    head = [x[0] for x in cls_with_count[:one_third]]
    middle = [x[0] for x in cls_with_count[one_third : 2 * one_third]]
    tail = [x[0] for x in cls_with_count[2 * one_third :]]

    return {"head": head, "middle": middle, "tail": tail}


def evaluate_retrieval(
    query_codes: np.ndarray,
    query_labels: np.ndarray,
    db_codes: np.ndarray,
    db_labels: np.ndarray,
    topk: int = 0,
    ks: Iterable[int] = (10, 50, 100),
    class_counts: List[int] = None,
):
    # 1) 连续码统一二值化，保证检索距离定义一致。
    query_codes = sign_hash(query_codes)
    db_codes = sign_hash(db_codes)

    # ks 可能传 tuple，转 list 便于后续迭代和字典构造。
    ks = list(ks)
    ap_list = []
    # 用字典收集每个 K 的所有 query 结果，最后取平均。
    precision_at_k = {k: [] for k in ks}
    recall_at_k = {k: [] for k in ks}

    # rank_len = 数据库规模。
    rank_len = db_codes.shape[0]
    # 累积每个 rank 位上的 precision/recall，最后除以 query 数得到平均 PR 曲线。
    pr_precision = np.zeros(rank_len, dtype=np.float64)
    pr_recall = np.zeros(rank_len, dtype=np.float64)

    # 2) 逐 query 计算检索排序和指标。
    for i in range(query_codes.shape[0]):
        q_code = query_codes[i]
        q_label = query_labels[i]

        # 先按汉明距离升序排序，再基于排序结果计算 AP/P@K/R@K。
        dists = hamming_distance(q_code, db_codes)
        # stable 保持等距样本的原顺序稳定，利于可复现。
        order = np.argsort(dists, kind="stable")
        sorted_labels = db_labels[order]
        # 与 query 同标签视为相关样本。
        sorted_matches = (sorted_labels == q_label).astype(np.int32)

        total_pos = int(np.sum(sorted_matches))
        ap_list.append(_compute_ap(sorted_matches, topk=topk))

        # cumulative_hits[r-1] = 前 r 个结果里的命中数。
        cumulative_hits = np.cumsum(sorted_matches)
        ranks = np.arange(1, rank_len + 1)
        # 聚合全体 query 的 PR 曲线（先逐 query 统计，再在末尾取均值）。
        pr_precision += cumulative_hits / ranks
        if total_pos > 0:
            # 只有该 query 有正样本时 recall 才有意义。
            pr_recall += cumulative_hits / total_pos

        for k in ks:
            p, r = _precision_recall_at_k(sorted_matches, total_pos, k)
            precision_at_k[k].append(p)
            recall_at_k[k].append(r)

    # 3) 汇总全局指标（对所有 query 取平均）。
    result = {
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "P": {k: float(np.mean(v)) if v else 0.0 for k, v in precision_at_k.items()},
        "R": {k: float(np.mean(v)) if v else 0.0 for k, v in recall_at_k.items()},
        "pr_curve": {
            "rank": np.arange(1, rank_len + 1),
            "precision": pr_precision / max(1, query_codes.shape[0]),
            "recall": pr_recall / max(1, query_codes.shape[0]),
        },
    }

    if class_counts is not None:
        # 若提供类别频次，则额外输出 head/middle/tail 分组指标。
        groups = _group_classes_by_frequency(class_counts)
        group_result = {}
        for group_name, group_classes in groups.items():
            # np.isin: 判断每个 query_label 是否属于该组类别集合。
            mask = np.isin(query_labels, np.asarray(group_classes))
            if np.sum(mask) == 0:
                # 该组没有 query 样本时，指标置 0。
                group_result[group_name] = {"mAP": 0.0, "P": {k: 0.0 for k in ks}, "R": {k: 0.0 for k in ks}}
                continue

            # 递归调用自身计算子集指标；class_counts=None 避免无限递归分组。
            sub = evaluate_retrieval(
                query_codes[mask],
                query_labels[mask],
                db_codes,
                db_labels,
                topk=topk,
                ks=ks,
                class_counts=None,
            )
            group_result[group_name] = {
                "mAP": sub["mAP"],
                "P": sub["P"],
                "R": sub["R"],
            }

        result["groups"] = group_result

    return result


def save_pr_curve_csv(path: str, pr_curve: Dict[str, np.ndarray]):
    # newline="" 是 csv 官方推荐写法，避免 Windows 空行问题。
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "precision", "recall"])
        # zip 三列并行写出。
        for rank, p, r in zip(pr_curve["rank"], pr_curve["precision"], pr_curve["recall"]):
            writer.writerow([int(rank), float(p), float(r)])
