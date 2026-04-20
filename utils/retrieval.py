import csv
from typing import Dict, Iterable, List, Tuple

import numpy as np


def sign_hash(codes: np.ndarray) -> np.ndarray:
    binary = np.where(codes >= 0, 1, -1)
    return binary.astype(np.int8)


def hamming_distance(query_code: np.ndarray, db_codes: np.ndarray) -> np.ndarray:
    bits = query_code.shape[0]
    return 0.5 * (bits - np.dot(db_codes, query_code))


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _compute_ap(sorted_matches: np.ndarray, topk: int = 0) -> float:
    if topk > 0:
        sorted_matches = sorted_matches[:topk]

    positive = np.sum(sorted_matches)
    if positive == 0:
        return 0.0

    hit_count = 0.0
    precision_sum = 0.0
    for idx, is_rel in enumerate(sorted_matches, start=1):
        if is_rel:
            hit_count += 1.0
            precision_sum += hit_count / idx

    return precision_sum / positive


def _precision_recall_at_k(sorted_matches: np.ndarray, total_pos: int, k: int) -> Tuple[float, float]:
    k = min(k, len(sorted_matches))
    if k == 0:
        return 0.0, 0.0

    hits = float(np.sum(sorted_matches[:k]))
    precision = hits / float(k)
    recall = _safe_div(hits, total_pos)
    return precision, recall


def _group_classes_by_frequency(class_counts: List[int]) -> Dict[str, List[int]]:
    cls_with_count = list(enumerate(class_counts))
    cls_with_count.sort(key=lambda x: x[1], reverse=True)

    n_cls = len(cls_with_count)
    one_third = max(1, n_cls // 3)

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
    query_codes = sign_hash(query_codes)
    db_codes = sign_hash(db_codes)

    ks = list(ks)
    ap_list = []
    precision_at_k = {k: [] for k in ks}
    recall_at_k = {k: [] for k in ks}

    rank_len = db_codes.shape[0]
    pr_precision = np.zeros(rank_len, dtype=np.float64)
    pr_recall = np.zeros(rank_len, dtype=np.float64)

    for i in range(query_codes.shape[0]):
        q_code = query_codes[i]
        q_label = query_labels[i]

        dists = hamming_distance(q_code, db_codes)
        order = np.argsort(dists, kind="stable")
        sorted_labels = db_labels[order]
        sorted_matches = (sorted_labels == q_label).astype(np.int32)

        total_pos = int(np.sum(sorted_matches))
        ap_list.append(_compute_ap(sorted_matches, topk=topk))

        cumulative_hits = np.cumsum(sorted_matches)
        ranks = np.arange(1, rank_len + 1)
        pr_precision += cumulative_hits / ranks
        if total_pos > 0:
            pr_recall += cumulative_hits / total_pos

        for k in ks:
            p, r = _precision_recall_at_k(sorted_matches, total_pos, k)
            precision_at_k[k].append(p)
            recall_at_k[k].append(r)

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
        groups = _group_classes_by_frequency(class_counts)
        group_result = {}
        for group_name, group_classes in groups.items():
            mask = np.isin(query_labels, np.asarray(group_classes))
            if np.sum(mask) == 0:
                group_result[group_name] = {"mAP": 0.0, "P": {k: 0.0 for k in ks}, "R": {k: 0.0 for k in ks}}
                continue

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
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "precision", "recall"])
        for rank, p, r in zip(pr_curve["rank"], pr_curve["precision"], pr_curve["recall"]):
            writer.writerow([int(rank), float(p), float(r)])
