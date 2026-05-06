import argparse
import csv
from glob import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


"""
检索曲线绘图脚本。

输入：test.py 导出的 PR 曲线 CSV，默认列为 rank, precision, recall。
输出：
1) Recall-Number 曲线图
2) Precision-Number 曲线图
3) Precision-Recall 曲线图

用法示例：
python plot_retrieval_curves.py --curve_csv pr_curve_bits32_if0.01.csv --out_dir plots
也可以直接运行，脚本会自动查找当前目录下的 pr_curve*.csv 文件。

当前脚本会优先把“同一数据集 + 不同 beta”的曲线画在一张图里，
输出目录结构为：plots/<数据集名>/*.png
"""

DATASET_NAMES = ("PatternNet", "NWPU", "RSSCN7", "CLRS")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot retrieval curves from test CSV files")
    parser.add_argument(
        "--curve_csv",
        nargs="*",
        default=[],
        help="test.py 导出的 PR 曲线 CSV，可以传一个或多个文件",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="图像输出目录",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="绘图采样步长，例如 10 或 50；1 表示不采样",
    )
    return parser.parse_args()


def resolve_curve_csvs(curve_csvs):
    if curve_csvs:
        return curve_csvs

    default_files = sorted(glob("pr_curve*.csv"))
    if default_files:
        print("未提供 --curve_csv，已自动使用当前目录下的 pr_curve*.csv 文件。")
        return default_files

    raise SystemExit("未找到可绘制的曲线文件，请传入 --curve_csv pr_curve*.csv")


def infer_dataset_name(path: str) -> str:
    stem = Path(path).stem.lower()
    for dataset in DATASET_NAMES:
        if dataset.lower() in stem:
            return dataset
    return "Unknown"


def infer_beta_label(path: str):
    stem = Path(path).stem.lower()

    match = re.search(r"(?:^|_)(?:b|beta)(\d+p\d+)(?:_|$)", stem)
    if not match:
        return None, None

    raw = match.group(1)
    label = raw.replace("p", ".")
    try:
        value = float(label)
    except ValueError:
        value = None
    return label, value


def load_curve_csv(path: str):
    ranks = []
    precisions = []
    recalls = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["rank"]))
            precisions.append(float(row["precision"]))
            recalls.append(float(row["recall"]))
    return np.asarray(ranks), np.asarray(precisions), np.asarray(recalls)


def sample_curve(x: np.ndarray, y: np.ndarray, step: int):
    if step <= 1 or len(x) == 0:
        return x, y

    idx = np.arange(0, len(x), step)
    if idx.size == 0 or idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return x[idx], y[idx]


def plot_line(output_path: Path, xs, ys, xlabel: str, ylabel: str, title: str, labels):
    # 全局样式调整，放大文字并提高线条可辨识性
    plt.figure(figsize=(11, 7), dpi=150)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        mk = markers[i % len(markers)]
        # 数据点较多时稀疏绘制标记，增强线上可辨识度
        markevery = max(1, int(len(x) / 20)) if hasattr(x, "__len__") else None
        plt.plot(
            x,
            y,
            linewidth=3.0,
            marker=mk,
            markersize=8,
            markevery=markevery,
            label=label,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    if len(labels) > 1:
        plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def build_dataset_groups(curve_csvs):
    grouped = {}
    for csv_path in curve_csvs:
        dataset_name = infer_dataset_name(csv_path)
        beta_label, beta_value = infer_beta_label(csv_path)
        grouped.setdefault(dataset_name, []).append(
            {
                "path": csv_path,
                "stem": Path(csv_path).stem,
                "beta_label": beta_label,
                "beta_value": beta_value,
            }
        )

    selected_groups = {}
    for dataset_name, items in grouped.items():
        beta_items = [item for item in items if item["beta_label"] is not None]
        if beta_items:
            selected_groups[dataset_name] = beta_items
        else:
            selected_groups[dataset_name] = items

    if any(name != "Unknown" for name in selected_groups):
        selected_groups = {name: items for name, items in selected_groups.items() if name != "Unknown"}

    return selected_groups


def plot_dataset_group(dataset_name: str, items, out_dir: Path, step: int):
    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    ordered_items = sorted(
        items,
        key=lambda item: (
            item["beta_value"] is None,
            item["beta_value"] if item["beta_value"] is not None else float("inf"),
            item["stem"],
        ),
    )

    curve_labels = []
    sampled_ranks = []
    sampled_recalls = []
    sampled_precisions = []
    full_recalls = []
    full_precisions = []

    for item in ordered_items:
        ranks, precisions, recalls = load_curve_csv(item["path"])
        srank, srecall = sample_curve(ranks, recalls, step)
        _, sprecision = sample_curve(ranks, precisions, step)
        sampled_ranks.append(srank)
        sampled_recalls.append(srecall)
        sampled_precisions.append(sprecision)
        full_recalls.append(recalls)
        full_precisions.append(precisions)

        if item["beta_label"] is None:
            curve_labels.append(item["stem"])
        else:
            curve_labels.append(f"beta={item['beta_label']}")

    plot_line(
        dataset_dir / f"{dataset_name}_recall_number.png",
        sampled_ranks,
        sampled_recalls,
        xlabel="Retrieval number K",
        ylabel="Recall@K",
        title=f"Recall-Number Curve ({dataset_name})",
        labels=curve_labels,
    )
    plot_line(
        dataset_dir / f"{dataset_name}_precision_number.png",
        sampled_ranks,
        sampled_precisions,
        xlabel="Retrieval number K",
        ylabel="Precision@K",
        title=f"Precision-Number Curve ({dataset_name})",
        labels=curve_labels,
    )
    plot_line(
        dataset_dir / f"{dataset_name}_precision_recall.png",
        full_recalls,
        full_precisions,
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall Curve ({dataset_name})",
        labels=curve_labels,
    )

    return dataset_dir


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_csvs = resolve_curve_csvs(args.curve_csv)
    grouped = build_dataset_groups(curve_csvs)

    if not grouped:
        raise ValueError("没有可绘制的曲线数据。")

    for dataset_name, items in grouped.items():
        dataset_dir = plot_dataset_group(dataset_name, items, out_dir, args.step)
        print(f"已输出到: {dataset_dir.resolve()}")
        print(f"生成文件: {dataset_name}_recall_number.png, {dataset_name}_precision_number.png, {dataset_name}_precision_recall.png")


if __name__ == "__main__":
    main()