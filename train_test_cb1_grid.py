import argparse
import csv
import os
import subprocess
import sys
import random
from typing import Dict, List

import numpy as np


"""cb-1 网格实验脚本：
1) 依次训练四个数据集在不同 IF 与 bits 下的模型；
2) 评估阶段只打印平均 mAP；
3) 结果统一落到 CSV，便于后续整理表格。
"""


def parse_args():
    parser = argparse.ArgumentParser(description="批量训练/评测 cb-1 网格实验")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["PatternNet", "NWPU-RESISC45", "RSSCN7", "CLRS"],
        help="data_root 下的数据集文件夹名称",
    )
    parser.add_argument("--ifs", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--bits", nargs="+", type=int, default=[16, 32, 64])

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--center_batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--query_ratio", type=float, default=0.2)
    parser.add_argument("--cls_weighting", type=str, default="class_balanced", choices=["none", "sqrt_inv", "class_balanced"])
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--cb_mode", type=str, default="1", choices=["1-beta", "1"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="启用自动混合精度训练（仅 CUDA 生效）")
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument(
        "--weights_template",
        type=str,
        default="model_cb1_{dataset}_if{imb_factor}_bits{hash_bits}.pth",
        help="模板字段: dataset, imb_factor, hash_bits",
    )
    parser.add_argument("--split_dir", type=str, default=".")
    parser.add_argument(
        "--split_template",
        type=str,
        default="split_{dataset}_if{imb_factor}.json",
        help="模板字段: dataset, imb_factor",
    )
    parser.add_argument("--results_csv", type=str, default="cb1_grid_results.csv")
    parser.add_argument("--force", action="store_true", help="即使权重已存在也强制重新训练")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_ckpt_mismatch", action="store_true", help="允许 checkpoint 与评估参数不完全一致")
    return parser.parse_args()


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_value(value) -> str:
    return str(value)


def build_weights_path(args, dataset: str, imb_factor: float, hash_bits: int) -> str:
    filename = args.weights_template.format(
        dataset=dataset,
        imb_factor=format_value(imb_factor),
        hash_bits=hash_bits,
    )
    return os.path.join(args.weights_dir, filename)


def build_split_path(args, dataset: str, imb_factor: float) -> str:
    filename = args.split_template.format(dataset=dataset, imb_factor=format_value(imb_factor))
    return os.path.join(args.split_dir, filename)


def run_train(args, dataset: str, imb_factor: float, hash_bits: int, weights_out: str, split_path: str):
    dataset_root = os.path.join(args.data_root, dataset)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    cmd: List[str] = [
        sys.executable,
        "train.py",
        "--root",
        dataset_root,
        "--imb_factor",
        str(imb_factor),
        "--hash_bits",
        str(hash_bits),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--center_batch_size",
        str(args.center_batch_size),
        "--alpha",
        str(args.alpha),
        "--gamma",
        str(args.gamma),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--weights_out",
        weights_out,
        "--device",
        str(args.device),
        "--query_ratio",
        str(args.query_ratio),
        "--split_path",
        split_path,
        "--cls_weighting",
        str(args.cls_weighting),
        "--cb_beta",
        str(args.cb_beta),
        "--cb_mode",
        str(args.cb_mode),
        "--num_workers",
        str(args.num_workers),
    ]

    if args.amp:
        cmd.append("--amp")
    if args.no_pretrained:
        cmd.append("--no_pretrained")
    cmd.append("--skip_center_init")

    print("[训练] " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def evaluate_map(args, dataset: str, imb_factor: float, hash_bits: int, weights_path: str, split_path: str) -> float:
    from test import choose_device, evaluate_once

    device = choose_device(args.device)
    dataset_root = os.path.join(args.data_root, dataset)
    result = evaluate_once(
        root=dataset_root,
        imb_factor=imb_factor,
        hash_bits=hash_bits,
        weights_path=weights_path,
        batch_size=args.batch_size,
        query_ratio=args.query_ratio,
        split_path=split_path,
        topk=0,
        seed=args.seed,
        device=device,
        out_tag="",
        tsne=False,
        tsne_max=2000,
        num_workers=args.num_workers,
        allow_ckpt_mismatch=args.allow_ckpt_mismatch,
        verbose=False,
        save_pr_curve=False,
    )
    return float(result["mAP"])


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.weights_dir)
    ensure_dir(args.split_dir)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root 不存在: {args.data_root}")

    rows: List[Dict[str, object]] = []
    failed = 0

    for dataset in args.datasets:
        for imb_factor in args.ifs:
            split_path = build_split_path(args, dataset, imb_factor)
            for hash_bits in args.bits:
                try:
                    weights_path = build_weights_path(args, dataset, imb_factor, hash_bits)

                    if os.path.exists(weights_path) and not args.force:
                        print(f"[跳过训练] {weights_path}")
                    else:
                        run_train(args, dataset, imb_factor, hash_bits, weights_path, split_path)

                    mAP = evaluate_map(args, dataset, imb_factor, hash_bits, weights_path, split_path)
                    print(f"[mAP] dataset={dataset} if={imb_factor} bits={hash_bits} mAP={mAP:.6f}")

                    rows.append(
                        {
                            "dataset": dataset,
                            "imb_factor": imb_factor,
                            "hash_bits": hash_bits,
                            "weights": weights_path,
                            "split_path": split_path,
                            "mAP": mAP,
                        }
                    )
                except Exception as exc:
                    failed += 1
                    print(f"[失败] dataset={dataset} if={imb_factor} bits={hash_bits}: {exc}")
                    if not args.continue_on_error:
                        raise

    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "imb_factor", "hash_bits", "weights", "split_path", "mAP"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 结果已保存到: {args.results_csv}")
    if failed:
        print(f"[汇总] 失败组合数: {failed}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[失败] 子进程退出码: {exc.returncode}")
        raise