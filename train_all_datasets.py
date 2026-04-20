import argparse
import os
import subprocess
import sys
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multiple datasets in one command and skip finished ones"
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["PatternNet", "NWPU-RESISC45", "RSSCN7", "CLRS"],
        help="Dataset folder names under data_root",
    )

    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--hash_bits", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--center_batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--query_ratio", type=float, default=0.2)
    parser.add_argument("--cls_weighting", type=str, default="none", choices=["none", "sqrt_inv", "class_balanced"])
    parser.add_argument("--cb_beta", type=float, default=0.9999)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument(
        "--weights_template",
        type=str,
        default="model_{cls_weighting}_{dataset}.pth",
        help="Template fields: dataset, cls_weighting, imb_factor, hash_bits",
    )
    parser.add_argument("--split_dir", type=str, default=".")
    parser.add_argument("--force", action="store_true", help="Train even if weight exists")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")

    return parser.parse_args()


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def build_weights_path(args, dataset_name: str) -> str:
    filename = args.weights_template.format(
        dataset=dataset_name,
        cls_weighting=args.cls_weighting,
        imb_factor=args.imb_factor,
        hash_bits=args.hash_bits,
    )
    return os.path.join(args.weights_dir, filename)


def build_split_path(args, dataset_name: str) -> str:
    return os.path.join(args.split_dir, f"split_{dataset_name}.json")


def run_one_dataset(args, dataset_name: str):
    dataset_root = os.path.join(args.data_root, dataset_name)
    if not os.path.isdir(dataset_root):
        print(f"[SKIP] Dataset folder missing: {dataset_root}")
        return "skipped", 0

    weights_out = build_weights_path(args, dataset_name)
    split_path = build_split_path(args, dataset_name)

    if os.path.exists(weights_out) and not args.force:
        print(f"[SKIP] Trained weights already exist: {weights_out}")
        return "skipped", 0

    cmd: List[str] = [
        sys.executable,
        "train.py",
        "--root",
        dataset_root,
        "--imb_factor",
        str(args.imb_factor),
        "--hash_bits",
        str(args.hash_bits),
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
        "--num_workers",
        str(args.num_workers),
    ]

    if args.no_pretrained:
        cmd.append("--no_pretrained")

    print("=" * 90)
    print(f"[RUN ] Dataset: {dataset_name}")
    print(f"[RUN ] Weights: {weights_out}")
    print("[CMD ] " + " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode == 0:
        return "trained", 0
    return "failed", result.returncode


def main():
    args = parse_args()

    ensure_dir(args.weights_dir)
    ensure_dir(args.split_dir)

    total = len(args.datasets)
    skipped = 0
    failed = 0
    trained = 0

    for dataset_name in args.datasets:
        status, code = run_one_dataset(args, dataset_name)

        if status == "trained":
            trained += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"[FAIL] Dataset: {dataset_name}, exit code: {code}")
            if not args.continue_on_error:
                print("[STOP] Stop on first failure. Use --continue_on_error to continue.")
                sys.exit(code)

    print("=" * 90)
    print("Batch training summary")
    print(f"Total:   {total}")
    print(f"Trained: {trained}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
