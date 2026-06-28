import argparse
import os
import subprocess
import sys
from typing import List


"""批量训练入口：按数据集循环调用 train.py，并支持跳过已完成权重。"""


def parse_args():
    # 这个脚本可以看成“训练调度器”：
    # 不直接训练模型，而是多次调用 train.py。
    parser = argparse.ArgumentParser(
        description="一条命令训练多个数据集，并自动跳过已完成项"
    )
    # data_root 下应包含多个数据集子目录。
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["PatternNet", "NWPU-RESISC45", "RSSCN7", "CLRS"],
        help="data_root 下的数据集文件夹名称",
    )

    # 以下参数会透传给 train.py。
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--hash_bits", type=int, default=32)
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

    # 权重输出目录和命名模板。
    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument(
        "--weights_template",
        type=str,
        default="model_cb_{dataset}.pth",
        help="模板字段: dataset, cls_weighting, imb_factor, hash_bits",
    )
    # 各数据集 split json 的输出目录。
    parser.add_argument("--split_dir", type=str, default=".")
    # 强制重训：即使目标权重已存在也重跑。
    parser.add_argument("--force", action="store_true", help="即使权重已存在也强制训练")
    parser.add_argument("--no_pretrained", action="store_true")
    # 某个数据集训练失败后，是否继续后续数据集。
    parser.add_argument("--continue_on_error", action="store_true")

    return parser.parse_args()


def ensure_dir(path: str):
    # 目录不存在时自动创建，避免后续写文件失败。
    if path:
        os.makedirs(path, exist_ok=True)


def build_weights_path(args, dataset_name: str) -> str:
    # 根据模板渲染权重文件名。
    # 例如模板 model_{dataset}_{hash_bits}.pth -> model_CLRS_32.pth
    filename = args.weights_template.format(
        dataset=dataset_name,
        cls_weighting=args.cls_weighting,
        imb_factor=args.imb_factor,
        hash_bits=args.hash_bits,
    )
    return os.path.join(args.weights_dir, filename)


def build_split_path(args, dataset_name: str) -> str:
    # 每个数据集独立 split 文件，防止互相覆盖。
    return os.path.join(args.split_dir, f"split_{dataset_name}.json")


def run_one_dataset(args, dataset_name: str):
    # 目标数据目录：data_root/dataset_name
    dataset_root = os.path.join(args.data_root, dataset_name)
    if not os.path.isdir(dataset_root):
        print(f"[跳过] 数据集目录不存在: {dataset_root}")
        # 返回 (状态, 退出码)
        return "skipped", 0

    weights_out = build_weights_path(args, dataset_name)
    split_path = build_split_path(args, dataset_name)

    # 已有权重且未指定 --force，就直接跳过节省时间。
    if os.path.exists(weights_out) and not args.force:
        print(f"[跳过] 已存在训练权重: {weights_out}")
        return "skipped", 0

    # 组装子进程命令。
    # 这里不使用 shell=True，而是“列表参数形式”更安全可靠。
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
        "--cb_mode",
        str(args.cb_mode),
        "--num_workers",
        str(args.num_workers),
    ]

    # 这些是布尔开关参数，只有 True 时才追加到命令列表。
    if args.amp:
        cmd.append("--amp")

    if args.no_pretrained:
        cmd.append("--no_pretrained")

    print("=" * 90)
    print(f"[运行] 数据集: {dataset_name}")
    print(f"[运行] 权重输出: {weights_out}")
    print("[命令] " + " ".join(cmd))

    # 同步执行子进程，阻塞等待训练完成。
    result = subprocess.run(cmd)
    if result.returncode == 0:
        return "trained", 0
    return "failed", result.returncode


def main():
    args = parse_args()

    # 预先创建输出目录，避免中途因路径问题失败。
    ensure_dir(args.weights_dir)
    ensure_dir(args.split_dir)

    # 汇总计数器。
    total = len(args.datasets)
    skipped = 0
    failed = 0
    trained = 0

    # 逐数据集调度 train.py。
    for dataset_name in args.datasets:
        status, code = run_one_dataset(args, dataset_name)

        if status == "trained":
            trained += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"[失败] 数据集: {dataset_name}, 退出码: {code}")
            if not args.continue_on_error:
                print("[停止] 遇到首个失败即停止。可使用 --continue_on_error 继续。")
                # 直接把子进程错误码透传给当前进程，便于 CI/脚本判断失败。
                sys.exit(code)

    print("=" * 90)
    print("批量训练汇总")
    print(f"总计:   {total}")
    print(f"完成:   {trained}")
    print(f"跳过:   {skipped}")
    print(f"失败:   {failed}")

    if failed > 0:
        # 批处理里只要有一个失败，最终退出码置为非 0。
        sys.exit(1)


if __name__ == "__main__":
    main()
