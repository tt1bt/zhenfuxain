import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.patternnet_dataset import PatternNetDataset, build_train_query_db_splits
from models.hash_model import HashModel
from utils.retrieval import evaluate_retrieval, save_pr_curve_csv


"""
检索评估入口（教学注释版）。

给 C++ 背景同学的直觉：
1) 训练脚本产出哈希模型权重，本文件负责“加载权重并评测检索效果”。
2) 评测流程：
   - 准备 query 集和 database 集；
   - 提取两边的哈希码；
   - 计算 mAP、P@K、R@K 和 PR 曲线；
   - 可选导出 t-SNE 可视化坐标。
3) 这里默认可在 CPU 上评估（choose_device 允许 auto 回退）。
"""


def parse_args():
    # 命令行参数定义区，和 train.py 一样由 argparse 管理。
    parser = argparse.ArgumentParser(description="CIAH reproduction evaluation")
    # 数据根目录。
    parser.add_argument("--root", type=str, default="data/CLRS")
    # 长尾不平衡因子（必须和训练时保持一致，才能复现实验条件）。
    parser.add_argument("--imb_factor", type=float, default=0.01)
    # 哈希码位数。
    parser.add_argument("--hash_bits", type=int, default=32)
    # 单次评估时使用的权重文件。
    parser.add_argument("--weights", type=str, default="model_none_CLRS10轮.pth")
    # paper_like 模式下用此模板自动拼接权重路径。
    parser.add_argument("--weights_template", type=str, default="model_bits{bits}_if{imb_factor}.pth")
    # 评估批大小。
    parser.add_argument("--batch_size", type=int, default=64)
    # query 子集占比。
    parser.add_argument("--query_ratio", type=float, default=0.2)
    # 随机种子。
    parser.add_argument("--seed", type=int, default=42)
    # 是否导出 t-SNE 坐标。
    parser.add_argument("--tsne", action="store_true")
    # t-SNE 最大采样点数，太大时会先随机下采样。
    parser.add_argument("--tsne_max", type=int, default=2000)
    # 设备：auto/cuda/cpu 等。
    parser.add_argument("--device", type=str, default="auto")
    # 划分文件路径，保证 query/db 可复现。
    parser.add_argument("--split_path", type=str, default="split_CLRS.json")
    # topk=0 表示按全部数据库评估 AP；>0 则截断到前 K。
    parser.add_argument("--topk", type=int, default=0)
    # 是否使用论文风格批量组合评估（IF x bits 网格）。
    parser.add_argument("--paper_like", action="store_true")
    # 输出文件名后缀标签。
    parser.add_argument("--out_tag", type=str, default="")
    # DataLoader worker 数。
    parser.add_argument("--num_workers", type=int, default=0)
    # 允许忽略 checkpoint 与当前评估参数不一致（默认不允许）。
    parser.add_argument("--allow_ckpt_mismatch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    # 固定随机性，保证划分和采样可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    # 测试阶段允许自动回退到 CPU，便于仅评估不训练的场景。
    if name == "auto":
        # 三元表达式：条件为真选前者，否则后者。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def extract_codes(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    # eval() 切到推理模式：关闭 Dropout 的随机性，BatchNorm 用统计量。
    model.eval()
    all_codes = []
    all_labels = []
    # no_grad() 关闭梯度追踪，减少显存和计算开销。
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            codes, _ = model(imgs)
            # 这里保留连续值哈希码；在 evaluate_retrieval 内统一做 sign 二值化。
            all_codes.append(codes.cpu().numpy())
            all_labels.append(labels.numpy())

    # np.concatenate 把“多个 batch 数组”拼接成一个大数组。
    return np.concatenate(all_codes, axis=0), np.concatenate(all_labels, axis=0)


def maybe_save_tsne(path: str, codes: np.ndarray, labels: np.ndarray, max_points: int, seed: int):
    # 局部导入：只有用户开启 --tsne 才会触发对 sklearn 的依赖。
    try:
        from sklearn.manifold import TSNE  # type: ignore[reportMissingImports]
    except ImportError:
        print("Skip t-SNE: scikit-learn is not installed.")
        return

    if len(codes) == 0:
        # 没有数据就不生成文件。
        return

    if len(codes) > max_points:
        # 点数过多时随机抽样，避免 t-SNE 计算过慢。
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(codes), size=max_points, replace=False)
        codes = codes[idx]
        labels = labels[idx]

    # 2 维嵌入，便于后续画散点图。
    emb = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(codes)

    # 直接写 CSV，列为 x,y,label。
    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y,label\n")
        # zip 会并行迭代两个序列。
        for (x, y), lab in zip(emb, labels):
            f.write(f"{x},{y},{int(lab)}\n")


def print_result(prefix: str, result: Dict):
    # f-string 用于格式化打印；:.6f 表示保留 6 位小数。
    print(f"{prefix}mAP: {result['mAP']:.6f}")
    for k in [10, 50, 100]:
        # 字典存在性判断：避免访问不存在的键。
        if k in result["P"]:
            print(f"{prefix}P@{k}: {result['P'][k]:.4f}  R@{k}: {result['R'][k]:.4f}")


def evaluate_once(
    root: str,
    imb_factor: float,
    hash_bits: int,
    weights_path: str,
    batch_size: int,
    query_ratio: float,
    split_path: str,
    topk: int,
    seed: int,
    device: torch.device,
    out_tag: str,
    tsne: bool,
    tsne_max: int,
    num_workers: int,
    allow_ckpt_mismatch: bool,
    verbose: bool = True,
    save_pr_curve: bool = True,
):

    checkpoint = torch.load(weights_path, map_location="cpu")
    # 兼容两种权重格式：
    # 1) 新格式 dict，含 model_state + 元信息。
    # 2) 旧格式，直接就是 state_dict。
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        num_classes_from_ckpt = checkpoint.get("num_classes")
        hash_bits_from_ckpt = checkpoint.get("hash_bits", hash_bits)
        ckpt_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        num_classes_from_ckpt = None
        hash_bits_from_ckpt = hash_bits
        ckpt_args = {}

    # 防止“训练参数与评估参数不一致”导致的虚高/失真结论。
    ckpt_imb = ckpt_args.get("imb_factor") if isinstance(ckpt_args, dict) else None
    ckpt_qr = ckpt_args.get("query_ratio") if isinstance(ckpt_args, dict) else None
    mismatch_msgs = []

    if ckpt_imb is not None and abs(float(ckpt_imb) - float(imb_factor)) > 1e-12:
        mismatch_msgs.append(f"imb_factor: ckpt={ckpt_imb}, eval={imb_factor}")
    if ckpt_qr is not None and abs(float(ckpt_qr) - float(query_ratio)) > 1e-12:
        mismatch_msgs.append(f"query_ratio: ckpt={ckpt_qr}, eval={query_ratio}")
    if int(hash_bits_from_ckpt) != int(hash_bits):
        mismatch_msgs.append(f"hash_bits: ckpt={hash_bits_from_ckpt}, eval={hash_bits}")

    if mismatch_msgs:
        msg = "checkpoint 与当前评估参数不一致: " + "; ".join(mismatch_msgs)
        if allow_ckpt_mismatch:
            print("[WARN] " + msg)
        else:
            raise ValueError(msg + "。可加 --allow_ckpt_mismatch 强制评估。")

    # 与训练一致的预处理，否则特征分布会漂移导致评估不公平。
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 与训练复用同一 split 规则，保证 query/db 对比公平可复现。
    long_tail = imb_factor < 1.0
    train_items, query_items, db_items, class_to_idx = build_train_query_db_splits(
        root=root,
        long_tail=long_tail,
        imb_factor=imb_factor,
        query_ratio=query_ratio,
        split_path=split_path,
        seed=seed,
    )

    # train_items 在评估流程里不需要，显式删除可读性更强。
    del train_items

    # 构建 query / db 两个数据集对象。
    query_dataset = PatternNetDataset(root=root, transform=transform, items=query_items)
    db_dataset = PatternNetDataset(root=root, transform=transform, items=db_items)

    # 注意评估不打乱顺序（shuffle=False），便于可复现和调试。
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    db_loader = DataLoader(
        db_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = num_classes_from_ckpt if num_classes_from_ckpt is not None else len(class_to_idx)

    # 测试时固定 pretrained=False，避免覆盖我们加载的权重。
    model = HashModel(hash_bits=hash_bits_from_ckpt, num_classes=num_classes, pretrained=False).to(device)
    # strict=True：键名或 shape 对不上就立刻报错，防止 silent bug。
    model.load_state_dict(state_dict, strict=True)

    # 分别提取 query 与 db 的哈希码及标签。
    query_codes, query_labels = extract_codes(model, query_loader, device)
    db_codes, db_labels = extract_codes(model, db_loader, device)

    # 用数据库集频次定义 head/middle/tail，和论文分组评估思路一致。
    class_counts = [0] * len(class_to_idx)
    for lab in db_labels:
        class_counts[int(lab)] += 1

    result = evaluate_retrieval(
        query_codes=query_codes,
        query_labels=query_labels,
        db_codes=db_codes,
        db_labels=db_labels,
        topk=topk,
        class_counts=class_counts,
    )

    topk_text = "full" if topk == 0 else str(topk)
    if verbose:
        print("=" * 72)
        print(
            f"IF={imb_factor}  bits={hash_bits_from_ckpt}  topk={topk_text}  "
            f"weights={weights_path}"
        )
        print_result("", result)

    if verbose and "groups" in result:
        # 如果 evaluate_retrieval 返回了 head/middle/tail 分组指标，则逐组打印。
        for group_name in ["head", "middle", "tail"]:
            if group_name in result["groups"]:
                group_result = result["groups"][group_name]
                print_result(f"{group_name} ", group_result)

    # 输出文件后缀拼接逻辑：out_tag 为空则不加后缀。
    tag = out_tag.strip() if out_tag else ""
    suffix = f"_{tag}" if tag else ""
    if save_pr_curve:
        pr_path = f"pr_curve_bits{hash_bits_from_ckpt}_if{imb_factor}{suffix}.csv"
        save_pr_curve_csv(pr_path, result["pr_curve"])
        if verbose:
            print(f"Saved PR curve to {pr_path}")

    if tsne:
        tsne_path = f"tsne_bits{hash_bits_from_ckpt}_if{imb_factor}{suffix}.csv"
        maybe_save_tsne(tsne_path, query_codes, query_labels, tsne_max, seed)
        if verbose:
            print(f"Saved t-SNE to {tsne_path}")

    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    device = choose_device(args.device)

    if args.paper_like:
        # 论文风格批量评估：遍历 IF 与 bits 组合，读取命名模板权重。
        for if_value in [0.1, 0.05, 0.01]:
            for bits in [16, 32, 64]:
                # 模板替换示例：model_bits32_if0.01.pth
                weights_path = args.weights_template.format(bits=bits, imb_factor=if_value)
                evaluate_once(
                    root=args.root,
                    imb_factor=if_value,
                    hash_bits=bits,
                    weights_path=weights_path,
                    batch_size=args.batch_size,
                    query_ratio=args.query_ratio,
                    split_path=args.split_path,
                    topk=args.topk,
                    seed=args.seed,
                    device=device,
                    out_tag=args.out_tag,
                    tsne=args.tsne,
                    tsne_max=args.tsne_max,
                    num_workers=args.num_workers,
                    allow_ckpt_mismatch=args.allow_ckpt_mismatch,
                    verbose=True,
                    save_pr_curve=True,
                )
    else:
            # 单次评估模式：只跑一个指定权重。
        evaluate_once(
            root=args.root,
            imb_factor=args.imb_factor,
            hash_bits=args.hash_bits,
            weights_path=args.weights,
            batch_size=args.batch_size,
            query_ratio=args.query_ratio,
            split_path=args.split_path,
            topk=args.topk,
            seed=args.seed,
            device=device,
            out_tag=args.out_tag,
            tsne=args.tsne,
            tsne_max=args.tsne_max,
            num_workers=args.num_workers,
            allow_ckpt_mismatch=args.allow_ckpt_mismatch,
            verbose=True,
            save_pr_curve=True,
        )


if __name__ == "__main__":
    main()