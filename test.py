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


"""检索评估入口：输出 mAP、P@K/R@K，并可导出 PR 曲线和 t-SNE。"""


def parse_args():
    parser = argparse.ArgumentParser(description="CIAH 复现实验评估脚本")
    parser.add_argument("--root", type=str, default="data/PatternNet")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--hash_bits", type=int, default=32)
    parser.add_argument("--weights", type=str, default="model_none_PatternNet.pth")
    parser.add_argument("--weights_template", type=str, default="model_bits{bits}_if{imb_factor}.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--query_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne_max", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split_path", type=str, default="split_patternnet.json")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--paper_like", action="store_true")
    parser.add_argument("--out_tag", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    # 固定随机种子，保证结果可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def extract_codes(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_codes = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            codes, _ = model(imgs)
            all_codes.append(codes.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_codes, axis=0), np.concatenate(all_labels, axis=0)


def maybe_save_tsne(path: str, codes: np.ndarray, labels: np.ndarray, max_points: int, seed: int):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("跳过 t-SNE：未安装 scikit-learn。")
        return

    if len(codes) == 0:
        return

    if len(codes) > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(codes), size=max_points, replace=False)
        codes = codes[idx]
        labels = labels[idx]

    emb = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(codes)

    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y,label\n")
        for (x, y), lab in zip(emb, labels):
            f.write(f"{x},{y},{int(lab)}\n")


def print_result(prefix: str, result: Dict):
    print(f"{prefix}mAP: {result['mAP']:.6f}")
    for k in [10, 50, 100]:
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
):
    if not os.path.isfile(weights_path):
        print(f"跳过：未找到权重 -> {weights_path}")
        return

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    long_tail = imb_factor < 1.0
    train_items, query_items, db_items, class_to_idx = build_train_query_db_splits(
        root=root,
        long_tail=long_tail,
        imb_factor=imb_factor,
        query_ratio=query_ratio,
        split_path=split_path,
        seed=seed,
    )

    del train_items

    query_dataset = PatternNetDataset(root=root, transform=transform, items=query_items)
    db_dataset = PatternNetDataset(root=root, transform=transform, items=db_items)

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

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        num_classes = checkpoint.get("num_classes", len(class_to_idx))
        hash_bits_from_ckpt = checkpoint.get("hash_bits", hash_bits)
    else:
        state_dict = checkpoint
        num_classes = len(class_to_idx)
        hash_bits_from_ckpt = hash_bits

    model = HashModel(hash_bits=hash_bits_from_ckpt, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state_dict, strict=True)

    query_codes, query_labels = extract_codes(model, query_loader, device)
    db_codes, db_labels = extract_codes(model, db_loader, device)

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
    print("=" * 72)
    print(
        f"IF={imb_factor}  bits={hash_bits_from_ckpt}  topk={topk_text}  "
        f"weights={weights_path}"
    )
    print_result("", result)

    if "groups" in result:
        for group_name in ["head", "middle", "tail"]:
            if group_name in result["groups"]:
                group_result = result["groups"][group_name]
                print_result(f"{group_name} ", group_result)

    tag = out_tag.strip() if out_tag else ""
    suffix = f"_{tag}" if tag else ""
    pr_path = f"pr_curve_bits{hash_bits_from_ckpt}_if{imb_factor}{suffix}.csv"
    save_pr_curve_csv(pr_path, result["pr_curve"])
    print(f"PR 曲线已保存到: {pr_path}")

    if tsne:
        tsne_path = f"tsne_bits{hash_bits_from_ckpt}_if{imb_factor}{suffix}.csv"
        maybe_save_tsne(tsne_path, query_codes, query_labels, tsne_max, seed)
        print(f"t-SNE 结果已保存到: {tsne_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"数据集根目录不存在: {args.root}")

    device = choose_device(args.device)

    if args.paper_like:
        for if_value in [0.1, 0.05, 0.01]:
            for bits in [16, 32, 64]:
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
                )
    else:
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
        )


if __name__ == "__main__":
    main()
