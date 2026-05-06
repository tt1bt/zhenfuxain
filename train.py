import argparse
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.patternnet_dataset import PatternNetDataset, build_train_query_db_splits
from models.hash_model import HashModel
from utils.centripetal_loss import CentripetalLoss
from utils.iam_loss import IAMLoss, compute_class_weights


"""单数据集训练入口：向心损失 + 可选类别平衡分类损失。"""


def parse_args():
    parser = argparse.ArgumentParser(description="CIAH 复现实验训练脚本")
    parser.add_argument("--root", type=str, default="data/PatternNet")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--hash_bits", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--center_batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights_out", type=str, default="model_cb_PatternNet.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query_ratio", type=float, default=0.2)
    parser.add_argument("--split_path", type=str, default="split_patternnet.json")
    parser.add_argument("--cls_weighting", type=str, default="class_balanced", choices=["none", "sqrt_inv", "class_balanced"])
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--cb_mode", type=str, default="1", choices=["1-beta", "1"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="启用自动混合精度训练（仅 CUDA 生效）")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    # 固定随机种子，保证实验可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    # 强制使用 CUDA：仅接受 auto/cuda/cuda:N。
    normalized = name.lower()
    if normalized == "auto":
        normalized = "cuda"

    if not normalized.startswith("cuda"):
        raise ValueError(f"当前训练已强制使用 CUDA，不支持设备参数: {name}")

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA 设备，无法开始训练。")

    return torch.device(name)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"数据集根目录不存在: {args.root}")

    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"训练设备: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"GPU: {gpu_name}")
        print(f"AMP: {'开启' if use_amp else '关闭'}")
    elif args.amp:
        print("AMP: 已请求，但当前不是 CUDA 设备，自动关闭。")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    long_tail = args.imb_factor < 1.0
    train_items, _, _, class_to_idx = build_train_query_db_splits(
        root=args.root,
        long_tail=long_tail,
        imb_factor=args.imb_factor,
        query_ratio=args.query_ratio,
        split_path=args.split_path,
        seed=args.seed,
    )

    dataset = PatternNetDataset(root=args.root, transform=transform, items=train_items)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(class_to_idx)
    model = HashModel(
        hash_bits=args.hash_bits,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    counter = Counter(dataset.labels)
    class_counts = [counter.get(i, 1) for i in range(num_classes)]

    class_weights = compute_class_weights(
        class_counts=class_counts,
        mode=args.cls_weighting,
        cb_beta=args.cb_beta,
        cb_mode=args.cb_mode,
    )
    cls_criterion = IAMLoss(class_weights=class_weights).to(device)

    hash_criterion = CentripetalLoss(
        num_classes=num_classes,
        hash_bits=args.hash_bits,
        gamma=args.gamma,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(hash_criterion.parameters()),
        lr=args.lr,
    )

    print("类别分布:")
    print(counter)

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.perf_counter()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_hash_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"轮次 {epoch + 1}/{args.epochs}", ncols=100)
        for img, label in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            batch_size = img.size(0)
            total_samples += batch_size

            with torch.cuda.amp.autocast(enabled=use_amp):
                hash_code, pred = model(img)

                cls_loss = cls_criterion(pred, label)
                hash_loss = hash_criterion(hash_code, label)
                loss = hash_loss + args.alpha * cls_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_hash_loss += hash_loss.item()

            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                hash=f"{hash_loss.item():.4f}",
            )

        avg_loss = total_loss / max(1, len(train_loader))
        avg_cls_loss = total_cls_loss / max(1, len(train_loader))
        avg_hash_loss = total_hash_loss / max(1, len(train_loader))
        epoch_time = max(1e-9, time.perf_counter() - epoch_start)
        samples_per_sec = total_samples / epoch_time

        print(
            f"轮次 {epoch + 1}: 总损失={avg_loss:.6f} "
            f"分类损失={avg_cls_loss:.6f} 向心损失={avg_hash_loss:.6f} "
            f"耗时={epoch_time:.2f}s 吞吐={samples_per_sec:.2f}样本/s"
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "hash_bits": args.hash_bits,
        "class_to_idx": class_to_idx,
        "args": vars(args),
    }
    torch.save(checkpoint, args.weights_out)
    print(f"权重已保存到: {args.weights_out}")


if __name__ == "__main__":
    main()
