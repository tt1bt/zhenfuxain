import argparse
import os
import random
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.patternnet_dataset import PatternNetDataset, build_train_query_db_splits
from models.hash_model import HashModel
from utils.centripetal_loss import CentripetalLoss
from utils.iam_loss import IAMLoss


def parse_args():
    parser = argparse.ArgumentParser(description="CIAH reproduction training")
    parser.add_argument("--root", type=str, default="data/PatternNet")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--hash_bits", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--center_batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights_out", type=str, default="model_plain_PatternNet.pth")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--query_ratio", type=float, default=0.2)
    parser.add_argument("--split_path", type=str, default="split_patternnet.json")
    parser.add_argument("--cls_weighting", type=str, default="none", choices=["none", "sqrt_inv", "class_balanced"])
    parser.add_argument("--cb_beta", type=float, default=0.9999)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    device = choose_device(args.device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
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

    if args.cls_weighting == "none":
        cls_criterion = torch.nn.CrossEntropyLoss()
    else:
        cls_criterion = IAMLoss(
            class_counts=class_counts,
            mode=args.cls_weighting,
            cb_beta=args.cb_beta,
        ).to(device)

    hash_criterion = CentripetalLoss(
        num_classes=num_classes,
        hash_bits=args.hash_bits,
        gamma=args.gamma,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(hash_criterion.parameters()),
        lr=args.lr,
    )

    print("Class Distribution:")
    print(counter)

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_hash_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)
        for img, label in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            hash_code, pred = model(img)

            cls_loss = cls_criterion(pred, label)
            hash_loss = hash_criterion(hash_code, label)
            loss = hash_loss + args.alpha * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        print(
            f"Epoch {epoch + 1}: Loss={avg_loss:.6f} "
            f"Cls={avg_cls_loss:.6f} Hash={avg_hash_loss:.6f}"
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "hash_bits": args.hash_bits,
        "class_to_idx": class_to_idx,
        "args": vars(args),
    }
    torch.save(checkpoint, args.weights_out)
    print(f"Saved weights to {args.weights_out}")


if __name__ == "__main__":
    main()
