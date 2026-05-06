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
from utils.iam_loss import ClassBalancedCELoss, compute_class_weights


"""
单数据集训练入口（教学注释版）。

给 C++ 背景同学的快速定位：
1) 本文件相当于一个 main.cpp，负责“参数解析 -> 数据准备 -> 模型构建 -> 训练循环 -> 保存权重”。
2) Python 没有头文件/源文件分离，这里直接在一个文件里定义函数并在底部调用。
3) 训练目标是两部分损失的加权和：
    - 分类损失（IAMLoss）: 让类别预测正确。
    - 向心损失（CentripetalLoss）: 让同类样本的哈希码向同一个中心聚拢。
"""


def parse_args():
    # argparse 可理解为 C++ 里自己写的命令行参数解析器。
    # parse_args() 最终返回一个对象 args，可通过 args.xxx 访问每个参数。
    parser = argparse.ArgumentParser(description="CIAH 复现实验训练脚本")

    # 数据集根目录，默认是 PatternNet。
    parser.add_argument("--root", type=str, default="data/CLRS")
    # 不平衡因子，越小表示长尾越严重。
    parser.add_argument("--imb_factor", type=float, default=0.01)
    # 哈希码位数（例如 32 位二值码）。
    parser.add_argument("--hash_bits", type=int, default=32)
    # 训练轮次（完整遍历训练集的次数）。
    parser.add_argument("--epochs", type=int, default=10)
    # mini-batch 大小。
    parser.add_argument("--batch_size", type=int, default=32)
    # 这里保留参数是为了和实验配置兼容（当前文件中不直接使用）。
    parser.add_argument("--center_batch_size", type=int, default=128)
    # 总损失中分类项的权重 alpha：L = L_hash + alpha * L_cls。
    parser.add_argument("--alpha", type=float, default=0.2)
    # 向心损失系数 gamma（在损失内部使用）。
    parser.add_argument("--gamma", type=float, default=1.0)
    # Adam 学习率。
    parser.add_argument("--lr", type=float, default=1e-5)
    # 随机种子，控制可复现。
    parser.add_argument("--seed", type=int, default=42)
    # 权重输出路径。
    parser.add_argument("--weights_out", type=str, default="model_none_CLRS10轮.pth")
    # 设备名，支持 cuda / cuda:0 / auto（但本项目训练中最终强制 CUDA）。
    parser.add_argument("--device", type=str, default="cuda")
    # query 集比例（其余进入 db，当前实现 train 使用 db）。
    parser.add_argument("--query_ratio", type=float, default=0.2)
    # 持久化划分文件，确保每次运行样本划分一致。
    parser.add_argument("--split_path", type=str, default="split_CLRS.json")
    # 分类损失权重策略：none=普通CE，sqrt_inv=样本数开方倒数，class_balanced=有效样本数重加权。
    parser.add_argument("--cls_weighting", type=str, default="none", choices=["none", "sqrt_inv", "class_balanced"])
    # class_balanced 权重里的 beta。
    parser.add_argument("--cb_beta", type=float, default=0.999)
    # class_balanced 分子模式（1-beta 或 1）。
    parser.add_argument("--cb_mode", type=str, default="1", choices=["1-beta", "1"])
    # DataLoader 的并行 worker 数量。Windows 下常先用 0 保守启动。
    parser.add_argument("--num_workers", type=int, default=0)
    # 布尔开关参数：命令行写了 --amp 就为 True，没写就是 False。
    parser.add_argument("--amp", action="store_true",default="true", help="启用自动混合精度训练（仅 CUDA 生效）")
    # 是否跳过训练前的全量中心估计，直接使用随机中心初始化。
    parser.add_argument("--skip_center_init", action="store_true", help="跳过类别中心的模型均值初始化")
    # 是否关闭 ImageNet 预训练。
    parser.add_argument("--no_pretrained", action="store_true")

    # 返回 Namespace 对象，后续通过 args.xxx 读取。
    return parser.parse_args()


def set_seed(seed: int):
    # 固定随机种子，保证实验可复现。
    # 对应 C++ 里你会做的：为不同随机数引擎设置同一个 seed。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 多 GPU 场景下给所有 CUDA 设备设置种子。
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    # 强制使用 CUDA：仅接受 auto/cuda/cuda:N。
    normalized = name.lower()
    if normalized == "auto":
        # auto 在训练路径中被解释为 cuda。
        normalized = "cuda"

    if not normalized.startswith("cuda"):
        # Python 常用 raise 抛异常，等价于 C++ throw。
        raise ValueError(f"当前训练已强制使用 CUDA，不支持设备参数: {name}")

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA 设备，无法开始训练。")

    return torch.device(normalized)


def build_class_centers_from_model(model, dataloader, num_classes: int, hash_bits: int, device: torch.device):
    """按论文思路：用当前未训练模型输出的哈希码均值初始化类别中心。"""
    model.eval()
    sums = torch.zeros(num_classes, hash_bits, dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            hash_code, _ = model(img)

            for cls_idx in range(num_classes):
                mask = label == cls_idx
                if mask.any():
                    sums[cls_idx] += hash_code[mask].sum(dim=0)
                    counts[cls_idx] += mask.sum()

    valid = counts > 0
    centers = torch.zeros_like(sums)
    centers[valid] = sums[valid] / counts[valid].unsqueeze(1)

    # 若某类无样本（极端情况下），用小随机值兜底。
    if (~valid).any():
        missing = int((~valid).sum().item())
        centers[~valid] = torch.randn(missing, hash_bits, device=device) * 0.02

    return centers


def main():
    # 1) 读命令行参数。
    args = parse_args()
    # 2) 固定随机性。
    set_seed(args.seed)

    # 3) 基础输入检查。
    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"数据集根目录不存在: {args.root}")

    # 4) 选择设备并配置混合精度。
    device = choose_device(args.device)
    # bool(...) 是显式转布尔，防止 args.amp 之外的类型混入。
    use_amp = bool(args.amp and device.type == "cuda")
    # GradScaler 用于 AMP 下的梯度缩放，降低 FP16 下数值下溢风险。
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
            # 统一输入尺寸，便于进入固定结构的 CNN。
            transforms.Resize((224, 224)),
            # PIL 图像 -> Tensor，像素从 [0,255] 转为 [0,1]。
            transforms.ToTensor(),
            # 使用 ImageNet 常见均值方差归一化，和预训练主干匹配。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 与论文一致：当 imb_factor<1 时构造长尾分布；split_path 保证 train/test 划分可复现。
    long_tail = args.imb_factor < 1.0
    # 返回四元组：train_items / query_items / db_items / class_to_idx。
    # 这里用下划线 _ 接收不关心的值（Python 常见写法）。
    train_items, _, _, class_to_idx = build_train_query_db_splits(
        root=args.root,
        long_tail=long_tail,
        imb_factor=args.imb_factor,
        query_ratio=args.query_ratio,
        split_path=args.split_path,
        seed=args.seed,
    )

    dataset = PatternNetDataset(root=args.root, transform=transform, items=train_items)
    # DataLoader 相当于“带多线程预取和 batch 组装的数据迭代器”。
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # CUDA 训练时 pin_memory=True 通常能提升主机到显存传输效率。
        pin_memory=torch.cuda.is_available(),
    )

    # 类别数来自类名到索引映射。
    num_classes = len(class_to_idx)
    # 构建哈希模型并搬到目标设备。
    model = HashModel(
        hash_bits=args.hash_bits,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    # Counter 统计每个标签出现次数（类似 C++ map<label, count>）。
    counter = Counter(dataset.labels)
    # class_counts 用于类平衡权重，避免头部类在 CE 中主导梯度。
    # counter.get(i, 1) 的意思是：如果某类不存在，用 1 兜底，避免除零。
    class_counts = [counter.get(i, 1) for i in range(num_classes)]

    # 先根据样本频次计算“类别权重向量”。
    class_weights = compute_class_weights(
        class_counts=class_counts,
        mode=args.cls_weighting,
        cb_beta=args.cb_beta,
        cb_mode=args.cb_mode,
    )
    # 再构建带权重的分类损失。
    cls_criterion = ClassBalancedCELoss(class_weights=class_weights).to(device)

    # 默认在论文风格和训练速度之间做折中：网格实验可跳过这一步，直接随机初始化中心。
    if args.skip_center_init:
        initial_centers = None
    else:
        # 先用当前模型（未训练）估计每类中心，贴近论文中的中心均值初始化做法。
        initial_centers = build_class_centers_from_model(
            model=model,
            dataloader=train_loader,
            num_classes=num_classes,
            hash_bits=args.hash_bits,
            device=device,
        )

    # 向心损失：学习类别中心并约束类内哈希码聚合。
    hash_criterion = CentripetalLoss(
        num_classes=num_classes,
        hash_bits=args.hash_bits,
        gamma=args.gamma,
        initial_centers=initial_centers,
    ).to(device)

    # 注意：优化器同时更新主干网络参数与向心中心参数。
    # list(model.parameters()) + list(hash_criterion.parameters())
    # 相当于把两组可训练参数拼起来交给同一个 Adam。
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(hash_criterion.parameters()),
        lr=args.lr,
    )

    print("类别分布:")
    print(counter)

    for epoch in range(args.epochs):
        # train() 会启用训练行为（例如 BatchNorm/Dropout 的训练分支）。
        model.train()
        epoch_start = time.perf_counter()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_hash_loss = 0.0
        total_samples = 0

        # tqdm 是进度条库，便于查看训练进度和当前损失。
        pbar = tqdm(train_loader, desc=f"轮次 {epoch + 1}/{args.epochs}", ncols=100)
        # Python 可直接“解包” batch：img, label = 一个二元组。
        for img, label in pbar:
            # non_blocking=True 在 pin_memory 条件下可异步拷贝，提升吞吐。
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            batch_size = img.size(0)
            total_samples += batch_size

            # autocast 在 AMP 打开时，自动为部分算子用半精度。
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 前向：得到哈希码和分类 logits。
                hash_code, pred = model(img)

                # 分类监督项（论文中的 L_C1）。
                cls_loss = cls_criterion(pred, label)
                # 向心聚合项（论文中的 L_C2）。
                hash_loss = hash_criterion(hash_code, label)
                # 总目标：L = L_C2 + alpha * L_C1（与当前实现保持一致）。
                loss = hash_loss + args.alpha * cls_loss

            # 经典三步：清梯度 -> 反向传播 -> 参数更新。
            # 这里通过 scaler 做“缩放后的反向与更新”。
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_hash_loss += hash_loss.item()

            # 把当前 batch 的损失显示到进度条右侧。
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

    # 训练结束后保存 checkpoint。
    # 这里不仅存模型参数，还存元信息，便于测试阶段自动对齐配置。
    checkpoint = {
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "hash_bits": args.hash_bits,
        "class_to_idx": class_to_idx,
        "args": vars(args),
    }
    # torch.save 默认使用 Python pickle 序列化。
    torch.save(checkpoint, args.weights_out)
    print(f"权重已保存到: {args.weights_out}")


if __name__ == "__main__":
    # Python 程序入口保护：
    # 只有“直接运行本文件”才执行 main()；被 import 时不会自动执行。
    main()
