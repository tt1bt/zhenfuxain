import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.longtail_dataset import get_img_num_per_cls


"""
PatternNet 数据集与划分逻辑（教学注释版）。

本文件包含两类功能：
1) 工具函数：扫描目录、构造长尾采样、读写 split 文件。
2) Dataset 类：实现 PyTorch 的 Dataset 接口（__len__ / __getitem__）。

给 C++ 背景同学的关键点：
- Python 的 list 推导、字典、元组在这里用得很多；
- Dataset 对象本质是“按索引读取样本”的容器，DataLoader 会反复调用 __getitem__。
"""


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(filename: str) -> bool:
    # os.path.splitext("a.jpg") -> ("a", ".jpg")
    # lower() 统一小写，避免 .JPG 这类扩展名大小写问题。
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def _list_classes(root: str) -> List[str]:
    # 只保留子目录名（每个子目录视为一个类别）。
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    # 排序保证“类名 -> 索引”映射稳定可复现。
    return sorted(classes)


def _list_images_in_class(root: str, cls: str) -> List[str]:
    cls_path = os.path.join(root, cls)
    # 只收集图像文件名，不含路径。
    files = [f for f in os.listdir(cls_path) if _is_image_file(f)]
    # 排序让结果 deterministic。
    return sorted(files)


def _build_longtail_selection(
    root: str,
    classes: List[str],
    long_tail: bool,
    imb_factor: float,
    seed: int,
) -> Dict[str, List[str]]:
    # 独立随机数发生器，不污染全局 np.random 状态。
    rng = np.random.RandomState(seed)

    # img_max = 所有类别中样本数最大值。
    # 这一步相当于先统计“头部类上限”。
    img_max = 0
    for cls in classes:
        img_max = max(img_max, len(_list_images_in_class(root, cls)))

    if img_max == 0:
        raise RuntimeError(f"数据集根目录下未找到图像: {root}")

    if long_tail:
        # 按指数衰减生成每类样本量，模拟真实长尾分布。
        img_num_per_cls = get_img_num_per_cls(len(classes), img_max, imb_factor)
    else:
        # 不做长尾时，各类统一按 img_max 目标数量取样。
        img_num_per_cls = [img_max] * len(classes)

    selected = {}
    # enumerate 同时拿到索引和元素，等价于 C++ for (idx, value)
    for cls_idx, cls in enumerate(classes):
        files = _list_images_in_class(root, cls)
        # 每类内部先打乱，再按长尾目标数量截断。
        rng.shuffle(files)
        take_n = min(len(files), img_num_per_cls[cls_idx])
        # selected 的结构：{ "airplane": ["a1.jpg", ...], ... }
        selected[cls] = files[:take_n]

    return selected


def _load_split_file(path: str):
    # JSON 反序列化为 Python dict/list。
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_split_file(path: str, payload: dict):
    # ensure_ascii=False 让中文类名可读，不转义成 \uXXXX。
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _create_split(
    selected: Dict[str, List[str]],
    classes: List[str],
    query_ratio: float,
    seed: int,
):
    # 使用 seed+7 让“类内抽样随机性”与前面 longtail 选择阶段相对独立。
    rng = np.random.RandomState(seed + 7)

    query_split = {}
    db_split = {}

    for cls in classes:
        # [:] 是浅拷贝，避免就地打乱影响原 selected。
        files = selected[cls][:]
        rng.shuffle(files)

        if len(files) <= 1:
            # 样本太少时，全部给 query，避免负样本分割异常。
            n_query = len(files)
        else:
            # round 后再夹紧到 [1, len(files)-1]，保证 query 和 db 都非空。
            n_query = int(round(len(files) * query_ratio))
            n_query = max(1, min(len(files) - 1, n_query))

        # 前 n_query 个作为 query，剩余作为 db。
        query_split[cls] = files[:n_query]
        db_split[cls] = files[n_query:]

    return {"query": query_split, "db": db_split}


def _materialize_items(root: str, split: Dict[str, List[str]], class_to_idx: Dict[str, int]):
    # 把“类名 + 文件名”物化成“绝对/相对路径 + 数值标签”的样本列表。
    # 结构是 List[Tuple[path, label]]。
    items = []
    for cls, files in split.items():
        label = class_to_idx[cls]
        for name in files:
            items.append((os.path.join(root, cls, name), label))
    return items


def build_train_query_db_splits(
    root: str,
    long_tail: bool = True,
    imb_factor: float = 0.01,
    query_ratio: float = 0.2,
    split_path: str = "split_patternnet.json",
    seed: int = 42,
):
    # 1) 扫描类别并固定类名顺序。
    classes = _list_classes(root)
    # 2) 类别名映射到连续整数标签。
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # 3) 根据 long_tail 和 imb_factor 选择每类可用样本。
    selected = _build_longtail_selection(
        root=root,
        classes=classes,
        long_tail=long_tail,
        imb_factor=imb_factor,
        seed=seed,
    )

    split_data = None
    if split_path and os.path.exists(split_path):
        # 如果 split 文件存在，尝试复用。
        loaded = _load_split_file(split_path)
        loaded_classes = sorted(list(loaded.get("class_to_idx", {}).keys()))
        loaded_imb = float(loaded.get("imb_factor", -1.0))
        loaded_query_ratio = float(loaded.get("query_ratio", -1.0))
        # 浮点比较用“阈值近似”，避免直接 == 带来的精度问题。
        same_imb = abs(loaded_imb - float(imb_factor)) < 1e-12
        same_query = abs(loaded_query_ratio - float(query_ratio)) < 1e-12
        # 仅当类别集合、imb_factor、query_ratio 全部一致时复用旧划分。
        if loaded_classes == classes and same_imb and same_query:
            split_data = loaded.get("splits")

    if split_data is None:
        # 没有可复用 split 时，重新生成并落盘。
        split_data = _create_split(selected, classes, query_ratio=query_ratio, seed=seed)
        if split_path:
            payload = {
                "class_to_idx": class_to_idx,
                "query_ratio": query_ratio,
                "imb_factor": imb_factor,
                "splits": split_data,
            }
            _save_split_file(split_path, payload)

    # 4) 构造训练/查询/数据库样本列表。
    # 训练集必须与 query 严格隔离，避免检索评估时出现信息泄漏。
    # 因此这里使用 db 子集作为训练数据来源。
    train_items = _materialize_items(root, split_data["db"], class_to_idx)
    # query/db 用于检索评估划分。
    query_items = _materialize_items(root, split_data["query"], class_to_idx)
    db_items = _materialize_items(root, split_data["db"], class_to_idx)

    # 防泄漏检查：query 与 train/db 不能有任何样本重叠。
    query_set = {p for p, _ in query_items}
    db_set = {p for p, _ in db_items}
    train_set = {p for p, _ in train_items}
    if query_set & db_set:
        raise RuntimeError("数据划分错误：query 与 db 存在重叠样本。")
    if query_set & train_set:
        raise RuntimeError("数据泄漏：query 样本出现在训练集中。")

    return train_items, query_items, db_items, class_to_idx


class PatternNetDataset(Dataset):
    def __init__(self, root, transform=None, items: List[Tuple[str, int]] = None):
        # root: 数据根目录
        # transform: 图像增强/预处理函数（可调用对象）
        # items: 可选样本清单，若传入则按清单构建数据集
        self.root = root
        self.transform = transform

        # samples 存路径，labels 存整数标签，两者下标一一对应。
        self.samples = []
        self.labels = []

        if items is None:
            # 如果没传 items，就扫描整个 root 构建全集。
            classes = _list_classes(root)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for fname in _list_images_in_class(root, cls):
                    self.samples.append(os.path.join(root, cls, fname))
                    self.labels.append(class_to_idx[cls])
            self.class_to_idx = class_to_idx
        else:
            # 使用外部提供的样本列表（常用于 train/query/db 子集）。
            # 列表推导：从 [(path,label), ...] 拆出两个并行列表。
            self.samples = [x[0] for x in items]
            self.labels = [x[1] for x in items]
            # 外部 items 已是数值标签，这里不再维护类名字典。
            self.class_to_idx = {}

    def __len__(self):
        # len(dataset) 时会调用这里。
        return len(self.samples)

    def __getitem__(self, idx):
        # DataLoader 通过下标 idx 取样本。
        # PIL 读图后统一转 RGB，避免灰度图/带 alpha 图导致通道不一致。
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            # transform 通常把 PIL Image -> Tensor 并做归一化。
            img = self.transform(img)
        label = self.labels[idx]
        # 返回 (图像张量, 标签)，与训练循环 for img, label in loader 对应。
        return img, label

    def class_distribution(self):
        # 返回各标签计数，便于检查长尾分布是否生效。
        return Counter(self.labels)
