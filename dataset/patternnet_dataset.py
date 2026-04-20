import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.longtail_dataset import get_img_num_per_cls


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def _list_classes(root: str) -> List[str]:
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(classes)


def _list_images_in_class(root: str, cls: str) -> List[str]:
    cls_path = os.path.join(root, cls)
    files = [f for f in os.listdir(cls_path) if _is_image_file(f)]
    return sorted(files)


def _build_longtail_selection(
    root: str,
    classes: List[str],
    long_tail: bool,
    imb_factor: float,
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.RandomState(seed)

    img_max = 0
    for cls in classes:
        img_max = max(img_max, len(_list_images_in_class(root, cls)))

    if img_max == 0:
        raise RuntimeError(f"No images found in dataset root: {root}")

    if long_tail:
        img_num_per_cls = get_img_num_per_cls(len(classes), img_max, imb_factor)
    else:
        img_num_per_cls = [img_max] * len(classes)

    selected = {}
    for cls_idx, cls in enumerate(classes):
        files = _list_images_in_class(root, cls)
        rng.shuffle(files)
        take_n = min(len(files), img_num_per_cls[cls_idx])
        selected[cls] = files[:take_n]

    return selected


def _load_split_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_split_file(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _create_split(
    selected: Dict[str, List[str]],
    classes: List[str],
    query_ratio: float,
    seed: int,
):
    rng = np.random.RandomState(seed + 7)

    query_split = {}
    db_split = {}

    for cls in classes:
        files = selected[cls][:]
        rng.shuffle(files)

        if len(files) <= 1:
            n_query = len(files)
        else:
            n_query = int(round(len(files) * query_ratio))
            n_query = max(1, min(len(files) - 1, n_query))

        query_split[cls] = files[:n_query]
        db_split[cls] = files[n_query:]

    return {"query": query_split, "db": db_split}


def _materialize_items(root: str, split: Dict[str, List[str]], class_to_idx: Dict[str, int]):
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
    classes = _list_classes(root)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    selected = _build_longtail_selection(
        root=root,
        classes=classes,
        long_tail=long_tail,
        imb_factor=imb_factor,
        seed=seed,
    )

    split_data = None
    if split_path and os.path.exists(split_path):
        loaded = _load_split_file(split_path)
        loaded_classes = sorted(list(loaded.get("class_to_idx", {}).keys()))
        loaded_imb = float(loaded.get("imb_factor", -1.0))
        loaded_query_ratio = float(loaded.get("query_ratio", -1.0))
        same_imb = abs(loaded_imb - float(imb_factor)) < 1e-12
        same_query = abs(loaded_query_ratio - float(query_ratio)) < 1e-12
        if loaded_classes == classes and same_imb and same_query:
            split_data = loaded.get("splits")

    if split_data is None:
        split_data = _create_split(selected, classes, query_ratio=query_ratio, seed=seed)
        if split_path:
            payload = {
                "class_to_idx": class_to_idx,
                "query_ratio": query_ratio,
                "imb_factor": imb_factor,
                "splits": split_data,
            }
            _save_split_file(split_path, payload)

    query_items = _materialize_items(root, split_data["query"], class_to_idx)
    db_items = _materialize_items(root, split_data["db"], class_to_idx)
    train_items = db_items[:]

    return train_items, query_items, db_items, class_to_idx


class PatternNetDataset(Dataset):
    def __init__(self, root, transform=None, items: List[Tuple[str, int]] = None):
        self.root = root
        self.transform = transform

        self.samples = []
        self.labels = []

        if items is None:
            classes = _list_classes(root)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for fname in _list_images_in_class(root, cls):
                    self.samples.append(os.path.join(root, cls, fname))
                    self.labels.append(class_to_idx[cls])
            self.class_to_idx = class_to_idx
        else:
            self.samples = [x[0] for x in items]
            self.labels = [x[1] for x in items]
            self.class_to_idx = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def class_distribution(self):
        return Counter(self.labels)
