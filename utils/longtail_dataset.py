import numpy as np


def get_img_num_per_cls(cls_num, img_max, imb_factor):
    """Compute per-class sample counts for exponential long-tail distribution."""
    if cls_num <= 0:
        return []

    if cls_num == 1:
        return [int(img_max)]

    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(max(1, int(num)))

    return img_num_per_cls
