import numpy as np


"""
长尾分布工具（教学注释版）。

本文件只做一件事：
给定类别数 cls_num、头部类样本上限 img_max、不平衡因子 imb_factor，
按指数衰减公式生成每个类别应该采样多少张图。

典型规律：
- 第 0 类接近 img_max（头部类）
- 最后一个类接近 img_max * imb_factor（尾部类）
"""


def get_img_num_per_cls(cls_num, img_max, imb_factor):
    """根据指数长尾分布计算每个类别的样本数。"""
    # 边界情况：类别数 <= 0，直接返回空列表。
    if cls_num <= 0:
        return []

    # 只有 1 个类别时，不存在长尾，直接返回 img_max。
    if cls_num == 1:
        return [int(img_max)]

    img_num_per_cls = []
    for cls_idx in range(cls_num):
        # 指数衰减公式：
        # num_i = img_max * imb_factor^(i / (cls_num - 1))
        # i=0 时为 img_max，i=cls_num-1 时约为 img_max*imb_factor。
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        # 至少取 1，避免某些类别完全没有样本。
        img_num_per_cls.append(max(1, int(num)))

    return img_num_per_cls
