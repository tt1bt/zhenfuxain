import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
类平衡分类损失（教学注释版）。

目标：缓解长尾数据里“头部类样本太多，尾部类被忽视”的问题。

支持三种权重模式：
1) none: 每类权重都为 1，等价普通交叉熵。
2) sqrt_inv: 权重 = 1 / sqrt(n_c)，温和提升尾部类权重。
3) class_balanced: 基于有效样本数 E_n = 1 - beta^n 的重加权。
"""


def compute_class_weights(class_counts, mode="class_balanced", cb_beta=0.999, cb_mode="1"):
    """根据类别样本数预计算权重向量。"""

    # 保护下界，避免类别计数为 0 导致除零。
    # np.asarray 把输入统一转换为 numpy 数组。
    counts = np.asarray(class_counts, dtype=np.float64)
    # np.maximum 是逐元素 max(counts[i], 1.0)
    counts = np.maximum(counts, 1.0)

    if mode == "none":
        # 返回与 counts 同形状的全 1 向量。
        return np.ones_like(counts, dtype=np.float64)

    if mode == "sqrt_inv":
        # 开方倒数：比 1/n 更平滑，不至于把尾部权重放太大。
        weights = 1.0 / np.sqrt(counts)
    elif mode == "class_balanced":
        # 有效样本数 E_n = 1 - beta^n，样本越多边际增益越小。
        effective_num = 1.0 - np.power(cb_beta, counts)
        effective_num = np.maximum(effective_num, 1e-12)
        if cb_mode == "1-beta":
            # 原论文常见形式： (1-beta) / (1-beta^n)
            numerator = 1.0 - cb_beta
        elif cb_mode == "1":
            # 分子为 1 的变体会进一步放大尾部类别权重。
            numerator = 1.0
        else:
            raise ValueError(f"不支持的类平衡分子模式: {cb_mode}")
        weights = numerator / effective_num
    else:
        raise ValueError(f"不支持的权重模式: {mode}")

    # 返回 numpy 向量，后续在 IAMLoss 里转为 torch Tensor。
    return weights


class IAMLoss(nn.Module):
    def __init__(self, class_counts=None, class_weights=None, mode="class_balanced", cb_beta=0.999, cb_mode="1"):
        super().__init__()
        # 支持两种输入方式：
        # 1) 直接传 class_weights；
        # 2) 传 class_counts + 规则，在内部计算权重。
        if class_weights is None:
            if class_counts is None:
                raise ValueError("class_counts 和 class_weights 不能同时为空")
            class_weights = compute_class_weights(
                class_counts,
                mode=mode,
                cb_beta=cb_beta,
                cb_mode=cb_mode,
            )

        # register_buffer: 注册“随模型移动设备、保存到 state_dict，但不参与梯度更新”的张量。
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weights", weights)

    def forward(self, pred, label):
        # pred: [B, C]，label: [B]
        # reduction="none" 先保留逐样本 CE，后续手动乘权重再平均。
        per_sample_ce = F.cross_entropy(pred, label, reduction="none")
        # 按标签索引每个样本对应类别权重，再与 CE 逐样本相乘。
        # gather(0, label) 可理解为 weights[label[i]] 的向量化版本。
        sample_weights = self.weights.gather(0, label)
        # 最终返回 batch 的加权均值损失。
        return (per_sample_ce * sample_weights).mean()


class ClassBalancedCELoss(IAMLoss):
    """语义化别名：当前文件实现的是类别重加权 CE，而非注意力模块。"""

    pass
