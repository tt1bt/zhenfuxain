import torch
import torch.nn as nn
import torch.nn.functional as F


"""
向心损失（Centripetal Loss，教学注释版）。

核心思想：
1) 为每个类别维护一个“可学习中心向量”（centers）。
2) 每个样本的哈希向量与所有类别中心做相似度打分。
3) 用交叉熵推动样本更接近“自己类别中心”，远离其它类别中心。

直观上，这会让同类样本在哈希空间中更集中。
"""


class CentripetalLoss(nn.Module):
    def __init__(self, num_classes, hash_bits, gamma=1.0, temperature=0.1, center_init_std=0.02, initial_centers=None):
        super().__init__()
        # gamma 是损失缩放系数，便于和其他损失项加权。
        self.gamma = gamma
        # temperature 越小，softmax 前的分布越“尖锐”。
        self.temperature = temperature
        # 为每个类别维护一个哈希中心向量。
        # 若提供 initial_centers，则按论文公式的类别均值初始化；否则随机初始化。
        if initial_centers is not None:
            init = torch.as_tensor(initial_centers, dtype=torch.float32)
            if init.shape != (num_classes, hash_bits):
                raise ValueError(
                    f"initial_centers 形状应为 ({num_classes}, {hash_bits})，实际为 {tuple(init.shape)}"
                )
            self.centers = nn.Parameter(init.clone())
        else:
            # nn.Parameter 表示这是可训练参数，会被优化器更新。
            self.centers = nn.Parameter(torch.randn(num_classes, hash_bits) * center_init_std)

    def forward(self, hash_code, labels):
        # hash_code: [B, hash_bits]
        # labels: [B]

        # 归一化后点积等价于余弦相似度，便于稳定优化。
        # F.normalize(..., dim=1) 表示按“特征维”归一化每个样本。
        hash_code = F.normalize(hash_code, dim=1)
        centers = F.normalize(self.centers, dim=1)

        # 以类别中心为“分类原型”，鼓励样本靠近自身类别中心。
        # centers.t() 是转置：[C, D] -> [D, C]
        # 矩阵乘法后 logits 形状为 [B, C]，可直接喂给 cross_entropy。
        logits = torch.matmul(hash_code, centers.t()) / self.temperature

        # 用可微分的 CE 近似优化“向心聚合”目标。
        # labels 中每个值是正确类别索引。
        loss = F.cross_entropy(logits, labels)
        # 返回缩放后的最终损失值。
        return self.gamma * loss
