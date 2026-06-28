import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


"""
哈希模型定义（教学注释版）。

C++ 背景同学可这样理解：
1) 这个类继承 nn.Module，类似“继承一个神经网络基类”，并重写前向函数。
2) 模型由三段组成：
    - features: ResNet34 主干，负责提特征；
    - hash_layer: 线性层，把 512 维特征映射到 hash_bits 维；
    - classifier: 用哈希向量做分类监督。
3) forward 返回两个值：(hash_code, logits)。
"""


class IAM(nn.Module):
    """集中注意力模块：共享全连接层 + tanh + 注意力参数 gamma + 拼接输出。"""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(1, in_features // 2)

        # f1 和 f2 共享同一组参数，符合论文“初始状态共享参数”的描述。
        self.shared_fc = nn.Linear(in_features, hidden_features)
        # f3 用于把拼接后的增强特征映射回原始特征维度，方便后续 hash_layer 接收。
        self.output_fc = nn.Linear(hidden_features * 2, in_features)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]，对应论文中的初始特征 d。
        d_f1 = self.shared_fc(x)
        d_f2 = self.shared_fc(x)

        # tanh 对应论文中的 b1、b2，保留连续可导特性，避免 sign 阻断反传。
        b1 = self.activation(d_f1)
        b2 = self.activation(d_f2)

        # gamma = b1 * b2，逐元素相乘，突出同符号/一致响应的特征维度。
        gamma = b1 * b2

        # 将注意力参数加回两路特征，得到增强后的 d_gamma1 和 d_gamma2。
        d_gamma1 = d_f1 + gamma
        d_gamma2 = d_f2 + gamma

        # concat 对应论文中的特征拼接。
        d_gamma = torch.cat([d_gamma1, d_gamma2], dim=1)

        # f3 输出最终的 IAM 特征 d_f3。
        d_f3 = self.output_fc(d_gamma)
        return d_f3


class HashModel(nn.Module):
    def __init__(self, hash_bits=32, num_classes=38, pretrained=True):
        # super().__init__() 等价于 C++ 里显式调用父类构造。
        super().__init__()

        # 兼容不同版本的 torchvision：
        # - 新版 torchvision 使用 `weights=...` 与 ResNet34_Weights 枚举；
        # - 旧版 torchvision 使用 `pretrained=` 布尔参数。
        try:
            if pretrained:
                backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                backbone = models.resnet34(weights=None)
        except Exception:
            # 回退到旧接口，避免因 API 变动导致的异常（例如 TypeError/NameError）。
            backbone = models.resnet34(pretrained=bool(pretrained))

        # 保留到 layer4，去掉 avgpool/fc，得到空间特征图以便 IAM 发挥作用。
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.iam = IAM(in_features=512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 线性层：512 -> hash_bits。
        self.hash_layer = nn.Linear(512, hash_bits)
        # 分类层：hash_bits -> num_classes。
        self.classifier = nn.Linear(hash_bits, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]，B 是 batch 大小。

        # 主干输出空间特征图 [B, 512, H', W']。
        features = self.features(x)
        # 全局池化回到 [B, 512, 1, 1]。
        features = self.pool(features)
        # flatten(features, 1) 表示从维度 1 开始展平。
        # [B, 512, 1, 1] -> [B, 512]。
        features = torch.flatten(features, 1)

        # IAM 处理向量特征，符合论文中“初步特征 d 输入 IAM”的公式描述。
        features = self.iam(features)

        # tanh 将连续值压到 [-1,1]，为后续 sign 二值化做近似。
        # 为什么不用 sign 直接二值化？因为 sign 不可导，训练难以稳定反传。
        hash_code = torch.tanh(self.hash_layer(features))
        # 分类头提供类别监督，驱动哈希特征具备判别性。
        logits = self.classifier(hash_code)

        # Python 支持“多返回值”，调用时可写：hash_code, logits = model(x)
        return hash_code, logits
