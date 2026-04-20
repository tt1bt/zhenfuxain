import torch
import torch.nn as nn


class CentripetalLoss(nn.Module):
    def __init__(self, num_classes, hash_bits, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.centers = nn.Parameter(torch.randn(num_classes, hash_bits) * 0.02)

    def forward(self, hash_code, labels):
        centers_batch = self.centers[labels]
        loss = (hash_code - centers_batch).pow(2).mean()
        return self.gamma * loss
