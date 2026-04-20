import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(class_counts, mode="sqrt_inv", cb_beta=0.9999):
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)

    if mode == "none":
        return np.ones_like(counts, dtype=np.float64)

    if mode == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    elif mode == "class_balanced":
        effective_num = 1.0 - np.power(cb_beta, counts)
        effective_num = np.maximum(effective_num, 1e-12)
        weights = (1.0 - cb_beta) / effective_num
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    weights = weights / np.sum(weights) * len(weights)
    return weights


class IAMLoss(nn.Module):
    def __init__(self, class_counts, mode="sqrt_inv", cb_beta=0.9999):
        super().__init__()
        weights = compute_class_weights(class_counts, mode=mode, cb_beta=cb_beta)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, pred, label):
        return self.loss_fn(pred, label)
