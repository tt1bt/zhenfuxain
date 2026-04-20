import torch
import torch.nn as nn
import torchvision.models as models


class HashModel(nn.Module):
    def __init__(self, hash_bits=32, num_classes=38, pretrained=True):
        super().__init__()

        if pretrained:
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            backbone = models.resnet34(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.hash_layer = nn.Linear(512, hash_bits)
        self.classifier = nn.Linear(hash_bits, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)

        hash_code = torch.tanh(self.hash_layer(features))
        logits = self.classifier(hash_code)

        return hash_code, logits
