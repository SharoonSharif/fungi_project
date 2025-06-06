# src/model.py

import torch.nn as nn
import torchvision.models as models

class FungiClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FungiClassifier, self).__init__()
        # Load a pretrained ResNet18 backbone
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        # Replace the final FC layer with one matching our num_classes
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
