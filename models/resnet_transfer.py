import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model