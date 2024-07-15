import torch
from torch import nn
from torchvision import models

class DogEmotionResNet(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        
        self.resnet = models.resnet50(weights=weights)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
