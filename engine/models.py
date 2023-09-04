import torch.nn as nn
import torchsummary
import torchvision


class Resnet50Flower102(nn.Module):
    def __init__(self, device, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.device = device
        
        if pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        else: 
            weights = None
            
        self.model = torchvision.models.resnet50(weights=weights)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 102),
            
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)


