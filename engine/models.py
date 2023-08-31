import segmentation_models_pytorch as smp
import torch.nn as nn
import torchsummary
import torchvision


class Resnet50Flower102(nn.Module):
    def __init__(self, pretrained=True, freeze_Resnet=True):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=pretrained)
        if freeze_Resnet:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 102),
        )

    def forward(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)


class SegmentationResNet50(nn.Module):
    def __init__(
        self, classes: int, activation=None, encoder_weights="imagenet", device="cuda"
    ):
        super().__init__()
        self.model = smp.Unet(
            "resnet50",
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        ).to(device)

        # * show input and output shapes
        self.model.classification_head = nn.Sequential(
            # * input shape is (batch_size, 2048, 7, 7)
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 102, kernel_size=1),
            nn.BatchNorm2d(102),
            nn.ReLU(),
            # * output shape is (batch_size, 102, 7, 7)
            nn.Flatten(),
            nn.Linear(7 * 7 * 102, 102),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
