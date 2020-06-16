import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
            self.out_dim = 256*2*2
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
            self.out_dim = 256*2*2
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            self.out_dim = 1024*2*2
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
            self.out_dim = 1024*2*2
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=pretrained)
            self.out_dim = 1024*2*2

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x