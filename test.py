import torch
import mymodels
from mymodels.ModeNN import ResNet

if __name__ == "__main__":
    net = ResNet('resnet152')
    img = torch.rand(2,3,32,32)
    y = net(img)
    print(y.size())