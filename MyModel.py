import math
import torch
from torch import nn
from layer import DescartesExtension, MaskDE, LocalDE

class ModeNN(nn.Module):
    def __init__(self, input_dim, order, num_classes):
        super(ModeNN, self).__init__()
        print('{} order Descartes Extension'.format(order))
        DE_dim = int(math.factorial(input_dim + order - 1)/(math.factorial(order)*math.factorial(input_dim - 1)))
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de = DescartesExtension(order=order)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(DE_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.de(x)
        out = self.tanh(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out