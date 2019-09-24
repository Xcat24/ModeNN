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
        out = torch.flatten(x, 1)
        out = self.de(out)
        out = self.tanh(out)
        out = self.fc(out)
        # out = self.softmax(out)
        return out

class MyConv2D(nn.Module):
    r"""build a multi layer 2D CNN by repeating the same basic 2D CNN layer several times
    Args:
        in_channel (int): Number of channels in the input image
        out_channel (int): Number of channels produced by the convolution
        layer_num (int): Number of basic 2D CNN layer 
        kernel_size (int or tuple): Size of the convolving kernel of basic CNN layer
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        num_classes (int): Number of the node in last layer, the number of the classes
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        pooling (string or optional): Maxpooling or Avgpooling or not pooing. Default: ```Max```
        pool_shape(int or tuple, optional): Size of pooling
        norm (bool, optional): whether to use Batch Normalizaion. Default: ```Flase```
        dropout (float, optional): Percent of the dropout. Default:None
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where
        .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
    """
    def __init__(self, input_size, in_channel, out_channel, layer_num, dense_node, kernel_size, num_classes, stride=1, padding=0, 
                     pooling='Max', pool_shape=(2,2), norm=False, dropout=None):
        super(MyConv2D, self).__init__()
        self.layer_num = layer_num
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.initconv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_shape)
        self.avgpool = nn.AvgPool2d(kernel_size=pool_shape)
        self.norm = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear((input_size[0]//(pool_shape[0]**layer_num))*(input_size[1]//(pool_shape[1]**layer_num))*out_channel, dense_node)
        self.fc2 = nn.Linear(dense_node, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.initconv(x)
        if self.norm :
            out = self.norm(out)
        out = self.relu(out)
        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)

        for _ in range(self.layer_num - 1):
            out = self.conv(out)
            if self.norm :
                out = self.norm(out)
            out = self.relu(out)
            if self.pooling:
                if self.pooling == 'Max':
                    out = self.maxpool(out)
                elif self.pooling == 'Avg':
                    out = self.avgpool(out)

        if self.dropout:
            out = self.dropout(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out
