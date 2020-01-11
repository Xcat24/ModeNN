import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .BaseModel import BaseModel
from myutils.utils import compute_cnn_out, compute_5MODE_dim, compute_mode_dim, Pretrain_Mask, find_polyitem
from layer import DescartesExtension, MaskDE, LocalDE, SLConv, Mode, MaskLayer

def conv3x3(in_channel, out_channels, stride=1):
    return nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

def conv5x5(in_channel, out_channels, stride=1):
    return nn.Conv2d(in_channel, out_channels, kernel_size=5, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class double_conv_basic(nn.Module):
    r"""参考Wide-ResNet中的 wide_basic编写，去掉了其中的short-cut部分
    """
    def __init__(self, in_channel, out_channels, dropout_rate, kernel_size=3, stride=1):
        super(conv_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=True)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out

class single_conv_basic(nn.Module):
    r"""参考Wide-ResNet中的 wide_basic编写，去掉了其中的short-cut部分，以及dropout后半部分的conv，并在前半部分的conv中增加了stride
    """
    def __init__(self, in_channel, out_channels, dropout_rate, kernel_size=3, stride=1):
        super(conv_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        return out

class MyConv2D(BaseModel):
    #TODO fix the doc
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
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(MyConv2D, self).__init__(hparams=hparams, loss=loss)

        self.initconv = conv3x3(self.hparams.in_channel, self.hparams.out_channels, stride=self.hparams.stride)
        self.convs = self._make_layer()

        #TODO
        conv_outshape = self.hparams.conv_outshape
        self.bn1 = nn.BatchNorm2d(self.hparams.dense_nodes[0], momentum=0.9)
        self.fc = self._make_dense()

    def _make_layer():
        layers = []
        for _ in range(1, self.hparams.basic_num):
            if self.hparams.basic_mode == 'single':
                layers.append(single_conv_basic(self.hparams.out_channels[_-1], self.hparams.out_channels[_], self.hparams.dropout, 3, self.hparams.stride))
            elif self.hparams.basic_mode == 'double':
                layer.append(double_conv_basic(self.hparams.out_channels[_-1], self.hparams.out_channels[_], self.hparams.dropout, 3, self.hparams.stride))

        return nn.Sequential(*layers)

    def _make_dense():
        layers = [nn.Linear(self.hparams.conv_outshape, self.hparams.dense_nodes[0])]
        if len(self.hparams.dense_nodes) > 1:
            for _ in range(1, len(self.hparams.dense_num)):
                layers.append(nn.Linear(self.hparams.dense_nodes[_-1],self.hparams.dense_nodes[_]))
        layers.append(nn.Linear(self.hparams.dense_nodes[-1], self.hparams.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.initconv(x)
        out = self.convs(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.hparams.pool_shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def configure_optimizers(self):
        if self.hparams.opt == 'SGD':
            opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=True)
            if self.hparams.lr_milestones:
                return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
            else:
                return [opt]
        elif self.hparams.opt == 'Adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            if self.hparams.lr_milestones:
                return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
            else:
                return [opt]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='Conv', type=str, 
                            help='networ architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-milestones', nargs='+', type=int,
                                help='learning rate milestones')
        parser.add_argument('--lr-gamma', default=0.1, type=float,
                            help='number learning rate multiplied when reach the lr-milestones')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--dropout', default=0, type=float,
                                help='the rate of the dropout')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--log-weight', default=0, type=int,
                                help='log weight figure every x epoch')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
        parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data, now it refers to ORL dataset')
        #params in conv
        parser.add_argument('--kernel-size', nargs='+', type=int,
                                help='size of kernels, return as list, only support 3 or 5')
        parser.add_argument('--out-channels', nargs='+', type=int,
                                help='size of output channel, return as list, the length is 2 at least')
        parser.add_argument('--in-channel', default=3, type=int,
                                help='number of input channel')
        parser.add_argument('--stride', default=1, type=int,
                                help='stride')
        parser.add_argument('--dense-nodes', nargs='+', type=int,
                                help='numbers of dense layers nodels, return as list')
        parser.add_argument('--basic-mode', default='single', type=str,
                                help='conv basic to use')
        parser.add_argument('--basic-num', default=1, type=int,
                                help='how many conv basics to use')
        return parser
