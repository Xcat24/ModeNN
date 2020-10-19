import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .BaseModel import BaseModel
from layer import BreakupConv

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
        super(double_conv_basic, self).__init__()
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
        super(single_conv_basic, self).__init__()
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

        self.initconv = conv3x3(self.hparams.in_channel, self.hparams.out_channels[0], stride=1)
        self.convs = self._make_layer()

        #TODO
        conv_outshape = self.hparams.conv_outshape
        self.bn1 = nn.BatchNorm2d(self.hparams.out_channels[-1], momentum=0.9)
        self.fc = self._make_dense()
        self.out_layer = nn.Linear(self.hparams.dense_nodes[-1], self.hparams.num_classes)

    def _make_layer(self):
        layers = []
        for _ in range(1, len(self.hparams.out_channels)):
            if self.hparams.basic_mode == 'single':
                layers.append(single_conv_basic(self.hparams.out_channels[_-1], self.hparams.out_channels[_], self.hparams.dropout, 3, self.hparams.stride))
            elif self.hparams.basic_mode == 'double':
                layers.append(double_conv_basic(self.hparams.out_channels[_-1], self.hparams.out_channels[_], self.hparams.dropout, 3, self.hparams.stride))

        return nn.Sequential(*layers)

    def _make_dense(self):
        layers = [nn.Linear(self.hparams.conv_outshape, self.hparams.dense_nodes[0])]
        layers.append(nn.ReLU())
        if len(self.hparams.dense_nodes) > 1:
            for _ in range(1, len(self.hparams.dense_nodes)):
                layers.append(nn.Linear(self.hparams.dense_nodes[_-1],self.hparams.dense_nodes[_]))
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.initconv(x)
        if self.hparams.pooling:
            out = F.max_pool2d(out, self.hparams.pool_shape)
        out = self.convs(out)
        out = F.relu(self.bn1(out))
        if self.hparams.pooling:
            out = F.max_pool2d(out, self.hparams.pool_shape)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.out_layer(out)

        return out
    
    def conv_forward(self, x):
        out = self.initconv(x)
        out = self.convs(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.hparams.pool_shape)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        return out
    
    def dense_forward(self, x):
        out = self.initconv(x)
        out = self.convs(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.hparams.pool_shape)
        # print(out.shape)
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
     
    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        conv_out = self.conv_forward(x)
        dense_out = self.dense_forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
            'conv_out': conv_out,
            'dense_out': dense_out,
            'data': x,
            'label': y
        }

        return output

    def test_end(self, outputs):
        whole_test_data = torch.cat([x['data'].reshape((-1,3072)) for x in outputs], dim=0)
        whole_test_label = torch.cat([x['label'] for x in outputs], dim=0)
        whole_conv_out = torch.cat([x['conv_out'] for x in outputs], dim=0)
        whole_dense_out = torch.cat([x['dense_out'] for x in outputs], dim=0)
        #logger
        if self.logger:
            self.logger.experiment.add_embedding(whole_test_data, whole_test_label, tag='raw data')
            self.logger.experiment.add_embedding(whole_conv_out, whole_test_label, tag='CNN-conv-out data')
            self.logger.experiment.add_embedding(whole_dense_out, whole_test_label, tag='CNN-dense-out data')

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='MyConv2D', type=str, 
                            help='networ architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
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
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
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
        parser.add_argument('--pooling', dest='pooling', action='store_true',
                                help='whether to use pooling after conv layer')
        parser.add_argument('--pool-shape', default=2, type=int,
                                help='average pooling shape')
        parser.add_argument('--conv-outshape', default=1, type=int,
                                help='dimentions of conv layer output data')
        return parser


class BreakupConv2D(BaseModel):
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(BreakupConv2D, self).__init__(hparams=hparams, loss=loss)

        self.conv1 = BreakupConv(hparams.input_size, hparams.kernel_size, hparams.out_channels[0], hparams.in_channel)
        self.bn1 = nn.BatchNorm2d(hparams.out_channels[0])

        self.conv2 = BreakupConv((13, 13), hparams.kernel_size, hparams.out_channels[1], hparams.out_channels[0])
        self.bn2 = nn.BatchNorm2d(hparams.out_channels[1])

        self.fc = self._make_dense()
        self.out_layer = nn.Linear(self.hparams.dense_nodes[-1], self.hparams.num_classes)

    def _make_dense(self):
        layers = [nn.Linear(self.hparams.conv_outshape, self.hparams.dense_nodes[0])]
        layers.append(nn.ReLU())
        if len(self.hparams.dense_nodes) > 1:
            for _ in range(1, len(self.hparams.dense_nodes)):
                layers.append(nn.Linear(self.hparams.dense_nodes[_-1],self.hparams.dense_nodes[_]))
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.hparams.pooling:
            out = F.max_pool2d(out, self.hparams.pool_shape)
        out = F.relu(self.bn2(self.conv2(out)))
        if self.hparams.pooling:
            out = F.max_pool2d(out, self.hparams.pool_shape)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.out_layer(out)

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
    
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        conv_out = self.conv_forward(x)
        dense_out = self.dense_forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
            'conv_out': conv_out,
            'dense_out': dense_out,
            'data': x,
            'label': y
        }

        return output

    def test_end(self, outputs):
        whole_test_data = torch.cat([x['data'].reshape((-1,3072)) for x in outputs], dim=0)
        whole_test_label = torch.cat([x['label'] for x in outputs], dim=0)
        whole_conv_out = torch.cat([x['conv_out'] for x in outputs], dim=0)
        whole_dense_out = torch.cat([x['dense_out'] for x in outputs], dim=0)
        #logger
        if self.logger:
            self.logger.experiment.add_embedding(whole_test_data, whole_test_label, tag='raw data')
            self.logger.experiment.add_embedding(whole_conv_out, whole_test_label, tag='CNN-conv-out data')
            self.logger.experiment.add_embedding(whole_dense_out, whole_test_label, tag='CNN-dense-out data')

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='MyConv2D', type=str, 
                            help='networ architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
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
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
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
        parser.add_argument('--pooling', dest='pooling', action='store_true',
                                help='whether to use pooling after conv layer')
        parser.add_argument('--pool-shape', default=2, type=int,
                                help='average pooling shape')
        parser.add_argument('--conv-outshape', default=1, type=int,
                                help='dimentions of conv layer output data')
        return parser