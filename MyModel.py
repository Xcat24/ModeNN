import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import pytorch_lightning as pl
from layer import DescartesExtension, MaskDE, LocalDE, SLConv, Mode, MaskLayer
from torch.utils.data import Dataset, DataLoader
from myutils.datasets import ORLdataset, NumpyDataset
from myutils.utils import compute_cnn_out, compute_5MODE_dim, compute_mode_dim, Pretrain_Mask, find_polyitem
from sota_module import resnet, Wide_ResNet
from matplotlib import pyplot as plt




class C_MODENN(BaseModel):
    def __init__(self, input_size, in_channel, out_channel, order, num_classes, dataset, learning_rate=0.001, weight_decay=0.001,
                     share_fc_weights=False, loss=nn.CrossEntropyLoss(), dropout=0, lr_milestones=[60,120,160], norm=None, log_weight=50):
        super(C_MODENN,self).__init__(loss, dataset)
        
        self.dropout = dropout
        self.norm = norm
        self.log_weight=log_weight
        self.learning_rate = learning_rate
        self.lr_milestones = lr_milestones
        self.weight_decay = weight_decay
        self.share_fc_weights = share_fc_weights
        self.num_classes = num_classes

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1)
        self.pooling = nn.MaxPool2d((4,4))
        print('{} order Descartes Extension'.format(order))
        DE_dim = compute_mode_dim([64 for _ in range(order-1)]) + 64
        self.de_layer = Mode(order_dim=[64 for _ in range(order-1)])

        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        if self.norm:
            self.norm_layer = nn.BatchNorm1d(DE_dim)
        if share_fc_weights:
            self.fc = nn.Linear(DE_dim, num_classes)#公用一个fc权值矩阵,导致loss值巨大，最后变为nan
        else:
            self.fc = nn.ModuleList([nn.Linear(DE_dim, num_classes) for _ in range(out_channel)])
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_out = self.conv(x) #shape=(batch_size, 16, 32, 32)
        conv_out = self.relu(conv_out)
        conv_out = self.pooling(conv_out)
        out_sum = []
        for i in range(conv_out.size()[1]): #能否优化？不用for循环？
            de_in = conv_out[:,i,:,:]
            origin = torch.flatten(de_in, 1)
            de_out = self.de_layer(origin)
            de_out = torch.cat([origin, de_out], dim=-1)

            if self.norm:
                de_out = self.norm_layer(de_out)
            if self.dropout:
                de_out = self.dropout_layer(de_out)
            if self.share_fc_weights:
                de_out = self.fc(de_out)
            else:
                de_out = self.fc[i](de_out)
            out_sum.append(torch.unsqueeze(de_out, dim=0))
        out = torch.sum(torch.cat(out_sum), dim=0)
        return out

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9, nesterov=True)
        return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.lr_milestones, gamma=0.2)]

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss.item(), 'val_acc': '{0:.5f}'.format(avg_acc.item())}
        log_dict = ({'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()})

        return {
            'avg_val_loss': avg_loss,
            'val_acc': avg_acc,
            'progress_bar': tqdm_dict,
            'log': log_dict
            }

class resnext29(BaseModel):
    def __init__(self, input_size, in_channel, num_classes, loss=nn.CrossEntropyLoss(), dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(resnext29, self).__init__()
        self.dataset = dataset
        self.loss = loss
        self.resnext29 = resnet.resnext29_16x64d(num_classes=num_classes)

    def forward(self, x):
        return self.resnext29(x)

    def configure_optimizers(self):
        return [torch.optim.SGD(self.parameters(),lr=0.1, weight_decay=0.0005, momentum=0.9)]


class resnet18(BaseModel):
    def __init__(self, num_classes, learning_rate=0.1, weight_decay=0.0005, loss=nn.CrossEntropyLoss(), dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(resnet18, self).__init__(loss=loss, dataset=dataset)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.resnet18 = resnet.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet18(x)

    def configure_optimizers(self):
        return [torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)]

class wide_resnet(BaseModel):
    def __init__(self, depth, width, dropout, num_classes, learning_rate=0.1, weight_decay=0.0005, 
                loss=nn.CrossEntropyLoss(), dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':128, 'train_transform':None, 'val_transform':None}):
        super(wide_resnet, self).__init__(loss=loss, dataset=dataset)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.wide_resnet = Wide_ResNet.wide_resnet(depth, width, dropout, num_classes)

    def forward(self, x):
        return self.wide_resnet(x)
    
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)]

class CIFARConv_MODENN(BaseModel):
    
    def __init__(self, input_size, in_channel, layer_num, dense_node, kernel_size, num_classes, order=2,
                     stride=1, padding=0, pooling='Max', pool_shape=(2,2), norm=False, dropout=None, learning_rate=0.001,
                     weight_decay=0.0001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(CIFARConv_MODENN, self).__init__()
        self.dataset = dataset
        self.loss = loss
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.order = order
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        DE_dim = int(math.factorial(2048 + order - 1)/(math.factorial(order)*math.factorial(2048 - 1)))
        self.fc = nn.Linear(DE_dim, num_classes)
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de = DescartesExtension(order=order)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.pooling == 'Max':
            self.maxpool = nn.MaxPool2d(kernel_size=pool_shape)
        if self.pooling == 'Avg':
            self.avgpool = nn.AvgPool2d(kernel_size=pool_shape)
        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)


    def forward(self, x):
        out = self.conv1(x)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)
        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)

        out = self.conv2(out)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)
        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)
        
        out = self.conv3(out)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)
        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        if self.dropout:
            out = self.dropout_layer(out)
        out = self.de(out)
        out = self.relu(out)
        out = self.fc(out)
        # out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class MyCNN_MODENN(BaseModel):
   
    def __init__(self, input_size, in_channel, out_channel, kernel_size, num_classes, order=2, stride=1, padding=0, pooling='Max',
                     pool_shape=(2,2), norm=False, dropout=None, learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}, output_debug=False):
        super(MyCNN_MODENN, self).__init__()
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.output_debug = output_debug
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.pooling == 'Max':
            self.maxpool = nn.MaxPool2d(kernel_size=pool_shape)
        if self.pooling == 'Avg':
            self.avgpool = nn.AvgPool2d(kernel_size=pool_shape)
        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        print('{} order Descartes Extension'.format(order))
        de_in = (input_size[0]//(pool_shape[0]**2))*(input_size[1]//(pool_shape[1]**2))*out_channel
        DE_dim = int(math.factorial(de_in + order - 1)/(math.factorial(order)*math.factorial(de_in - 1)))
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de = DescartesExtension(order=order)
        self.fc = nn.Linear(DE_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        out = self.conv1(x)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)

        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)

        out = self.conv2(out)
        # self.data_statics('output of conv2', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)

        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        # self.data_statics('output of flatten', out, verbose=self.output_debug)
        out = self.de(out)
        # self.data_statics('output of de', out, verbose=self.output_debug)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout_layer(out)
        # self.data_statics('output of de_tanh', out, verbose=self.output_debug)
        out = self.fc(out)
        # self.data_statics('output of network', out, verbose=self.output_debug)
        # out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class SLCNN_MODENN(BaseModel):
       
    def __init__(self, input_size, in_channel, num_classes, order=2, stride=1, padding=0, pooling='Max', pool_shape=(2,2),
                     norm=False, dropout=None, learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}, output_debug=False):
        super(SLCNN_MODENN, self).__init__()
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.output_debug = output_debug
        self.conv1 = SLConv(in_channel, stride=stride, padding=padding)
        if self.pooling == 'Max':
            self.maxpool = nn.MaxPool2d(kernel_size=pool_shape)
        if self.pooling == 'Avg':
            self.avgpool = nn.AvgPool2d(kernel_size=pool_shape)
        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        print('{} order Descartes Extension'.format(order))
        de_in = (input_size[0]//(pool_shape[0]))*(input_size[1]//(pool_shape[1]))*3
        DE_dim = int(math.factorial(de_in + order - 1)/(math.factorial(order)*math.factorial(de_in - 1)))
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de = DescartesExtension(order=order)
        self.fc = nn.Linear(DE_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        out = self.conv1(x)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)

        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        # self.data_statics('output of flatten', out, verbose=self.output_debug)
        out = self.de(out)
        # self.data_statics('output of de', out, verbose=self.output_debug)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout_layer(out)
        # self.data_statics('output of de_tanh', out, verbose=self.output_debug)
        out = self.fc(out)
        # self.data_statics('output of network', out, verbose=self.output_debug)
        # out = self.softmax(out)

        return out

    
    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class SLCNN(BaseModel):
       
    def __init__(self, input_size, in_channel, num_classes, stride=1, padding=0, pooling='Max', pool_shape=(2,2),
                     norm=False, dropout=None, learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}, output_debug=False):
        super(SLCNN, self).__init__()
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.output_debug = output_debug
        self.conv1 = SLConv(in_channel, stride=stride, padding=padding)
        if self.pooling == 'Max':
            self.maxpool = nn.MaxPool2d(kernel_size=pool_shape)
        if self.pooling == 'Avg':
            self.avgpool = nn.AvgPool2d(kernel_size=pool_shape)
        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3*7*7, 128)
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        out = self.conv1(x)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)

        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        # self.data_statics('output of flatten', out, verbose=self.output_debug)
        out = self.fc1(out)
        # self.data_statics('output of de', out, verbose=self.output_debug)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout_layer(out)
        # self.data_statics('output of de_tanh', out, verbose=self.output_debug)
        out = self.fc(out)
        # self.data_statics('output of network', out, verbose=self.output_debug)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class NoHiddenBase(BaseModel):
    def __init__(self, input_size, num_classes, norm=False, dropout=None, learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}, output_debug=False):
        super(NoHiddenBase,self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.output_debug = output_debug

        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        
        if len(input_size) == 2:
            fc_in_dim = input_size[0]*input_size[1]
        elif len(input_size) == 3:
            fc_in_dim = input_size[0]*input_size[1]*input_size[2]
        elif len(input_size) == 1:
            fc_in_dim = input_size[0]

        self.fc = nn.Linear(fc_in_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        out = torch.flatten(x, 1)
        out = self.fc(out)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)

        if self.dropout:
            out = self.dropout_layer(out)

        # self.data_statics('output of network', out, verbose=self.output_debug)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class OneHiddenBase(BaseModel):
    def __init__(self, input_size, num_classes, norm=False, dropout=None, learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}, output_debug=False):
        super(OneHiddenBase,self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.output_debug = output_debug

        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        
        if len(input_size) == 2:
            fc_in_dim = input_size[0]*input_size[1]
        elif len(input_size) == 3:
            fc_in_dim = input_size[0]*input_size[1]*input_size[2]

        self.hiddenfc = nn.Linear(fc_in_dim, fc_in_dim)
        self.fc = nn.Linear(fc_in_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        out = torch.flatten(x, 1)
        out = self.hiddenfc(out)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        out = self.tanh(out)

        if self.dropout:
            out = self.dropout_layer(out)

        out = self.fc(out)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        
        if self.dropout:
            out = self.dropout_layer(out)

        # self.data_statics('output of network', out, verbose=self.output_debug)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class Pretrain_5MODENN(BaseModel):
    '''
    pick input dim according to the pretrained model's weight, select the dims related to the largest bins_num*bins_size weight,
    then split into bins_num group, imply 5 order DE layer to each group and then concat together. 
    Need to cooperate with the data transform of Pretrain_Select().
    Input size: (N, bins_num, bins_size)
    '''
    def __init__(self, num_classes, bins_size=9, bins_num=35, dropout=None, output_debug=False,
                     learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(), 
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(Pretrain_5MODENN, self).__init__()
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.bins_size = bins_size
        self.bins_num = bins_num
        self.output_debug = output_debug

        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
 
        self.de2 = DescartesExtension(order=2)
        self.de3 = DescartesExtension(order=3)
        self.de4 = DescartesExtension(order=4)
        self.de5 = DescartesExtension(order=5)

        DE_dim = compute_5MODE_dim(bins_size)*bins_num
        self.fc = nn.Linear(DE_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        x = torch.flatten(x, 1)
        temp = []
        for i in range(self.bins_num):
            origin = x[:,i*self.bins_size:(i+1)*self.bins_size]
            de2_out = self.de2(origin)
            de3_out = self.de3(origin)
            de4_out = self.de4(origin)
            de5_out = self.de5(origin)
            temp.append(torch.cat([origin, de2_out, de3_out, de4_out, de5_out], dim=-1))

        out = torch.cat(temp, dim=-1)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        out = self.tanh(out)
       
        if self.dropout:
            out = self.dropout_layer(out)
        # self.data_statics('output of de_tanh', out, verbose=self.output_debug)
        out = self.fc(out)
        # self.data_statics('output of network', out, verbose=self.output_debug)
        # out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]

    
class Select_MODE(BaseModel):
    '''
    对输入数据进行多阶笛卡尔扩张操作，为了避免在扩张过程中高阶数扩张结果维度过大，不同的阶数选择不同的输入维度
    '''
    def __init__(self, input_size, num_classes, model_path, order_dim=[300, 50, 20, 10], dropout=None, norm=False, output_debug=False,
                     learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(), 
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(Select_MODE, self).__init__()
        self.dropout = dropout
        self.norm = norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.mask = MaskLayer(Pretrain_Mask(model_path=model_path, num=order_dim[0]))
        self.mode = Mode(order_dim=order_dim)
        self.output_debug = output_debug
        DE_dim = compute_mode_dim(order_dim) + input_size

        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        if self.norm:
            self.norm_layer = nn.BatchNorm1d(DE_dim)

        self.fc = nn.Linear(DE_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        x = torch.flatten(x,1)
        select_x = self.mask(x)
        de = self.mode(select_x)
        out = torch.cat([x, de], dim=-1)
        if self.norm:
            out = self.norm_layer(out)
        if self.dropout:
            out = self.dropout_layer(out)
        # self.data_statics('output of de_tanh', out, verbose=self.output_debug)
        out = self.fc(out)
        # self.data_statics('output of network', out, verbose=self.output_debug)
        # out = self.softmax(out)
        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]

