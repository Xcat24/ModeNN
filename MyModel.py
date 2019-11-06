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
from sota_module import resnet

class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        return {
            'loss': loss,
            'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
            'log': {'training_loss': loss.item()}
        }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'val_loss': loss,
            'val_acc': torch.tensor(val_acc) # everything must be a tensor
        }

        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss.item(), 'val_acc': '{0:.5f}'.format(avg_acc.item())}
        log_dict = {'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()}
       
        #logger
        if self.logger:
            layer_names = list(self._modules)
            for i in range(len(layer_names)):
                mod_para = list(self._modules[layer_names[i]].parameters())
                if mod_para:
                    for j in range(len(mod_para)):
                        w = mod_para[j].clone().detach()
                        weight_name=layer_names[i]+'_'+str(w.shape)+'_weight'
                        self.logger.experiment.add_histogram(weight_name, w)

        return {
            'avg_val_loss': avg_loss,
            'val_acc': avg_acc,
            'progress_bar': tqdm_dict,
            'log': log_dict
            }

    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
        }

        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}
    
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        """
        Do something instead of the standard optimizer behavior
        :param epoch_nb:
        :param batch_nb:
        :param optimizer:
        :param optimizer_i:
        :return:
        """
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(second_order_closure)
        else:
            optimizer.step()

        self.on_before_zero_grad(optimizer)
        # clear gradients
        optimizer.zero_grad()

    @pl.data_loader
    def train_dataloader(self):
        if self.dataset['name'] == 'MNIST':
        # MNIST dataset
            train_dataset = torchvision.datasets.MNIST(root=self.dataset['dir'],
                                                    train=True,
                                                    transform=self.dataset['transform'],
                                                    download=True)
        elif self.dataset['name'] == 'ORL':
            train_dataset = ORLdataset(train=True,
                                        root_dir=self.dataset['dir'],
                                        transform=self.dataset['transform'],
                                        val_split=self.dataset['val_split'])
        elif self.dataset['name'] == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(root=self.dataset['dir'],
                                                    train=True,
                                                    transform=self.dataset['transform'],
                                                    download=True)
        elif self.dataset['name'] == 'NUMPY':
            train_dataset = NumpyDataset(root_dir=self.dataset['dir'], train=True)

        # Data loader
        return torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=self.dataset['batch_size'],
                                                shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        if self.dataset['name'] == 'MNIST':
            # MNIST dataset
            val_dataset = torchvision.datasets.MNIST(root=self.dataset['dir'],
                                                    train=False,
                                                    transform=self.dataset['transform'])
        elif self.dataset['name'] == 'ORL':
            val_dataset = ORLdataset(train=False,
                                        root_dir=self.dataset['dir'],
                                        transform=self.dataset['transform'],
                                        val_split=self.dataset['val_split'])
        elif self.dataset['name'] == 'CIFAR10':
            val_dataset = torchvision.datasets.CIFAR10(root=self.dataset['dir'],
                                                    train=False,
                                                    transform=self.dataset['transform'])

        elif self.dataset['name'] == 'NUMPY':
            val_dataset = NumpyDataset(root_dir=self.dataset['dir'], train=False)

        return torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=self.dataset['batch_size'],
                                                shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        if self.dataset['name'] == 'MNIST':
            # MNIST dataset
            test_dataset = torchvision.datasets.MNIST(root=self.dataset['dir'],
                                                    train=False,
                                                    transform=self.dataset['transform'])
        elif self.dataset['name'] == 'ORL':
            test_dataset = ORLdataset(train=False,
                                        root_dir=self.dataset['dir'],
                                        transform=self.dataset['transform'],
                                        val_split=self.dataset['val_split'])
        elif self.dataset['name'] == 'CIFAR10':
            test_dataset = torchvision.datasets.CIFAR10(root=self.dataset['dir'],
                                                    train=False,
                                                    transform=self.dataset['transform'])
        elif self.dataset['name'] == 'NUMPY':
            test_dataset = NumpyDataset(root_dir=self.dataset['dir'], train=False)

        return torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.dataset['batch_size'],
                                                shuffle=False)
    

class ModeNN(BaseModel):
    def __init__(self, input_size, order, num_classes, learning_rate=0.001, weight_decay=0.001, loss=nn.CrossEntropyLoss(), dropout=0,
                     norm=None, log_weight=True, dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':None, 'batch_size':100, 'transform':None}):
        super(ModeNN, self).__init__()
        if len(input_size) > 1:
            self.input_size = torch.tensor(input_size).prod().item()
        else:
            self.input_size = input_size[0]

        self.order=order
        self.dropout = dropout
        self.norm = norm
        self.log_weight=log_weight
        print('{} order Descartes Extension'.format(self.order))
        DE_dim = compute_mode_dim([self.input_size for _ in range(self.order-1)]) + self.input_size
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de_layer = Mode(order_dim=[self.input_size for _ in range(self.order-1)])

        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        if self.norm:
            self.norm_layer = nn.BatchNorm1d(DE_dim)

        self.tanh = nn.Tanh()
        self.fc = nn.Linear(DE_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset

    def forward(self, x):
        origin = torch.flatten(x, 1)
        out = self.de_layer(origin)
        out = torch.cat([origin, out], dim=-1)

        if self.norm:
            out = self.norm_layer(out)
        if self.dropout:
            out = self.dropout_layer(out)
    
        out = self.fc(out)
        # out = self.softmax(out)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return [opt]#, [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 160], gamma=0.1)]

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss.item(), 'val_acc': '{0:.5f}'.format(avg_acc.item())}
        log_dict = ({'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()})
        weight_dict = {}
       
        #log weight to tensorboard
        if self.log_weight:
            mode_para = self.fc.weight
            poly_item = find_polyitem(dim=self.input_size, order=self.order) 
            for i in range(len(mode_para)):
                for j in range(mode_para.shape[-1]):
                    w = mode_para[i][j].clone().detach()
                    weight_dict.update({'node{}_'.format(i)+poly_item[j]:w.item()})
            self.logger.experiment.add_scalars('mode_layer_weight', weight_dict, self.current_epoch)

        return {
            'avg_val_loss': avg_loss,
            'val_acc': avg_acc,
            'progress_bar': tqdm_dict,
            'log': log_dict
            }


    
class MyConv2D(BaseModel):
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
    def __init__(self, input_size, in_channel, out_channel, layer_num, dense_node, kernel_size, num_classes,
                     stride=1, padding=0, pooling='Max', pool_shape=(2,2), norm=False, dropout=None, learning_rate=0.001,
                     weight_decay=0.001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}, output_debug=False):
        super(MyConv2D, self).__init__()
        self.layer_num = layer_num
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
        self.output_debug = output_debug
        self.initconv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.pooling == 'Max':
            self.maxpool = nn.MaxPool2d(kernel_size=pool_shape)
        if self.pooling == 'Avg':
            self.avgpool = nn.AvgPool2d(kernel_size=pool_shape)
        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channel)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear((input_size[0]//(pool_shape[0]**layer_num))*(input_size[1]//(pool_shape[1]**layer_num))*out_channel, dense_node)
        self.fc2 = nn.Linear(dense_node, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # self.data_statics('input data', x, verbose=self.output_debug)
        out = self.initconv(x)
        # self.data_statics('output of conv1', out, verbose=self.output_debug)
        if self.norm :
            out = self.norm_layer(out)
        out = self.relu(out)
        if self.pooling:
            if self.pooling == 'Max':
                out = self.maxpool(out)
            elif self.pooling == 'Avg':
                out = self.avgpool(out)

        for _ in range(self.layer_num - 1):
            out = self.conv(out)
            # self.data_statics('output of conv'+str(_), out, verbose=self.output_debug)
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
        if self.dropout:
            out = self.dropout_layer(out)
        out = self.fc1(out)
        # self.data_statics('output of fc1', out, verbose=self.output_debug)
        out = self.relu(out)
        out = self.fc2(out)
        # self.data_statics('output of fc2', out, verbose=self.output_debug)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


class MNISTConv2D(BaseModel):
    def __init__(self, input_size, in_channel, num_classes, stride=(1,1), padding=(0,0), pooling='Max', pool_shape=(2,2),
                loss=nn.CrossEntropyLoss(), dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(MNISTConv2D, self).__init__()
        self.dataset = dataset
        self.loss = loss
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(compute_cnn_out(input_size=compute_cnn_out(input_size=input_size,kernel_size=(5,5),padding=padding,stride=stride,pooling=pool_shape),
                                            kernel_size=(5,5),padding=padding,stride=stride,pooling=pool_shape)[0]*
                            compute_cnn_out(input_size=compute_cnn_out(input_size=input_size,kernel_size=(5,5),padding=padding,stride=stride,pooling=pool_shape),
                                            kernel_size=(5,5),padding=padding,stride=stride,pooling=pool_shape)[1]*
                            64, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
      
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters())]


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
    def __init__(self, num_classes, loss=nn.CrossEntropyLoss(), dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(resnet18, self).__init__()
        self.dataset = dataset
        self.loss = loss
        self.resnet18 = resnet.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet18(x)

    def configure_optimizers(self):
        return [torch.optim.SGD(self.parameters(),lr=0.1, weight_decay=0.0005, momentum=0.9)]


class CIFARConv2D(BaseModel):

    def __init__(self, input_size, in_channel, layer_num, dense_node, kernel_size, num_classes,
                     stride=1, padding=0, pooling='Max', pool_shape=(2,2), norm=False, dropout=None, learning_rate=0.001,
                     weight_decay=0.0001, loss=nn.CrossEntropyLoss(),
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(CIFARConv2D, self).__init__()
        self.dataset = dataset
        self.loss = loss
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(2048,dense_node)
        self.fc2 = nn.Linear(dense_node, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
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
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]


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

