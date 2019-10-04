import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import pytorch_lightning as pl
from layer import DescartesExtension, MaskDE, LocalDE
from torch.utils.data import Dataset, DataLoader
from MyPreprocess import ORLdataset

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

class MyConv2D(pl.LightningModule):
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
                     dataset={'name':'MNIST', 'dir':'/disk/Dataset/', 'val_split':0.1, 'batch_size':100, 'transform':None}):
        super(MyConv2D, self).__init__()
        self.layer_num = layer_num
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss = loss
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
        out = self.initconv(x)
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

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        return {
            'loss': loss
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
            'val_acc': torch.tensor(val_acc), # everything must be a tensor
        }

        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean().item()
       
        #logger
        if self.logger:
            layer_names = list(self._modules)
            for i in range(len(layer_names)):
                mod_para = list(self._modules[layer_names[i]].parameters())
                if mod_para:
                    for j in range(len(mod_para)):
                        w = torch.tensor(mod_para[j])
                        self.logger.experiment.add_histogram(layer_names[i]+'_'+str(w.shape)+'_weight', w)


        return {'avg_val_loss': avg_loss, 'val_acc': avg_acc}

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
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().item()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean().item()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)]

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
        return torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.dataset['batch_size'],
                                                shuffle=False)