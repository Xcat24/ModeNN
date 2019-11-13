import math
import time
import os
import configparser
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import MyModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from myutils.utils import pick_edge, Pretrain_Select


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#================================== Read Setting ======================================
cf = configparser.ConfigParser()
# cf.read('config/XOR.conf')
# cf.read('config/iris.conf')
# cf.read('config/circle_square.conf')
cf.read('config/TC.conf')
# cf.read('config/mnist.conf')
# cf.read('config/mnist_bestcnn.conf')
# cf.read('./config/orl.conf')
# cf.read('./config/cifar10.conf')
# cf.read('./config/mnist_pretrain_5modenn.conf')
#Dataset Select
dataset_name = cf.get('dataset', 'dataset')
data_dir = cf.get('dataset', 'data_dir')

#model
model_name = cf.get('model', 'model_name')
saved_path = cf.get('model', 'saved_path')
    

#parameter setting
input_size = tuple([cf.getint('input_size', option) for option in cf['input_size']])
num_classes = cf.getint('para', 'num_classes')
num_epochs = cf.getint('para', 'num_epochs')
batch_size = cf.getint('input_size', 'batch_size')
learning_rate = cf.getfloat('para', 'learning_rate')
weight_decay = cf.getfloat('para', 'weight_decay')
val_split = cf.getfloat('para', 'val_split')
norm = cf.getboolean('para', 'norm')
dropout = cf.getfloat('para','dropout')
order = cf.getint('para', 'order')

try:
    resize=(cf.getint('input_size', 'resize_h'), cf.getint('input_size', 'resize_w'))
    in_channel = cf.getint('input_size', 'channel')
    out_channel = cf.getint('para', 'out_channel')
    layer_num = cf.getint('para', 'layer_num')
    kernel_size = (cf.getint('para', 'kernel_size'), cf.getint('para', 'kernel_size'))
    dense_node = cf.getint('para', 'dense_node')
    pretrain_model = cf.get('model', 'pretrain_model_path')
except:
    print('Does not contain CNN or pretrained model!')


#others
output_per = cf.getint('other', 'output_per')
log_dir = cf.get('other', 'log_dir')
tb_dir = cf.get('other', 'tb_dir')
patience = cf.getint('other', 'patience')
log_gpu = cf.getboolean('other', 'log_gpu')
try:
    gpus = cf.getint('other', 'gpus')
except ValueError as e:
    gpus = None
#================================= Read Setting End ===================================


#Dataset setting
if dataset_name == 'MNIST':
    transform = transforms.ToTensor()
elif dataset_name == 'CIFAR10':
    transform = transforms.ToTensor()
    # transform = transforms.Compose([pick_edge(), transforms.ToTensor()])
elif dataset_name == 'ORL':
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
elif dataset_name == 'NUMPY':
    transform = None

dataset = {'name':dataset_name, 'dir':data_dir, 'val_split':val_split, 'batch_size':batch_size, 'transform':transform}

# model = MyModel.MyConv2D(input_size=input_size[2:], in_channel=in_channel, out_channel=out_channel, layer_num=layer_num,
#                          dense_node=dense_node, kernel_size=kernel_size, num_classes=num_classes, padding=1, norm=norm,
#                          dropout=dropout, dataset=dataset, output_debug=True)

# model = MyModel.MyCNN_MODENN(input_size=input_size[2:], in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, num_classes=num_classes, pool_shape=(2,2),
#                             order=order, padding=1, norm=norm, dropout=dropout, dataset=dataset, learning_rate=learning_rate, weight_decay=weight_decay, output_debug=False)

# model = MyModel.CIFARConv2D(input_size=input_size[2:], in_channel=in_channel, layer_num=layer_num, pooling='Max',
#                          dense_node=dense_node, kernel_size=kernel_size, num_classes=num_classes, padding=1, norm=norm,
#                          dropout=dropout, dataset=dataset)

# model = MyModel.CIFARConv_MODENN(input_size=input_size[2:], in_channel=in_channel, layer_num=layer_num, pooling='Max',
#                          dense_node=dense_node, kernel_size=kernel_size, num_classes=num_classes, order=order, padding=1, norm=norm,
#                          dropout=dropout, dataset=dataset)

# model = MyModel.SLCNN(input_size=input_size[2:], in_channel=in_channel, stride=1, pooling='Max', pool_shape=(4,4), learning_rate=learning_rate, 
#                          weight_decay=weight_decay, num_classes=num_classes, padding=1, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.SLCNN_MODENN(input_size=input_size[2:], in_channel=in_channel, stride=1, pooling='Max', pool_shape=(4,4), learning_rate=learning_rate, 
                        #  weight_decay=weight_decay, num_classes=num_classes, order=order, padding=1, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.NoHiddenBase(input_size=input_size[1:], learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.Select_MODE(input_size=input_size[-1], model_path=pretrain_model, order_dim=[300, 55, 25, 15], learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes, norm=norm, dropout=dropout, dataset=dataset)


# model = MyModel.OneHiddenBase(input_size=input_size[1:], learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.Pretrain_5MODENN(num_classes=num_classes,bins_size=9, bins_num=35,
#                      dropout=None, learning_rate=learning_rate,weight_decay=weight_decay, loss=nn.CrossEntropyLoss(),
#                      dataset=dataset)

# model = MyModel.MNISTConv2D(input_size=input_size[2:], in_channel=in_channel, num_classes=num_classes, padding=(0,0), dataset=dataset)

# model = MyModel.resnext29(input_size=input_size[2:], in_channel=in_channel, num_classes=num_classes, dataset=dataset)

# model = MyModel.resnet18(num_classes=num_classes, dataset=dataset)

model = MyModel.ModeNN(input_size=input_size[1:], order=order, num_classes=num_classes, learning_rate=learning_rate, weight_decay=weight_decay, dataset=dataset, log_weight=10)
summary(model, input_size=input_size[1:], device='cpu')

early_stop_callback = EarlyStopping(
    monitor='val_acc',
    min_delta=0.00,
    patience=patience,
    verbose=True,
    mode='auto'
)

checkpoint_callback = ModelCheckpoint(
    filepath=saved_path,
    save_best_only=True,
    verbose=True,
    monitor='val_acc',
    mode='max',
    prefix=''
)

# tb_logger = SummaryWriter(log_dir=log_dir)
tb_logger = TestTubeLogger(
    save_dir=log_dir,
    name=tb_dir,
    debug=False,
    create_git_tag=False)
    
trainer = Trainer(
    min_nb_epochs=1,
    max_nb_epochs=num_epochs,
    log_gpu_memory=log_gpu,
    gpus=gpus,
    fast_dev_run=False, #activate callbacks, everything but only with 1 training and 1 validation batch
    gradient_clip_val=0,  #this will clip the gradient norm computed over all model parameters together
    track_grad_norm=-1,  #Looking at grad norms
    print_nan_grads=True,
    checkpoint_callback=checkpoint_callback,
    logger=tb_logger,
    row_log_interval=80,
    log_save_interval=80,
    early_stop_callback=early_stop_callback)


trainer.fit(model)
trainer.test()


    



