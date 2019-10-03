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
from MyPreprocess import ORLdataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TestTubeLogger


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================== Read Setting ======================================
cf = configparser.ConfigParser()
cf.read('config/mnist.conf')
# cf.read('./config/orl.conf')
#Dataset Select
dataset_name = cf.get('dataset', 'dataset')
data_dir = cf.get('dataset', 'data_dir')

#model
model_name = cf.get('model', 'model_name')
saved_name = cf.get('model', 'saved_name')

#parameter setting
resize=(cf.getint('input_size', 'resize_h'), cf.getint('input_size', 'resize_w'))
input_size = tuple([cf.getint('input_size', option) for option in cf['input_size']])
val_split = cf.getfloat('para', 'val_split')
order = cf.getint('para', 'order')
num_classes = cf.getint('para', 'num_classes')
num_epochs = cf.getint('para', 'num_epochs')
batch_size = cf.getint('input_size', 'batch_size')
dense_node = cf.getint('para', 'dense_node')
learning_rate = cf.getfloat('para', 'learning_rate')
weight_decay = cf.getfloat('para', 'weight_decay')

#others
output_per = cf.getint('other', 'output_per')
log_file_name = cf.get('other', 'log_file_name')
tb_dir = cf.get('other', 'tb_dir')
patience = cf.getint('other', 'patience')
#================================= Read Setting End ===================================


#Dataset setting
if dataset_name == 'MNIST':
    # MNIST dataset
    transform = transforms.ToTensor()

elif dataset_name == 'ORL':
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

dataset = {'name':dataset_name, 'dir':data_dir, 'val_split':val_split, 'batch_size':batch_size, 'transform':transform}

model = MyModel.MyConv2D(device=device, input_size=input_size[2:], in_channel=1, out_channel=32, layer_num=2, dense_node=dense_node, kernel_size=3, num_classes=num_classes,
                             padding=1, norm=True, dropout=0.25, dataset=dataset).to(device)

early_stop_callback = EarlyStopping(
    monitor='loss',
    min_delta=0.00,
    patience=30,
    verbose=True,
    mode='auto'
)

# exp = Experiment(
#     name='test_tube_exp',
#     debug=True,
#     save_dir=log_dir,
#     version=0,
#     autosave=False,
#     description='test demo'
# )

tt_logger = TestTubeLogger(
    save_dir=log_file_name,
    name="default",
    debug=True,
    create_git_tag=False
)
    
trainer = Trainer(
    min_nb_epochs=1,
    max_nb_epochs=1000,
    fast_dev_run=True, #activate callbacks, everything but only with 1 training and 1 validation batch
    gradient_clip_val=0,  #this will clip the gradient norm computed over all model parameters together
    track_grad_norm=1,  #Looking at grad norms
    print_nan_grads=True,
    logger=tt_logger,
    early_stop_callback=early_stop_callback)


trainer.fit(model)
trainer.test()


    



