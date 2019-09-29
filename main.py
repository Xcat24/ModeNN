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
from myutils.callback import EarlyStopping
from pytorch_lightning import Trainer


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================== Read Setting ======================================
cf = configparser.ConfigParser()
# cf.read('config/mnist.conf')
cf.read('./config/orl.conf')
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
trainer = Trainer()
trainer.fit(model)




def train_model(model, device, train_loader, optimizer, epoch, total_epoch):
    model.train()
    train_loss = 0
    t0 = time.time()
    for i, sample_batch in enumerate(train_loader):
        # Move tensors to the configured device
        if dataset == 'MNIST':
            images = sample_batch[0].reshape(input_size).to(device)
            labels = sample_batch[1].to(device)
        else:
            images = sample_batch['image'].reshape(input_size).to(device)
            labels = sample_batch['labels'].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % output_per) == 0:
            print ('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.2f}seconds'.format(epoch+1, total_epoch, i, len(train_loader), loss.item(), time.time()-t0))
    return train_loss/len(train_loader)

def test_model(model, device, test_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for sample_batch in test_loader:
            if dataset == 'MNIST':
                images = sample_batch[0].reshape(input_size).to(device)
                labels = sample_batch[1].to(device)
            else:
                images = sample_batch['image'].reshape(input_size).to(device)
                labels = sample_batch['labels'].to(device)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            # predicted = outputs.argmax(dim=1, keepdim=True)
            # correct += predicted.eq(labels.view_as(predicted)).sum().item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        val_loss /= len(test_loader)
        acc = 100 * correct / len(test_loader.dataset)
        print('Val: Avg loss:{:.4f}, Acurracy: {}/{} ({} %)\n'.format(val_loss, correct, len(test_loader.dataset), acc))
    return val_loss, acc
# Add tensorboard summary
writer = SummaryWriter('/disk/Log/torch/'+ tb_dir)

# Select model to use
if model_name == 'ModeNN':
    model = MyModel.ModeNN(input_size[-1]*input_size[-2], order, num_classes).to(device)
if model_name == 'MyCNN':
    model = MyModel.MyConv2D(input_size=input_size[2:], in_channel=1, out_channel=32, layer_num=2, dense_node=dense_node, kernel_size=3, num_classes=num_classes,
                             padding=1, norm=True, dropout=0.25).to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.Adadelta(model.parameters())

# Train the model
total_step = len(train_loader)
#trach loss during the training

early_stopping = EarlyStopping(patience=patience, verbose=True)

for epoch in range(num_epochs):
    train_loss = train_model(model, device, train_loader, optimizer, epoch, num_epochs)
    val_loss, acc = test_model(model, device,test_loader)

    writer.add_scalar('train loss', train_loss, epoch+1)
    writer.add_scalar('val loss', val_loss, epoch+1)
    writer.add_scalar('val acc', acc, epoch+1)

    early_stopping(train_loss, model)
    if early_stopping.early_stop:
        print('++++++++++++++++++++++++++++++')
        print('Early stopping!')
        print('++++++++++++++++++++++++++++++\n')
        break
    



