import math
import time
import os
import configparser
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import MyModel
from MyPreprocess import ORLdataset
from MyUtils import EarlyStopping


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================== Read Setting ======================================
cf = configparser.ConfigParser()
# cf.read('config/mnist.conf')
cf.read('./config/orl.conf')
#Dataset Select
dataset = cf.get('dataset', 'dataset')
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
patience = cf.getint('other', 'patience')
#================================= Read Setting End ===================================

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
        print('Val: Avg loss:{:.4f}, Acurracy: {}/{} ({} %)\n'.format(val_loss, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))
    return val_loss


#Dataset setting
if dataset == 'MNIST':
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=False,
                                            transform=transforms.ToTensor())
elif dataset == 'ORL':
    train_dataset = ORLdataset(train=True,
                                root_dir=data_dir,
                                transform=transforms.Compose([transforms.Resize(resize),
                                                            transforms.ToTensor()]),
                                val_split=val_split)
    test_dataset = ORLdataset(train=False,
                                root_dir=data_dir,
                                transform=transforms.Compose([transforms.Resize(resize),
                                                            transforms.ToTensor()]),
                                val_split=val_split)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



#Select model to use
if model_name == 'ModeNN':
    model = MyModel.ModeNN(input_size[-1]*input_size[-2], order, num_classes).to(device)
if model_name == 'MyCNN':
    model = MyModel.MyConv2D(input_size=input_size[2:], in_channel=1, out_channel=32, layer_num=2, dense_node=dense_node, kernel_size=3, num_classes=num_classes,
                             padding=1, norm=True, dropout=0.25).to(device)
summary(model, input_size=input_size[1:])

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
    val_loss = test_model(model, device,test_loader)

    early_stopping(train_loss, model)
    if early_stopping.early_stop:
        print('++++++++++++++++++++++++++++++')
        print('Early stopping!')
        print('++++++++++++++++++++++++++++++\n')
        break
    



