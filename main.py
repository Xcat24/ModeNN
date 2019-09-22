import math
import time
import os
import configparser
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import MyModel
from MyPreprocess import ORLdataset


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================== Read Setting ======================================
cf = configparser.ConfigParser()
# cf.read('config/mnist.conf')
cf.read('example.conf')
#Dataset Select
dataset = cf.get('dataset', 'dataset')
data_dir = cf.get('dataset', 'data_dir')

#model
model_name = cf.get('model', 'model_name')
saved_name = cf.get('model', 'saved_name')

#parameter setting
resize=(cf.getint('para', 'resize_h'), cf.getint('para', 'resize_w'))
input_size = tuple([cf.getint('input_size', option) for option in cf['input_size']])
val_split = cf.getfloat('para', 'val_split')
order = cf.getint('para', 'order')
num_classes = cf.getint('para', 'num_classes')
num_epochs = cf.getint('para', 'num_epochs')
batch_size = cf.getint('para', 'batch_size')
learning_rate = cf.getfloat('para', 'learning_rate')
weight_decay = cf.getfloat('para', 'weight_decay')

#others
output_per = cf.getint('other', 'output_per')
log_file_name = cf.get('other', 'log_file_name')
#================================= Read Setting End ===================================

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



# Fully connected neural network with one hidden layer

if model_name == 'ModeNN':
    model = MyModel.ModeNN(input_size[-1], order, num_classes).to(device)
if model_name == 'MyCNN':
    model = MyModel.MyConv2D(in_channel=1, out_channel=32, layer_num=2, kernel_size=3, num_classes=num_classes,
                             padding=1, norm=True, dropout=0.25).to(device)
summary(model, input_size=input_size[1:])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
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

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % output_per == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.2f}seconds'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), time.time()-t0))
            t0 = time.time()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for sample_batch in test_loader:
        if dataset == 'MNIST':
            images = sample_batch[0].reshape(input_size).to(device)
            labels = sample_batch[1].to(device)
        else:
            images = sample_batch['image'].reshape(input_size).to(device)
            labels = sample_batch['labels'].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), saved_name + '.ckpt')
