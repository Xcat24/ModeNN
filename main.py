import math
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from MyModel import ModeNN
from MyPreprocess import ORLdataset


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================== Human Setting ======================================
#Dataset Select
dataset = 'ORL'
data_dir = '/disk/Dataset/ORL'
val_split = 0.1

#TODO Model Select
model_name = 'ModeNN'

# Hyper-parameters
resize=(60, 50)
input_size = 60*50
# input_size = 784
order = 2
num_classes = 40
num_epochs = 5
batch_size = 2
learning_rate = 0.001

#others
output_per = 5
saved_name = dataset + model_name + 'resize60-50_softmax' + 'val3'
#================================= Human Setting End ===================================

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


model = ModeNN(input_size, order, num_classes).to(device)
summary(model, input_size=(input_size,))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    t0 = time.time()
    for i, sample_batch in enumerate(train_loader):
        # Move tensors to the configured device
        if dataset == 'MNIST':
            images = sample_batch[0].reshape(-1, input_size).to(device)
            labels = sample_batch[1].to(device)
        else:
            images = sample_batch['image'].reshape(-1, input_size).to(device)
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
            images = sample_batch[0].reshape(-1, input_size).to(device)
            labels = sample_batch[1].to(device)
        else:
            images = sample_batch['image'].reshape(-1, input_size).to(device)
            labels = sample_batch['labels'].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), saved_name + '.ckpt')
