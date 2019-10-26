import torch
import torchvision
import torchvision.transforms as transforms
import layer
from myutils.datasets import ORLdataset, NumpyDataset


mnist_train_data = torchvision.datasets.MNIST(root='/disk/Dataset/', train=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_val_data = torchvision.datasets.MNIST(root='/disk/Dataset/', train=False, transform=transforms.Compose([transforms.ToTensor()]))

temp = torch.utils.data.DataLoader(dataset=mnist_train_data, batch_size=60000, shuffle=False)
for i , j in enumerate(temp):
    index, train_data = i, j

temp = torch.utils.data.DataLoader(dataset=mnist_val_data, batch_size=10000, shuffle=False)
for i , j in enumerate(temp):
    index, val_data = i, j

slconv = layer.SLConv(in_channel=1, stride=1, padding=1)
pool = torch.nn.MaxPool2d(4)

# x = mnist_train_data.__getitem__(0)[0].unsqueeze(1)
out = slconv(val_data[0])


root = '/disk/Dataset'
# cifar_train = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=True, download=True)
# cifar_test = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=False, download=True)

imagenet_train = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='train', download=True)
imagenet_val = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='val', download=True)