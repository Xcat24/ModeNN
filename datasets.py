import torch
import torchvision

root = '/disk/Dataset'
# cifar_train = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=True, download=True)
# cifar_test = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=False, download=True)

imagenet_train = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='train', download=True)
imagenet_val = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='val', download=True)