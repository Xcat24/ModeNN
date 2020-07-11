import torch
import torchvision
import os
import logging as log
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class NumpyDataset(Dataset):
    def __init__(self, root_dir, train=False):
        if train:
            self.data = torch.tensor(np.load(os.path.join(root_dir,'train_data.npy')), dtype=torch.float32)
            self.labels = torch.tensor(np.load(os.path.join(root_dir, 'train_label.npy')))
        else:
            self.data = torch.tensor(np.load(os.path.join(root_dir,'val_data.npy')), dtype=torch.float32)
            self.labels = torch.tensor(np.load(os.path.join(root_dir, 'val_label.npy')))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        return x, label

class ORLdataset(Dataset):
    def __init__(self, train=True, root_dir='/disk/Dataset/ORL/', transform=None, val_split=0.1):
        self.dir = ['s'+str(x + 1) for x in range(40)]
        if val_split <= 1 and val_split >= 0:
            self.split_indice = int(np.floor((1 - val_split)*10))
        else:
            raise ValueError('the val_split should be in range (0, 1)')
        if train:
            self.data = torch.randperm(self.split_indice).tolist()
            self.num_perclass = self.split_indice
        else:
            self.data = (torch.randperm((10 - self.split_indice)) + self.split_indice).tolist()
            self.num_perclass = 10 - self.split_indice

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)*40

    def __getitem__(self, idx):
        # if idx not in range(1, 401):
        #     raise ValueError
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_dir = idx // self.num_perclass
        img_name = str(self.data[idx % self.num_perclass] + 1) + '.pgm'
        img = os.path.join(self.root_dir, self.dir[img_dir], img_name)
        image = Image.open(img)

        if self.transform:
            image = self.transform(image)

        labels = img_dir
        sample = image, labels

        return sample

def gray_cifar_train_dataloader(dataset, data_dir, batch_size, num_workers=4):
    train_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(120.7 / 255.0, 63.9 / 255.0)
            ])
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=True,
                                                transform=train_transform, #self.dataset['transform'],
                                                download=True)
    return torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)

def gray_cifar_val_dataloader(dataset, data_dir, batch_size, num_workers=4):
    val_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(120.7 / 255.0, 63.9 / 255.0)])
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=False,
                                                transform=val_transform)
    return torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)


def train_dataloader(dataset, data_dir, batch_size, num_workers=4, augmentation=True):
    log.info('Training data loader called.')
    if dataset == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
    elif dataset == 'ORL':
        train_dataset = ORLdataset(train=True,
                                    root_dir=data_dir,
                                    transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                                    val_split=dataset['val_split'])
    elif dataset == 'CIFAR10':
        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
            ])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=True,
                                                transform=train_transform, #self.dataset['transform'],
                                                download=True)
    elif dataset == 'NUMPY':
        train_dataset = NumpyDataset(root_dir=data_dir, train=True)

    # Data loader
    return torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)


def val_dataloader(dataset, data_dir, batch_size, num_workers=4):
    log.info('Valuating data loader called.')
    if dataset == 'MNIST':
        # MNIST dataset
        val_dataset = torchvision.datasets.MNIST(root=data_dir,
                                                train=False,
                                                transform=transforms.ToTensor())
    elif dataset == 'ORL':
        val_dataset = ORLdataset(train=False,
                                    root_dir=data_dir,
                                    transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                                    val_split=dataset['val_split'])
    elif dataset == 'CIFAR10':
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=False,
                                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)]))

    elif dataset == 'NUMPY':
        val_dataset = NumpyDataset(root_dir=data_dir, train=False)

    return torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)


def test_dataloader(dataset, data_dir, batch_size, num_workers=4):
    if dataset == 'MNIST':
        # MNIST dataset
        test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                                train=False,
                                                transform=transforms.ToTensor())
    elif dataset == 'ORL':
        test_dataset = ORLdataset(train=False,
                                    root_dir=data_dir,
                                    transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                                    val_split=dataset['val_split'])
    elif dataset == 'CIFAR10':
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=False,
                                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)]))
    elif dataset == 'NUMPY':
        test_dataset = NumpyDataset(root_dir=data_dir, train=False)

    return torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)

if __name__ == '__main__':
    #test
    x = ORLdataset(train=True, transform=transforms.Compose([transforms.Resize(50), transforms.ToTensor()]), val_split=0.2)

    # dataset_size = len(x)
    # validation_split = 0.2
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # train_indices, val_indices = indices[split:], indices[:split]

    # # Creating PT data samplers and loaders:
    # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    # valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    data = DataLoader(x, batch_size=2, shuffle=False)
    print(data.__len__())
    example = x.__getitem__(0)
    #plot picture
    plt.figure()
    plt.imshow(example['image'].reshape((60,50)), cmap='gray')
    plt.savefig('/disk/test.jpg')

    for i_batch, sample_batched in enumerate(data):
        print(i_batch, sample_batched['image'].size(), sample_batched['labels'].size())

    print('success')