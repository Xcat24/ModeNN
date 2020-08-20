# from layer import RandomDE, output
import torch
import torchvision
import os
import logging as log
import pandas as pd
import numpy as np
from PIL import Image
import cv2
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

class SVD_transform(object):
    def __call__(self, sample):
        u,s,v = torch.svd(sample)
        return s

class DE_transform(object):
    def __init__(self, order=2):
        self.order = order

    def __call__(self, sample):
        sample = sample.view((sample.shape[0], -1))
        de = torch.cat([torch.stack([torch.prod(torch.combinations(a, i+1, with_replacement=True), dim=1) for a in sample]) for i in range(self.order)], dim=-1)
        return de.squeeze()

class RandomSampleDE_transform(object):
    def __init__(self, input_size_x, input_size_y, group_num, order):
        self.order = order
        self.group_num = group_num
        self.x = torch.normal(mean=input_size_x/2, std=4, size=(group_num, 3)).long().clamp(0,input_size_x-1)
        self.y = torch.normal(mean=input_size_y/2, std=4, size=(group_num, 3)).long().clamp(0,input_size_y-1)

    def __call__(self, sample):
        assert len(sample.shape) == 3
        assert isinstance(sample, torch.Tensor)
        origin = torch.flatten(sample, 1)
        result = []
        for i in range(self.group_num):
            tmp = torch.index_select(sample, dim=1, index=self.x[i])
            tmp = torch.index_select(tmp, dim=2, index=self.y[i])
            result.append(tmp)
        random_sample = torch.stack(result, dim=1).view(-1, 9)
        de = torch.cat([torch.stack([torch.prod(torch.combinations(a, i+2, with_replacement=True), dim=1) for a in random_sample]) for i in range(self.order - 1)], dim=-1)
        de = de.squeeze().view(sample.shape[0],-1)
        return torch.cat([origin, de], dim=-1).squeeze()

class DCT_transform(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        assert len(sample.shape) == 3
        tmp = sample.numpy()
        tmp = [torch.tensor(cv2.dct(x))[:self.dim, :self.dim] for x in tmp]
        return torch.stack(tmp, dim=0)


class RandomPick_transform(object):
    def __init__(self, input_dim=784, output_dim=64):
        self.mask = torch.randint(input_dim, (output_dim,))

    def __call__(self, sample):
        origin = torch.flatten(sample, 1)
        selected = torch.index_select(origin,dim=1,index=self.mask)
        return (origin, selected)

def gray_cifar_train_dataloader(dataset, data_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                DCT_transform(10),
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

def gray_cifar_val_dataloader(dataset, data_dir, batch_size, num_workers):
    val_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                DCT_transform(10),
                transforms.Normalize(120.7 / 255.0, 63.9 / 255.0)])
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=False,
                                                transform=val_transform)
    return torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)


def train_dataloader(dataset, data_dir, batch_size, num_workers, random_sample=False,random_in_dim=784, random_out_dim=316,
                    dct=False, dct_dim=10, svd=False, de=False, randomsample_de=False, group_num=64, order=2):
    log.info('Training data loader called.')
    if dataset == 'MNIST':
        transformers = [transforms.ToTensor()]
        if dct:
            transformers.append(DCT_transform(dct_dim))
        if svd:
            transformers.append(SVD_transform())
        if random_sample:
            transformers.append(RandomPick_transform(random_in_dim, random_out_dim))
        if de:
            transformers.append(DE_transform(order=order))
        if randomsample_de:
            transformers.append(RandomSampleDE_transform(28,28, group_num, order))
        
        trans = transforms.Compose(transformers)
        train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=True,
                                            transform=trans,
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


def val_dataloader(dataset, data_dir, batch_size, num_workers, random_sample=False,random_in_dim=784, random_out_dim=316,
                    dct=False, dct_dim=10, svd=False, de=False, randomsample_de=False, group_num=64, order=2):
    log.info('Valuating data loader called.')
    if dataset == 'MNIST':
        # MNIST dataset
        transformers = [transforms.ToTensor()]
        if dct:
            transformers.append(DCT_transform(dct_dim))
        if svd:
            transformers.append(SVD_transform())
        if random_sample:
            transformers.append(RandomPick_transform(random_in_dim, random_out_dim))
        if de:
            transformers.append(DE_transform(order=order))
        if randomsample_de:
            transformers.append(RandomSampleDE_transform(28,28, group_num, order))
        
        trans = transforms.Compose(transformers)
        val_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=False,
                                            transform=trans)

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


def test_dataloader(dataset, data_dir, batch_size, num_workers, random_sample=False,random_in_dim=784, random_out_dim=316,
                    dct=False, dct_dim=10, svd=False, de=False, randomsample_de=False, group_num=64, order=2):
    if dataset == 'MNIST':
        # MNIST dataset
        transformers = [transforms.ToTensor()]
        if dct:
            transformers.append(DCT_transform(dct_dim))
        if svd:
            transformers.append(SVD_transform())
        if random_sample:
            transformers.append(RandomPick_transform(random_in_dim, random_out_dim))
        if de:
            transformers.append(DE_transform(order=order))
        if randomsample_de:
            transformers.append(RandomSampleDE_transform(28,28, group_num, order))
        
        trans = transforms.Compose(transformers)
        test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=False,
                                            transform=trans)

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
    x = torchvision.datasets.MNIST(root='/home/xucong/Data/MNIST', train=True, transform=transforms.Compose([transforms.ToTensor(), DCT_transform(10)]))
    # x = torchvision.datasets.MNIST(root='/home/xucong/Data/MNIST', train=True, transform=transforms.Compose([transforms.ToTensor(), RandomSampleDE_transform(28,28,4,5)]))

    data = DataLoader(x, batch_size=2, shuffle=False)
    print(data.__len__())
    example = x.__getitem__(1)

    for batch in data:
        x, y = batch
        print(x)
        print(y)


    plt.figure()
    plt.imshow(example[0][0], cmap='gray')
    plt.savefig('mnist-dct-%s.jpg'%(example[1]))

    u,s,v = torch.svd(example[0][0])
    print(s.shape)
    print(s)
    new = s*s*s
    new = torch.diag(new)
    decode = torch.mm(torch.mm(u,new), v.t())
    #plot picture
    plt.figure()
    plt.imshow(decode, cmap='gray')
    plt.savefig('3-order-after.jpg')

    for i_batch, sample_batched in enumerate(data):
        print(i_batch, sample_batched['image'].size(), sample_batched['labels'].size())

    print('success')