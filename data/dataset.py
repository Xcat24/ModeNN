# from layer import RandomDE, output
import torch
import torchvision
import os
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .feature_transform import DCT_transform, SVD_transform, RandomPick_transform, RandomSampleDE_transform, DE_transform, HOG_transform
import medmnist
from medmnist import INFO, Evaluator

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
                    dct=False, dct_dim=10, svd=False, de=False, randomsample_de=False, group_num=64, order=2, resize=(28,28), augmentation=False, hog=False):
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
        if hog:
            transformers.append(HOG_transform(4,4,7))
        
        trans = transforms.Compose(transformers)
        train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=True,
                                            transform=trans,
                                            download=True)
    
    elif dataset.split('.')[0] == 'MedMNIST':
        data_flag = dataset.split('.')[1]
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        transformers = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[.5], std=[.5])
                        ])
        train_dataset = DataClass(split='train', transform=transformers, download=True, root=data_dir)

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

    elif dataset == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), 
                                                        transform=transforms.Compose([
                                                                    transforms.RandomResizedCrop(224),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    normalize,
                                                                    ]))

    # Data loader
    return torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)


def val_dataloader(dataset, data_dir, batch_size, num_workers, random_sample=False,random_in_dim=784, random_out_dim=316,
                    dct=False, dct_dim=10, svd=False, de=False, randomsample_de=False, group_num=64, order=2, resize=(28,28), hog=False):
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
        if hog:
            transformers.append(HOG_transform(4,4,7))
        
        trans = transforms.Compose(transformers)
        val_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=False,
                                            transform=trans)

    elif dataset.split('.')[0] == 'MedMNIST':
        data_flag = dataset.split('.')[1]
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        transformers = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[.5], std=[.5])
                        ])
        val_dataset = DataClass(split='val', transform=transformers, download=True, root=data_dir)

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

    elif dataset == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), 
                                                        transform=transforms.Compose([
                                                                    transforms.Resize(256),
                                                                    transforms.RandomResizedCrop(224),
                                                                    transforms.ToTensor(),
                                                                    normalize,
                                                                    ]))

    return torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)


def test_dataloader(dataset, data_dir, batch_size, num_workers, random_sample=False,random_in_dim=784, random_out_dim=316,
                    dct=False, dct_dim=10, svd=False, de=False, randomsample_de=False, group_num=64, order=2, resize=(28,28), hog=False):
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
        if hog:
            transformers.append(HOG_transform(4,4,7))
        
        trans = transforms.Compose(transformers)
        test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                            train=False,
                                            transform=trans)

    elif dataset.split('.')[0] == 'MedMNIST':
        data_flag = dataset.split('.')[1]
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        transformers = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[.5], std=[.5])
                        ])
        test_dataset = DataClass(split='test', transform=transformers, download=True, root=data_dir)

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

