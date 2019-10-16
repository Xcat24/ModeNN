import torch
import torchvision
import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class NumpyDataset(Dataset):
    def __init__(self, data_root_dir, train=False):
        if train:
            self.data = torch.tensor(np.load(os.path.join(data_root_dir,'train_data.npy')))
            self.labels = torch.tensor(np.load(os.path.join(data_root_dir, 'train_label.npy')))
        else:
            self.data = torch.tensor(np.load(os.path.join(data_root_dir,'val_data.npy')))
            self.labels = torch.tensor(np.load(os.path.join(data_root_dir, 'val_label.npy')))
    
    def __len__():
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