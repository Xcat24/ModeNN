import torch
import argparse
import torchvision
import torchvision.transforms as transforms
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from myutils.datasets import ORLdataset, NumpyDataset

class BaseModel(pl.LightningModule):
    def __init__(self, hparams, loss):
        super(BaseModel, self).__init__()
        self.hparams = hparams
        self.loss = loss

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        return {
            'loss': loss,
            'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
            'log': {'training_loss': loss.item()}
        }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'val_loss': loss,
            'val_acc': torch.tensor(val_acc) # everything must be a tensor
        }

        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss.item(), 'val_acc': '{0:.5f}'.format(avg_acc.item())}
        log_dict = {'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()}
       
        #logger
        if self.logger:
            layer_names = list(self._modules)
            for i in range(len(layer_names)):
                mod_para = list(self._modules[layer_names[i]].parameters())
                if mod_para:
                    for j in range(len(mod_para)):
                        w = mod_para[j].clone().detach()
                        weight_name=layer_names[i]+'_'+str(w.shape)+'_weight'
                        self.logger.experiment.add_histogram(weight_name, w)

        return {
            'avg_val_loss': avg_loss,
            'val_acc': avg_acc,
            'progress_bar': tqdm_dict,
            'log': log_dict
            }

    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
        }

        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}
    
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        """
        Do something instead of the standard optimizer behavior
        :param epoch_nb:
        :param batch_nb:
        :param optimizer:
        :param optimizer_i:
        :return:
        """
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(second_order_closure)
        else:
            optimizer.step()

        self.on_before_zero_grad(optimizer)
        # clear gradients
        optimizer.zero_grad()

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.dataset == 'MNIST':
            train_dataset = torchvision.datasets.MNIST(root=self.hparams.data_dir,
                                                    train=True,
                                                    transform=transforms.ToTensor(),
                                                    download=True)
        elif self.hparams.dataset == 'ORL':
            train_dataset = ORLdataset(train=True,
                                        root_dir=self.hparams.data_dir,
                                        transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                                        val_split=self.dataset['val_split'])
        elif self.hparams.dataset == 'CIFAR10':
            if hparams.augmentation:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    val_transform
                ])
            else:
                train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)])
            train_dataset = torchvision.datasets.CIFAR10(root=self.hparams.data_dir,
                                                    train=True,
                                                    transform=train_transform, #self.dataset['transform'],
                                                    download=True)
        elif self.hparams.dataset == 'NUMPY':
            train_dataset = NumpyDataset(root_dir=self.hparams.data_dir, train=True)

        # Data loader
        return torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=self.hparams.batch_size,
                                                shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset == 'MNIST':
            # MNIST dataset
            val_dataset = torchvision.datasets.MNIST(root=self.hparams.data_dir,
                                                    train=False,
                                                    transform=transforms.ToTensor())
        elif self.hparams.dataset == 'ORL':
            val_dataset = ORLdataset(train=False,
                                        root_dir=self.hparams.data_dir,
                                        transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                                        val_split=self.dataset['val_split'])
        elif self.hparams.dataset == 'CIFAR10':
            val_dataset = torchvision.datasets.CIFAR10(root=self.hparams.data_dir,
                                                    train=False,
                                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)]))

        elif self.hparams.dataset == 'NUMPY':
            val_dataset = NumpyDataset(root_dir=self.hparams.data_dir, train=False)

        return torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=self.hparams.batch_size,
                                                shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        if self.hparams.dataset == 'MNIST':
            # MNIST dataset
            test_dataset = torchvision.datasets.MNIST(root=self.hparams.data_dir,
                                                    train=False,
                                                    transform=transforms.ToTensor())
        elif self.hparams.dataset == 'ORL':
            test_dataset = ORLdataset(train=False,
                                        root_dir=self.hparams.data_dir,
                                        transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                                        val_split=self.dataset['val_split'])
        elif self.hparams.dataset == 'CIFAR10':
            test_dataset = torchvision.datasets.CIFAR10(root=self.hparams.data_dir,
                                                    train=False,
                                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)]))
        elif self.hparams.dataset == 'NUMPY':
            test_dataset = NumpyDataset(root_dir=self.hparams.data_dir, train=False)

        return torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.hparams.batch_size,
                                                shuffle=False)
    
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parent_parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parent_parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
        parent_parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data')
        return parser