import numpy as np
import logging as log
import torch
import argparse
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
from myutils.datasets import ORLdataset, NumpyDataset

class BaseModel(LightningModule):
    def __init__(self, hparams, loss):
        super(BaseModel, self).__init__()
        self.hparams = hparams
        self.loss = loss

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        
        return OrderedDict({
            'loss': loss,
            'progress_bar': {'training_loss': loss.item()}, # optional (MUST ALL BE TENSORS)
            'log': {'training_loss': loss.item()}
        })

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # return whatever you need for the collation function validation_end
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': torch.tensor(val_acc) # everything must be a tensor
        })

        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}

        return {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_acc': val_loss_mean
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
        tqdm_dict = {'avg_test_loss': avg_loss.item(), 'test_acc': avg_acc.item()}
        return {
            'progress_bar': tqdm_dict,
            'log': {'test_acc': avg_acc.item()}
        }
     
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