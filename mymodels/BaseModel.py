import numpy as np
import logging as log
import torch
import argparse
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, AveragePrecision, ConfusionMatrix, Recall, Precision, ROC
from torch.utils.data import Dataset, DataLoader
from myutils.datasets import ORLdataset, NumpyDataset

class BaseModel(LightningModule):
    def __init__(self, hparams, loss):
        super(BaseModel, self).__init__()
        self.hparams = hparams
        self.loss = loss
        self.metric = {
            'accuracy': Accuracy(),
            'confusionmatrix': ConfusionMatrix(),
            'recall': Recall(),
            'precision': Precision(),
            'roc': ROC(),
            'averageprecision': AveragePrecision()
        }

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = self.metric['accuracy'](labels_hat, y)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        results = {'log': logs}
        return results

    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = self.metric['accuracy'](labels_hat, y)

        # return whatever you need for the collation function validation_end
        return {
            'test_loss': loss,
            'test_acc': test_acc # everything must be a tensor
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test_loss': loss, 'test_acc': avg_acc}
        results = {'log': logs}
        return results

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
     
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
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