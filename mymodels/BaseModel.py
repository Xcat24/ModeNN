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
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall
from torch.utils.data import Dataset, DataLoader
from myutils.datasets import ORLdataset, NumpyDataset

class BaseModel(LightningModule):
    def __init__(self, hparams, loss):
        super(BaseModel, self).__init__()
        self.hparams = hparams
        self.loss = loss
        self.metric = {
            'accuracy': Accuracy(),
            'recall': Recall(),
            'precision': Precision(),
        }

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # if self.logger is not None:
        #     for i in range(len(self.logger.experiment)):
        #         self.logger[i].experiment.log({'train_loss': loss})

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = self.metric['accuracy'](labels_hat, y)
        # labels_hat = torch.argmax(out, dim=1, keepdim=True)
        # res  = labels_hat.eq(y.view_as(labels_hat)).sum().item()
        # val_acc = torch.tensor(res / len(x))

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        # if self.logger is not None:
        #     for i in range(len(self.logger.experiment)):
        #         self.logger[i].experiment.log({'val_loss': val_loss_mean, 'val_acc': val_acc_mean})
        self.log('val_loss', val_loss_mean, prog_bar=True, logger=True)
        self.log('val_acc', val_acc_mean, prog_bar=True, logger=True)


    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = self.metric['accuracy'](labels_hat, y)
        test_acc /= self.hparams.gpus

        # return whatever you need for the collation function validation_end
        return {
            'test_loss': loss,
            'test_acc': test_acc # everything must be a tensor
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        results = {'progress_bar': logs}
        return results

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
     
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=100, type=int, metavar='N',
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
        parent_parser.add_argument('--num-class', default=None, type=int,
                                help='number of the total classes')
        return parser