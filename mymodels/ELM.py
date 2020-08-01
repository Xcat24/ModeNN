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
from layer import DescartesExtension, Mode
from myutils.utils import compute_mode_dim
from MyOptimizer import pseudoInverse

class ModeELM(LightningModule):
    def __init__(self, hparams, loss):
        super(ModeELM,self).__init__()
        self.hparams = hparams
        self.loss = loss
        if len(self.hparams.input_size) > 1:
            self.input_size = torch.tensor(self.hparams.input_size).prod().item()
        else:
            self.input_size = self.hparams.input_size[0]
        if self.hparams.pooling:
            self.input_size //= (self.hparams.pooling*self.hparams.pooling)
        self.DE_dim = compute_mode_dim([self.input_size for _ in range(self.hparams.order-1)]) + self.input_size
        self.de_layer = Mode(order_dim=[self.input_size for _ in range(self.hparams.order-1)])
        self.fc = nn.Linear(self.DE_dim, self.hparams.num_class, bias=False)
        self.opt = pseudoInverse(self.fc.weight, de_dim=self.DE_dim, batch_size=self.hparams.batch_size, num_class=self.hparams.num_class, C=0.001, L=0 )
        self.Flag = True
        self.batch_size = self.hparams.batch_size
        self.metric = {
            'accuracy': Accuracy(),
            'confusionmatrix': ConfusionMatrix(),
            'recall': Recall(),
            'precision': Precision(),
            'roc': ROC(),
            'averageprecision': AveragePrecision()
        }

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # x = self.de_layer(x)
        return out 

    def configure_optimizers(self):
        return None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        pass
        # if epoch==0 and batch_idx==0:
        #     optimizer.train(batch[0], batch[1])
        # else:
        #     optimizer.train_sequential(batch[0], batch[1])
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.view(x.size(0), -1)
        if self.Flag:
            self.opt.train(x,y)
            self.Flag = False
        else:
            self.opt.train_sequential(x,y)

        out = self.forward(x)
        loss = self.loss(out, y)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = self.metric['accuracy'](labels_hat, y)
        return {'val_loss': loss, 'val_acc': torch.tensor(val_acc)}

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
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
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
        parser.add_argument('--num-epochs', default=100, type=int, metavar='N',
                                help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=None,
                                help='seed for initializing training. ')
        parser.add_argument('--order', default=2, type=int,
                                help='order of Mode')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='dims of input data, return list')
        parser.add_argument('--pooling',default=0, type=int,
                                help='whether to decrease dimentions first')
        parser.add_argument('--num-class', default=None, type=int,
                                help='number of the total classes')
        return parser
