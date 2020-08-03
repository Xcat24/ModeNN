import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict
from .BaseModel import BaseModel
import mymodels.activation
from myutils.utils import compute_mode_dim



class AutoEncoder(LightningModule):
    def __init__(self, hparams, loss=None):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.loss = nn.MSELoss()
        if isinstance(self.hparams.input_size, (list, dict)):
            self.hparams.input_size = np.prod(self.hparams.input_size)
        self.encoder = nn.Sequential(
			nn.Linear(self.hparams.input_size, self.hparams.features[0]),
			nn.SELU(),
			nn.Linear(self.hparams.features[0], self.hparams.features[1]),
			nn.SELU(),
			nn.Linear(self.hparams.features[1], self.hparams.features[2]),
			nn.SELU())
        self.decoder = nn.Sequential(
			nn.Linear(self.hparams.features[2], self.hparams.features[1]),
			nn.SELU(),
			nn.Linear(self.hparams.features[1], self.hparams.features[0]),
			nn.SELU(),
			nn.Linear(self.hparams.features[0], self.hparams.input_size))
        self.main = nn.Sequential(
			self.encoder,
			self.decoder)

    def forward(self, x):
        size=x.size()
        x = torch.flatten(x, start_dim=1)
        output=self.main(x)
        return output.view(size)

    def configure_optimizers(self):
        if self.hparams.opt == 'SGD':
            opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=True)
            if self.hparams.lr_milestones:
                return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
            else:
                return [opt]
        elif self.hparams.opt == 'Adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            if self.hparams.lr_milestones:
                return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
            else:
                return [opt]
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, x)

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
        loss = self.loss(out, x)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # return whatever you need for the collation function validation_end
        output = OrderedDict({
            'val_loss': loss,
        })

        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0

        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)

        tqdm_dict = {'val_loss': val_loss_mean.item()}

        return {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': val_loss_mean
            }
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, x)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'data': x,
            'label': y
        }

        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='AutoEncoder', type=str, 
                            help='network architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-milestones', nargs='+', type=int,
                                help='learning rate milestones')
        parser.add_argument('--lr-gamma', default=0.1, type=float,
                            help='number learning rate multiplied when reach the lr-milestones')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--dropout', default=0, type=float,
                                help='the rate of the dropout')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--log-weight', default=0, type=int,
                                help='log weight figure every x epoch')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
        parser.add_argument('--features', nargs='+', type=int,
                                help='numbers of dense layers nodes, return as list')
        parser.add_argument('--activation', default='Tanh', type=str,
                                help='activation function to use')

        return parser


class AE_MP(LightningModule):
    #TODO,还没完成
    def __init__(self, hparams, loss=None):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.loss = nn.MSELoss()
        if isinstance(self.hparams.input_size, (list, dict)):
            self.hparams.input_size = np.prod(self.hparams.input_size)
        self.encoder = nn.Sequential(
			nn.Linear(self.hparams.input_size, self.hparams.features[0]),
			nn.SELU(),
			nn.Linear(self.hparams.features[0], self.hparams.features[1]),
			nn.SELU(),
			nn.Linear(self.hparams.features[1], self.hparams.features[2]),
			nn.SELU())
        self.decoder = nn.Sequential(
			nn.Linear(self.hparams.features[2], self.hparams.features[1]),
			nn.SELU(),
			nn.Linear(self.hparams.features[1], self.hparams.features[0]),
			nn.SELU(),
			nn.Linear(self.hparams.features[0], self.hparams.input_size))
        self.main = nn.Sequential(
			self.encoder,
			self.decoder)

        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.features[2], self.hparams.hidden_dense)
            nn.Linear(self.hparams.hidden_dense, self.hparams.num_class)
        )

    def forward(self, x):
        size=x.size()
        x = torch.flatten(x, start_dim=1)
        output=self.main(x)
        return output.view(size)

    def configure_optimizers(self):
        if self.hparams.opt == 'SGD':
            opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=True)
            if self.hparams.lr_milestones:
                return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
            else:
                return [opt]
        elif self.hparams.opt == 'Adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            if self.hparams.lr_milestones:
                return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
            else:
                return [opt]
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x) 
        loss = self.loss(out, x)

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
        loss = self.loss(out, x)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # return whatever you need for the collation function validation_end
        output = OrderedDict({
            'val_loss': loss,
        })

        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0

        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)

        tqdm_dict = {'val_loss': val_loss_mean.item()}

        return {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': val_loss_mean
            }
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, x)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'data': x,
            'label': y
        }

        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='AutoEncoder', type=str, 
                            help='network architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-milestones', nargs='+', type=int,
                                help='learning rate milestones')
        parser.add_argument('--lr-gamma', default=0.1, type=float,
                            help='number learning rate multiplied when reach the lr-milestones')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--dropout', default=0, type=float,
                                help='the rate of the dropout')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--log-weight', default=0, type=int,
                                help='log weight figure every x epoch')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
        parser.add_argument('--features', nargs='+', type=int,
                                help='numbers of dense layers nodes, return as list')
        parser.add_argument('--activation', default='Tanh', type=str,
                                help='activation function to use')

        return parser
