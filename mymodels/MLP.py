import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from .BaseModel import BaseModel
import mymodels.activation
from myutils.utils import compute_mode_dim



class MLP(BaseModel):
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(MLP, self).__init__(hparams=hparams, loss=loss)
        if self.hparams.de_trans:
            self.hparams.dense_nodes = [compute_mode_dim([np.prod(self.hparams.input_size) for _ in range(self.hparams.de_trans_order)])]
        #TODO
        self.fc = self._make_dense()
        self.out_layer = nn.Linear(self.hparams.dense_nodes[-1], self.hparams.num_classes)

    def _make_dense(self):
        layers = [nn.Linear(np.prod(self.hparams.input_size), self.hparams.dense_nodes[0])]
        if self.hparams.custom_activation:
            layers.append(mymodels.activation.__dict__[self.hparams.activation]())
        else:
            layers.append(nn.__dict__[self.hparams.activation]())
        if len(self.hparams.dense_nodes) > 1:
            for _ in range(1, len(self.hparams.dense_nodes)):
                layers.append(nn.Linear(self.hparams.dense_nodes[_-1],self.hparams.dense_nodes[_]))
                if self.hparams.custom_activation:
                    layers.append(mymodels.activation.__dict__[self.hparams.activation]())
                else:
                    layers.append(nn.__dict__[self.hparams.activation]())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        out = self.fc(x)
        out = self.out_layer(out)

        return out

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
            'data': x,
            'label': y
        }

        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='MLP', type=str, 
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
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data, now it refers to ORL dataset')
        parser.add_argument('--dense-nodes', nargs='+', type=int,
                                help='numbers of dense layers nodels, return as list')
        parser.add_argument('--custom-activation', action='store_true',
                               help='whether to use customized activation function')
        parser.add_argument('--activation', default='Tanh', type=str,
                                help='activation function to use')

        return parser
