from argparse import ArgumentParser
from sklearn.datasets import fetch_california_housing
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from utils.util import compute_mode_dim
from module.mode import Mode


class ModeNN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # print(self.hparams)
        self.save_hyperparameters()
        if len(self.hparams.input_size) > 1:
            self.input_size = torch.tensor(self.hparams.input_size).prod().item()
        else:
            self.input_size = self.hparams.input_size[0]
        if self.hparams.pooling:
            self.input_size //= (self.hparams.pooling*self.hparams.pooling)

        print('{} order Descartes Extension'.format(self.hparams.order))
        DE_dim = compute_mode_dim([self.input_size for _ in range(self.hparams.order-1)]) + self.input_size
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de_layer = Mode(order_dim=[self.input_size for _ in range(self.hparams.order-1)])

        if self.hparams.dropout:
            self.dropout_layer = nn.Dropout(self.hparams.dropout)
        if self.hparams.norm:
            self.norm_layer = nn.BatchNorm1d(DE_dim)

        if self.hparams.hidden_nodes:
            self.hidden = nn.Linear(DE_dim, self.hparams.hidden_nodes)
            self.hidden_bn = nn.BatchNorm1d(self.hparams.hidden_nodes)
            self.fc = nn.Linear(self.hparams.hidden_nodes, self.hparams.num_classes)
        else:
            self.fc = nn.Linear(DE_dim, self.hparams.num_classes)
        
        if self.hparams.init_prune:
            prune.random_unstructured(self.fc, name="weight", amount=self.hparams.prune_amount)

    def forward(self, x):
        if self.hparams.pooling:
            x = torch.nn.MaxPool2d(2)(x)
        origin = torch.flatten(x, 1)
        out = self.de_layer(origin)
        de_out = torch.cat([origin, out], dim=-1)

        if self.hparams.norm:
            de_out = self.norm_layer(de_out)
            if self.hparams.de_relu:
                de_out = F.relu(de_out)
        if self.hparams.dropout:
            de_out = self.dropout_layer(de_out)

        # de_out = self.tanh(de_out)
        if self.hparams.hidden_nodes:
            de_out = self.hidden_bn(F.relu(self.hidden(de_out))) #实际上是hidde_out

        out = self.fc(de_out)
        if self.hparams.out_tanh:
            out = F.tanh(out)
        if self.hparams.softmax:
            out = F.softmax(out)
        return out

    def loss(self, pred, y):
        if self.loss is not None:
            # print('Using Cross Entropy as the loss function!')
            if len(y.shape) == 2:
                y = torch.squeeze(y)
            return F.cross_entropy(pred, y)
        else:
            return self.loss(pred, y)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        if self.hparams.lr_milestones:
            return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]
        else:
            return [opt], [torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)]


    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        train_acc = accuracy(out, y)
        self.log('train_loss', loss.detach(), on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', train_acc.detach(), on_epoch=True, prog_bar=True, logger=True)
        log = {"loss": loss, "train_acc": train_acc, "output": out}

        return log

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        val_acc = accuracy(out, y)
        self.log('val_loss', loss.detach())
        self.log('val_acc', val_acc.detach())

        return {'val_loss': loss, "val_acc": val_acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        test_acc = accuracy(out, y)
        self.log('test_loss', loss.detach())
        self.log('test_acc', test_acc.detach())


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--init-prune', action='store_true',
                               help='whether to prune fc weight when initialize fc layer')
        parser.add_argument('--prune-amount', default=0.2, type=float,
                            help='pruning rate of the init-prune')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            help='initial learning rate', dest='lr')
        parser.add_argument('--lr-milestones', nargs='+', type=int,
                                help='learning rate milestones')
        parser.add_argument('--lr-gamma', default=0.1, type=float,
                            help='number learning rate multiplied when reach the lr-milestones')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true',
                            help='use nesterov in SGD')
        parser.add_argument('--dropout', default=0, type=float,
                                help='the rate of the dropout')
        parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--hidden-nodes', default=0, type=int,
                                help='use how many hidden nodes between de-layer and out-layer, 0 is not to use hidden layer(default)')
        parser.add_argument('--log-weight', default=0, type=int,
                                help='log weight figure every x epoch')
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='dims of input data, return list')
        parser.add_argument('--order', default=2, type=int,
                                help='order of Mode')
        parser.add_argument('--norm', action='store_true',
                               help='whether to use normalization layer')
        parser.add_argument('--out-tanh', action='store_true',
                               help='whether to use tanh activation in output layer')
        parser.add_argument('--softmax', action='store_true',
                               help='whether to use softmax in output layer')
        parser.add_argument('--de-relu', action='store_true',
                               help='whether to use a relu after normalization layer on de data')
        parser.add_argument('--pooling',default=0, type=int,
                               help='whether to decrease dimentions first')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data')
        return parser

