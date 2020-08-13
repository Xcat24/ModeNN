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
from layer import DescartesExtension
from myutils.utils import compute_cnn_out, compute_5MODE_dim, compute_mode_dim, Pretrain_Mask, find_polyitem, data_statics
from myutils.weight_analysis import load_model_weight, find_term

def W_feature_L2(tensor):
    '''
    对nn.Linear层的weight计算每个特征列的2范数，并求和。
    例：
        100维度，输出层10个类别，计算每个维度的10维向量的2范数，然后将100个结果求和
        输入：（10，100）的权值。 pytorch的权值shape是（下一层数目，上一层数目）
        输出：100个2范数的和。
    '''
    return torch.sum(torch.norm(tensor, dim=0))

def W_L2(tensor):
    return torch.norm(tensor)


class ModeNNAutoMachine(LightningModule):
    def __init__(self, hparams, loss):
        super(ModeNNAutoMachine,self).__init__()
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

        path = '/home/xucong/Log/MNIST/ModeNN/2order/best_98.35.ckpt'
        weight_name = 'fc.weight'
        w = load_model_weight(path, weight_name)
        feature_dim_norm = torch.norm(w, dim=0)
        selected_dim = set()
        selected_dim_3order = set()

        de_idx = torch.combinations(torch.arange(784), 2, with_replacement=True)
        mask_idx_2order = torch.nonzero(torch.where(feature_dim_norm[784:]>0.1, feature_dim_norm[784:], torch.zeros(feature_dim_norm[784:].shape).to(feature_dim_norm.device))).squeeze()
        mask_idx_3order = torch.nonzero(torch.where(feature_dim_norm[784:]>0.3, feature_dim_norm[784:], torch.zeros(feature_dim_norm[784:].shape).to(feature_dim_norm.device))).squeeze()
        for i in range(len(mask_idx_2order)):
            selected_dim.add(de_idx[mask_idx_2order[i]][0].item())
            selected_dim.add(de_idx[mask_idx_2order[i]][1].item())
        for i in range(len(mask_idx_3order)):
            selected_dim_3order.add(de_idx[mask_idx_3order[i]][0].item())
            selected_dim_3order.add(de_idx[mask_idx_3order[i]][1].item())
        self.select_mask, _ = torch.sort(torch.Tensor(list(selected_dim)).long())
        self.select_mask_3order, _ = torch.sort(torch.Tensor(list(selected_dim_3order)).long())
        self.de_layer = DescartesExtension(order=2)

        # path = '/home/xucong/Log/MNIST/ModeNN/WeightSelect/L2select/0.15/best_98.25.ckpt'
        # w = load_model_weight(path, weight_name)
        # feature_dim_norm = torch.norm(w, dim=0)
        # selected_dim = set()

        # de_idx = torch.combinations(torch.arange(len(self.select_mask)), 2, with_replacement=True)
        # mask_idx = torch.nonzero(torch.where(feature_dim_norm[784:]>0.15, feature_dim_norm[784:], torch.zeros(feature_dim_norm[784:].shape).to(feature_dim_norm.device))).squeeze()
        # for i in range(len(mask_idx)):
        #     selected_dim.add(self.select_mask[de_idx[mask_idx[i]][0]].item())
        #     selected_dim.add(self.select_mask[de_idx[mask_idx[i]][1]].item())
        # self.select_mask_3order, _ = torch.sort(torch.Tensor(list(selected_dim)).long())

        self.de_layer_3order = DescartesExtension(order=3)

        DE_dim = compute_mode_dim(len(self.select_mask), 2) + compute_mode_dim(len(self.select_mask_3order), 3) + 784

        if self.hparams.dropout:
            self.dropout_layer = nn.Dropout(self.hparams.dropout)
        if self.hparams.norm:
            self.norm_layer = nn.BatchNorm1d(DE_dim)

        self.fc = nn.Linear(DE_dim, self.hparams.num_classes)

    def forward(self, x):
        origin = torch.flatten(x, 1)
        self.select_mask = self.select_mask.to(x.device)
        self.select_mask_3order = self.select_mask_3order.to(x.device)

        de_input = torch.index_select(origin, dim=1, index=self.select_mask)
        out = self.de_layer(de_input)

        de_input_3order = torch.index_select(origin, dim=1, index=self.select_mask_3order)
        out_3order = self.de_layer_3order(de_input_3order)
        
        de_out = torch.cat([origin, out, out_3order], dim=-1)

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
        # out = self.softmax(out)
        return out

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]

    def training_step(self, batch, batch_nb):
        x, y = batch
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
        val_acc /= self.hparams.gpus
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': '{0:.5f}'.format(val_acc_mean)}

        logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        results = {'log': logs, 'progress_bar': tqdm_dict}
        return results

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
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
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
        parser.add_argument('--de-relu', action='store_true',
                               help='whether to use a relu after normalization layer on de data')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data')
        return parser
