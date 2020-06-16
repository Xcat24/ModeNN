import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .BaseModel import BaseModel
from sota_module import resnet, Wide_ResNet, partial_resnet

class wide_resnet(BaseModel):
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(wide_resnet, self).__init__(hparams=hparams, loss=loss)
        self.wide_resnet = Wide_ResNet.wide_resnet(self.hparams.depth, self.hparams.width, self.hparams.dropout, self.hparams.num_classes, self.hparams.use_wrn_linear, self.hparams.use_anti_relu)

    def forward(self, x):
        return self.wide_resnet(x)
    
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), 
                            lr=self.hparams.lr, 
                            weight_decay=self.hparams.weight_decay, 
                            momentum=self.hparams.momentum, 
                            nesterov=self.hparams.nesterov)
        return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=200, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='wide_resnet', type=str, 
                            help='networ architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=128, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-milestones', nargs='+', type=int,
                                help='learning rate milestones')
        parser.add_argument('--lr-gamma', default=0.2, type=float,
                            help='number learning rate multiplied when reach the lr-milestones')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true',
                            help='use nesterov in SGD')
        parser.add_argument('--dropout', default=0.3, type=float,
                                help='the rate of the dropout')
        parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--log-weight', default=0, type=int,
                                help='log weight figure every x epoch')
        parser.add_argument('--num-classes', default=10, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data, now it refers to ORL dataset')
        #params in conv
        parser.add_argument('--depth', default=28, type=int,
                                help='number of resnet layers')
        parser.add_argument('--width', default=10, type=int,
                                help='wide factor in wide resnet')
        parser.add_argument('--use-wrn-linear', dest='use_wrn_linear', action='store_true',
                            help='use linear output layer in Wide ResNet')
        parser.add_argument('--use-anti-relu', dest='use_anti_relu', action='store_true',
                            help='use anti-relu as activation inside the conv layers in Wide ResNet')
        return parser

class pretrain_resnet(BaseModel):
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(pretrain_resnet, self).__init__(hparams=hparams, loss=loss)
        self.pretrain_resnet = partial_resnet.ResNet(self.hparams.backbone)
        self.bn = nn.BatchNorm1d(self.pretrain_resnet.out_dim)
        self.fc = nn.Linear(self.pretrain_resnet.out_dim, self.hparams.num_classes)

    def forward(self, x):
        x = self.pretrain_resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.bn(x))
        return x
    
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), 
                            lr=self.hparams.lr, 
                            weight_decay=self.hparams.weight_decay, 
                            momentum=self.hparams.momentum, 
                            nesterov=self.hparams.nesterov)
        return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=200, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='pretrain_resnet', type=str, 
                            help='network architecture')
        parser.add_argument('--backbone', default='resnet18', type=str, 
                            help='resnet architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=128, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-milestones', nargs='+', type=int,
                                help='learning rate milestones')
        parser.add_argument('--lr-gamma', default=0.2, type=float,
                            help='number learning rate multiplied when reach the lr-milestones')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true',
                            help='use nesterov in SGD')
        parser.add_argument('--dropout', default=0.3, type=float,
                                help='the rate of the dropout')
        parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--log-weight', default=0, type=int,
                                help='log weight figure every x epoch')
        parser.add_argument('--num-classes', default=10, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data, now it refers to ORL dataset')

        return parser