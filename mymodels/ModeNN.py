import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .BaseModel import BaseModel
from .Conv import conv3x3, conv5x5, conv_init, single_conv_basic, double_conv_basic
from myutils.utils import compute_cnn_out, compute_5MODE_dim, compute_mode_dim, Pretrain_Mask, find_polyitem
from layer import DescartesExtension, MaskDE, LocalDE, SLConv, Mode, MaskLayer

class ModeNN(BaseModel):
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(ModeNN, self).__init__(hparams=hparams, loss=loss)
        if len(self.hparams.input_size) > 1:
            self.input_size = torch.tensor(self.hparams.input_size).prod().item()
        else:
            self.input_size = self.hparams.input_size[0]
        
        print('{} order Descartes Extension'.format(self.hparams.order))
        DE_dim = compute_mode_dim([self.input_size for _ in range(self.hparams.order-1)]) + self.input_size
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de_layer = Mode(order_dim=[self.input_size for _ in range(self.hparams.order-1)])

        if self.hparams.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        if self.hparams.norm:
            self.norm_layer = nn.BatchNorm1d(DE_dim)

        self.tanh = nn.Tanh()
        self.fc = nn.Linear(DE_dim, self.hparams.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        origin = torch.flatten(x, 1)
        out = self.de_layer(origin)
        de_out = torch.cat([origin, out], dim=-1)

        if self.hparams.norm:
            de_out = self.norm_layer(de_out)
        if self.hparams.dropout:
            de_out = self.dropout_layer(de_out)
    
        out = self.fc(de_out)
        # out = self.softmax(out)
        return out

    def de_forward(self, x):
        origin = torch.flatten(x, 1)
        out = self.de_layer(origin)
        de_out = torch.cat([origin, out], dim=-1)
        return de_out

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=True)
        return [opt], [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=self.hparams.lr_gamma)]

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss.item(), 'val_acc': '{0:.5f}'.format(avg_acc.item())}
        log_dict = ({'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()})
        weight_dict = {}
       
        #log weight to tensorboard
        #TODO 在大维度情况下，会产生过多的线程，导致崩溃（如CIFAR10数据）
        if self.hparams.log_weight:
            mode_para = self.fc.weight
            poly_item = find_polyitem(dim=self.input_size, order=self.hparams.order) 
            node_mean = mode_para.mean(dim=0)
            for j in range(len(node_mean)):
                w = node_mean[j].clone().detach()
                weight_dict.update({poly_item[j]:w.item()})
            self.logger.experiment.add_scalars('mode_layer_weight', weight_dict, self.current_epoch)

            if self.current_epoch%self.hparams.log_weight==0:
                #draw matplot figure
                labels = ['node{}'.format(i) for i in range(len(mode_para))]
                x = range(len(poly_item))
                fig = plt.figure(figsize=(0.2*mode_para.size()[0]*mode_para.size()[1],10))
                for i in range(len(mode_para)):
                    w = mode_para[i].cpu().numpy()
                    plt.bar([j+0.2*i for j in x], w, width=0.2, label=labels[i])
                plt.xticks(x, poly_item, rotation=-45, fontsize=6)
                plt.legend()
                self.logger.experiment.add_figure('epoch_{}'.format(self.current_epoch), fig, self.current_epoch)
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
        de_out = self.de_forward(x)
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
            'de_out': de_out,
            'data': x,
            'label': y
        }

        return output

    def test_end(self, outputs):
        whole_test_data = torch.cat([x['data'] for x in outputs], dim=0)
        whole_deout_data = torch.cat([x['de_out'] for x in outputs], dim=0)
        whole_test_label = torch.cat([x['label'] for x in outputs], dim=0)
        #logger
        if self.logger:
            self.logger.experiment.add_embedding(whole_test_data, whole_test_label, tag='raw data')
            self.logger.experiment.add_embedding(whole_deout_data, whole_test_label, tag='deout data')

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='ModeNN', type=str, 
                            help='networ architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
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
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='dims of input data, return list')
        parser.add_argument('--order', default=2, type=int,
                                help='order of Mode')
        parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
        parser.add_argument('--norm', action='store_true',
                               help='whether to use normalization layer')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data')
        return parser


class Conv_ModeNN(BaseModel):
    def __init__(self, hparams, loss=nn.CrossEntropyLoss()):
        super(Conv_ModeNN, self).__init__(hparams=hparams, loss=loss)

        self.initconv = conv3x3(self.hparams.in_channel, self.hparams.out_channels[0], stride=1)
        self.convs = self._make_layer()

        print('{} order Descartes Extension'.format(self.hparams.order))
        DE_dim = compute_mode_dim([self.hparams.conv_outshape for _ in range(self.hparams.order-1)]) + self.hparams.conv_outshape
        print('dims after DE: ', DE_dim)
        print('Estimated Total Size (MB): ', DE_dim*4/(1024*1024))
        self.de_layer = Mode(order_dim=[self.hparams.conv_outshape for _ in range(self.hparams.order-1)])

        #TODO
        conv_outshape = self.hparams.conv_outshape
        self.bn1 = nn.BatchNorm2d(self.hparams.out_channels[-1], momentum=0.9)
        self.bn2 = nn.BatchNorm1d(DE_dim)
        self.fc = nn.Linear(DE_dim, self.hparams.num_classes)

    def _make_layer(self):
        layers = []
        for _ in range(1, len(self.hparams.out_channels)):
            if self.hparams.basic_mode == 'single':
                layers.append(single_conv_basic(self.hparams.out_channels[_-1], self.hparams.out_channels[_], self.hparams.dropout, 3, self.hparams.stride))
            elif self.hparams.basic_mode == 'double':
                layer.append(double_conv_basic(self.hparams.out_channels[_-1], self.hparams.out_channels[_], self.hparams.dropout, 3, self.hparams.stride))

        return nn.Sequential(*layers)

    def de_forward(self, x):
        out = self.initconv(x)
        out = self.convs(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.hparams.pool_shape)
        # print(out.shape)
        origin = out.view(out.size(0), -1)
        out = self.de_layer(out)
        de_out = torch.cat([origin, out], dim=-1)
        return de_out
    
    def forward(self, x):
        out = self.initconv(x)
        out = self.convs(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.hparams.pool_shape)
        # print(out.shape)
        origin = out.view(out.size(0), -1)
        out = self.de_layer(origin)
        out = torch.cat([origin, out], dim=-1)
        out = self.bn2(out)
        out = self.fc(out)

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
        de_out = self.de_forward(x)
        out = self.forward(x)
        loss = self.loss(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # return whatever you need for the collation function validation_end
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
            'de_out': de_out,
            'label': y
        }

        return output

    def test_end(self, outputs):
        whole_de_data = torch.cat([x['de_out'] for x in outputs], dim=0)
        whole_test_label = torch.cat([x['label'] for x in outputs], dim=0)
        #logger
        if self.logger:
            self.logger.experiment.add_embedding(whole_de_data, whole_test_label, tag='C-MODE-de-out data')

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--num-epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--arch', default='Conv_ModeNN', type=str, 
                            help='networ architecture')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
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
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--num-classes', default=None, type=int,
                                help='number of the total classes')
        parser.add_argument('--input-size', nargs='+', type=int,
                                help='size of input data, return as list')
        parser.add_argument('--opt', default='SGD', type=str,
                                help='optimizer to use')
        parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
        parser.add_argument('--val-split', default=None, type=float,
                                help='how much data to split as the val data, now it refers to ORL dataset')
        parser.add_argument('--order', default=2, type=int,
                                help='order of Mode')
        #params in conv
        parser.add_argument('--kernel-size', nargs='+', type=int,
                                help='size of kernels, return as list, only support 3 or 5')
        parser.add_argument('--out-channels', nargs='+', type=int,
                                help='size of output channel, return as list, the length is 2 at least')
        parser.add_argument('--in-channel', default=3, type=int,
                                help='number of input channel')
        parser.add_argument('--stride', default=1, type=int,
                                help='stride')
        parser.add_argument('--dense-nodes', nargs='+', type=int,
                                help='numbers of dense layers nodels, return as list')
        parser.add_argument('--basic-mode', default='single', type=str,
                                help='conv basic to use')
        parser.add_argument('--pool-shape', default=2, type=int,
                                help='average pooling shape')
        parser.add_argument('--conv-outshape', default=1, type=int,
                                help='dimentions of conv layer output data')
        return parser