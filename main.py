import math
import time
import os
import random
import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import mymodels
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from myutils.utils import pick_edge, Pretrain_Select


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


#================================= Read Setting End ===================================

# model = MyModel.MyCNN_MODENN(input_size=input_size[2:], in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, num_classes=num_classes, pool_shape=(2,2),
#                             order=order, padding=1, norm=norm, dropout=dropout, dataset=dataset, learning_rate=learning_rate, weight_decay=weight_decay, output_debug=False)


# model = MyModel.CIFARConv_MODENN(input_size=input_size[2:], in_channel=in_channel, layer_num=layer_num, pooling='Max',
#                          dense_node=dense_node, kernel_size=kernel_size, num_classes=num_classes, order=order, padding=1, norm=norm,
#                          dropout=dropout, dataset=dataset)

# model = MyModel.SLCNN(input_size=input_size[2:], in_channel=in_channel, stride=1, pooling='Max', pool_shape=(4,4), learning_rate=learning_rate, 
#                          weight_decay=weight_decay, num_classes=num_classes, padding=1, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.SLCNN_MODENN(input_size=input_size[2:], in_channel=in_channel, stride=1, pooling='Max', pool_shape=(4,4), learning_rate=learning_rate, 
                        #  weight_decay=weight_decay, num_classes=num_classes, order=order, padding=1, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.NoHiddenBase(input_size=input_size[1:], learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.Select_MODE(input_size=input_size[-1], model_path=pretrain_model, order_dim=[300, 55, 25, 15], learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes, norm=norm, dropout=dropout, dataset=dataset)


# model = MyModel.OneHiddenBase(input_size=input_size[1:], learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes, norm=norm, dropout=dropout, dataset=dataset)

# model = MyModel.Pretrain_5MODENN(num_classes=num_classes,bins_size=9, bins_num=35,
#                      dropout=None, learning_rate=learning_rate,weight_decay=weight_decay, loss=nn.CrossEntropyLoss(),
#                      dataset=dataset)

# model = MyModel.resnext29(input_size=input_size[2:], in_channel=in_channel, num_classes=num_classes, dataset=dataset)

# model = MyModel.resnet18(num_classes=num_classes, dataset=dataset)

# model = MyModel.wide_resnet(depth=28, width=10, dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay, num_classes=num_classes,dataset=dataset)

# model = MyModel.C_MODENN(input_size=input_size[1:], in_channel=in_channel, out_channel=out_channel, order=order, num_classes=num_classes, share_fc_weights=share_fc_weights,
#                          norm=norm, learning_rate=learning_rate, weight_decay=weight_decay, dataset=dataset, log_weight=0, lr_milestones=lr_milestones)




def get_args(arch):
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--is-early-stop', action='store_true',
                                help='whether to use early stop callback')
    parent_parser.add_argument('--patience', default=50, type=int, 
                                help='the patience set in early stop callback')
    parent_parser.add_argument('--is-tensorboard', action='store_true',
                                help='whether to use tensorboard')
    parent_parser.add_argument('--log-dir', type=str,
                               help='path to save log')
    parent_parser.add_argument('--tb-dir', type=str,
                               help='path to save tensorboard')
    parent_parser.add_argument('--is-checkpoint', action='store_true',
                                help='whether to use check point callback')
    parent_parser.add_argument('--saved-path', metavar='DIR', type=str,
                               help='path to save model')
    parent_parser.add_argument('--dataset', type=str,
                               help='dataset to use')
    parent_parser.add_argument('--data-dir', type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save-path', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--log-gpu', action='store_true',
                               help='whether to log gpu usage')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use-16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-t', '--test', dest='test', action='store_true',
                               help='test model on test set')

    parser = mymodels.__dict__[arch].add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = mymodels.__dict__[hparams.arch](hparams, nn.CrossEntropyLoss())
    summary(model, input_size=tuple(hparams.input_size), device='cpu')
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        cudnn.deterministic = True
    if hparams.is_early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=hparams.patience,
            verbose=True,
            mode='auto'
        )
    else:
        early_stop_callback = None

    if hparams.is_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            filepath=hparams.saved_path,
            save_best_only=True,
            verbose=True,
            monitor='val_acc',
            mode='max',
            prefix=''
        )
    else:
        checkpoint_callback = None

    if hparams.is_tensorboard:
        tb_logger = TestTubeLogger(
            save_dir=hparams.log_dir,
            name=hparams.tb_dir,
            debug=False,
            create_git_tag=False)
    else:
        tb_logger = None
        
    trainer = Trainer(
        min_nb_epochs=1,
        max_nb_epochs=hparams.num_epochs,
        log_gpu_memory=hparams.log_gpu,
        gpus=hparams.gpus,
        fast_dev_run=False, #activate callbacks, everything but only with 1 training and 1 validation batch
        gradient_clip_val=0,  #this will clip the gradient norm computed over all model parameters together
        track_grad_norm=-1,  #Looking at grad norms
        # print_nan_grads=True,
        checkpoint_callback=checkpoint_callback,
        logger=tb_logger,
        # row_log_interval=80,
        # log_save_interval=80,
        early_stop_callback=early_stop_callback)

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)
    if hparams.test:
        trainer.test()


if __name__ == '__main__':
    main(get_args('wide_resnet'))
    # main(get_args('MyConv2D'))
    # main(get_args('Conv_ModeNN'))
    # main(get_args('ModeNN'))     




