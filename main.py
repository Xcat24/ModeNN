import time
import random
import argparse
import logging as log
from matplotlib import pyplot as plt
import wandb
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from myutils.utils import pick_edge, Pretrain_Select, find_polyitem, kernel_heatmap
from myutils.callback import LogCallback
from myutils.datasets import train_dataloader, val_dataloader, test_dataloader, gray_cifar_train_dataloader, gray_cifar_val_dataloader


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


#================================= Read Setting End ===================================
def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--is-early-stop', action='store_true',
                                help='whether to use early stop callback')
    parent_parser.add_argument('--patience', default=50, type=int, 
                                help='the patience set in early stop callback')
    parent_parser.add_argument('--is-wandb-logger', action='store_true',
                                help='whether to use wandb')
    parent_parser.add_argument('--run-name', type=str,
                               help='the name of this run')
    parent_parser.add_argument('--is-tb-logger', action='store_true',
                                help='whether to use tensorboard')
    parent_parser.add_argument('--wandb-dir', type=str,
                               help='path to save wandb log')
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
    parent_parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
    parent_parser.add_argument('--svd', action='store_true',
                                help='whether to use svd transform to data')
    parent_parser.add_argument('--de-trans', action='store_true',
                                help='whether to use de transform to data')
    parent_parser.add_argument('--de-trans-order', type=int, default=2,
                                help='order of de transformer')
    parent_parser.add_argument('--save-path', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--pretrained', default=None, type=str,
                               help='path to the saved modal')
    parent_parser.add_argument('--gpus', type=int, default=-1,
                               help='use which gpus')
    parent_parser.add_argument('--num-workers', type=int, default=1,
                               help='how many cpu kernels to use')
    parent_parser.add_argument('--log-gpu', action='store_true',
                               help='whether to log gpu usage')
    parent_parser.add_argument('--bar', type=int, default=1,
                               help='refresh rate of the progress bar')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--precision', dest='precision', default=32, type=int,
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('--use-amp', dest='use_amp', action='store_true',
                               help='amp setting')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-t', '--test', dest='test', action='store_true',
                               help='test model on test set')
    parent_parser.add_argument('--log-modenn-weight-scalars', action='store_true',
                               help='whether to log fc weight for each term in tensorboard')
    parent_parser.add_argument('--log-modenn-weight-fig', action='store_true',
                               help='whether to log final fc weight figure in tensorboard')
    parent_parser.add_argument('--log-weight-heatmap', action='store_true',
                               help='whether to log final conv/fc weight heatmap in tensorboard')
    parent_parser.add_argument('--gray-scale', action='store_true',
                               help='whether to turn picture to gray')
    parent_parser.add_argument('--net', default='wide_resnet', type=str, 
                                help='network architecture module to load')
    parent_parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                                help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
    
    args, unknown = parent_parser.parse_known_args()

    parser = mymodels.__dict__[args.net].add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    if hparams.gray_scale:
        train_data = gray_cifar_train_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers)
        val_data = gray_cifar_val_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers)
    else:
        train_data = train_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers, svd=hparams.svd, de=hparams.de_trans, order=hparams.de_trans_order)
        val_data = val_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers, svd=hparams.svd, de=hparams.de_trans, order=hparams.de_trans_order)
        test_data = test_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers, svd=hparams.svd, de=hparams.de_trans, order=hparams.de_trans_order)
    model = mymodels.__dict__[hparams.net](hparams, nn.CrossEntropyLoss())
    # model = mymodels.BaseModel(hparams, nn.CrossEntropyLoss())
    print(model)
    if hparams.pretrained:
        state_dict = torch.load(hparams.pretrained)['state_dict']
        for name, para in model.named_parameters():
            if name in state_dict:
                with torch.no_grad():
                    para.copy_(state_dict[name])
                para.requires_grad = False
    
    # summary(model, input_size=tuple(hparams.input_size), device='cpu')

    # if hparams.seed is not None:
    #     random.seed(hparams.seed)
    #     torch.manual_seed(hparams.seed)
    #     cudnn.deterministic = True
    if hparams.is_early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=hparams.patience,
            verbose=True,
            mode='auto'
        )
    else:
        early_stop_callback = False

    if hparams.is_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            filepath=hparams.saved_path,
            save_top_k = 1,
            verbose=True,
            monitor='val_acc',
            mode='max'
        )
    else:
        checkpoint_callback = False

    loggers = []
            
    if hparams.is_wandb_logger:    
        wandb_logger = WandbLogger(name=hparams.run_name, project='modenn', save_dir=hparams.wandb_dir)#, offline=True)
        loggers.append(wandb_logger)  

    if hparams.is_tb_logger:
        tb_logger = TensorBoardLogger(
            save_dir=hparams.tb_dir,
            name=hparams.run_name
            )
        loggers.append(tb_logger)
    
    # callbacks = [LogCallback()]

    trainer = Trainer(
        min_epochs=1,
        max_epochs=hparams.num_epochs,
        log_gpu_memory=hparams.log_gpu,
        weights_summary='full',
        gpus=hparams.gpus,
        fast_dev_run=False, #activate callbacks, everything but only with 1 training and 1 validation batch
        gradient_clip_val=0,  #this will clip the gradient norm computed over all model parameters together
        track_grad_norm=-1,  #Looking at grad norms
        precision=hparams.precision,
        auto_lr_find=False,
        # distributed_backend='ddp',
        # use_amp=hparams.use_amp,
        # print_nan_grads=True,
        checkpoint_callback=checkpoint_callback,
        logger=loggers,
        progress_bar_refresh_rate=hparams.bar,
        # row_log_interval=80,
        # log_save_interval=80,
        early_stop_callback=early_stop_callback,
        # callbacks=callbacks,
        profiler=True
        )

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model, train_data, val_data)
    if hparams.test:
        trainer.test(test_data)


if __name__ == '__main__':
    main(get_args())   




