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
from myutils.datasets import train_dataloader, val_dataloader, test_dataloader, gray_cifar_train_dataloader, gray_cifar_val_dataloader


# Device configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


#================================= Read Setting End ===================================

class LogCallback(Callback):
    def on_fit_start(self, trainer):
        """Called when the fit begins."""
        pass
    
    def on_fit_end(self, trainer):
        """Called when the fit ends."""
        pass

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        self.t0 = time.time()

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        #log term weight per epoch
        # if pl_module.hparams.log_modenn_weight_scalars:
        #     weight_dict = {}
        #     if pl_module.hparams.log_weight:
        #         mode_para = pl_module.fc.weight
        #         poly_item = find_polyitem(dim=pl_module.hparams.input_size, order=pl_module.hparams.order)
        #         node_mean = mode_para.mean(dim=0)
        #         for j in range(len(node_mean)):
        #             w = node_mean[j].clone().detach()
        #             weight_dict.update({poly_item[j]:w.item()})
        #         self.logger.experiment.add_scalars('mode_layer_weight', weight_dict, self.current_epoch)

        #print result
        if not pl_module.hparams.bar:
            # print(logs.keys())
            logs = trainer.callback_metrics
            epoch = trainer.current_epoch
            log.info('  epoch {0}: train_loss={1:.3f}, val_loss={2:.3f}, val_acc={3:.4f}%,  time spend={4:.1f} mins'.format(
                epoch, logs.get('training_loss'), logs.get('val_loss'), 100*logs.get('val_acc'), (time.time()-self.t0)/60.0))

    def on_batch_start(self, trainer, pl_module):
        """Called when the training batch begins."""
        pass

    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        #plt.bar部分在高维度时用时太多，暂不采用
        if pl_module.hparams.log_modenn_weight_fig:
            log.info('plot init modenn fc weight figure into tensorboard.')
            mode_para = pl_module.fc.weight
            poly_item = find_polyitem(dim=pl_module.hparams.input_size, order=pl_module.hparams.order)
            labels = ['node{}'.format(i) for i in range(len(mode_para))]
            x = range(len(poly_item))
            fig = plt.figure(figsize=(0.2*mode_para.size()[0]*mode_para.size()[1],10))
            for i in range(len(mode_para)):
                w = mode_para[i].cpu().detach().numpy()
                plt.bar([j+0.2*i for j in x], w, width=0.2, label=labels[i])
            plt.xticks(x, poly_item, rotation=-45, fontsize=6)
            plt.legend()
            pl_module.logger.experiment.add_figure('init weight', fig)
        
        log.info('training begins...')
        

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
                   
        #log weight to tensorboard
        if pl_module.hparams.log_modenn_weight_fig:
            log.info('plot final modenn fc weight figure into tensorboard.')
            mode_para = pl_module.fc.weight
            poly_item = find_polyitem(dim=pl_module.hparams.input_size, order=pl_module.hparams.order)
            labels = ['node{}'.format(i) for i in range(len(mode_para))]
            x = range(len(poly_item))
            fig = plt.figure(figsize=(0.2*mode_para.size()[0]*mode_para.size()[1],10))
            for i in range(len(mode_para)):
                w = mode_para[i].cpu().detach().numpy()
                plt.bar([j+0.2*i for j in x], w, width=0.2, label=labels[i])
            plt.xticks(x, poly_item, rotation=-45, fontsize=6)
            plt.legend()
            pl_module.logger.experiment.add_figure('final weight', fig)

        #draw final weight heatmap in tensorboard
        if pl_module.hparams.log_weight_heatmap:
            log.info('drawing final conv/fc weight heatmap in wandb...')
            for name, para in pl_module.named_parameters():
                if 'weight' in name:
                    f = kernel_heatmap(para, name)
                    if f:
                        pl_module.logger[0].experiment.log({"examples": [wandb.Image(f, caption='final_'+name)]})
                        # pl_module.logger[0].experiment.add_image('final_'+name, f, 0)


        log.info('training completed!')

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        pass


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
    parent_parser.add_argument('--save-path', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--pretrained', default=None, type=str,
                               help='path to the saved modal')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--log-gpu', action='store_true',
                               help='whether to log gpu usage')
    parent_parser.add_argument('--bar', type=int, default=1,
                               help='refresh rate of the progress bar')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--precision', dest='precision', default=32, type=int,
                               help='if true uses 16 bit precision')
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
        train_data = gray_cifar_train_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size)
        val_data = gray_cifar_val_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size)
    else:
        train_data = train_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size)
        val_data = val_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size)
        test_data = test_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size)
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
    
    callbacks = [LogCallback()]

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
        # print_nan_grads=True,
        checkpoint_callback=checkpoint_callback,
        logger=loggers,
        progress_bar_refresh_rate=hparams.bar,
        # row_log_interval=80,
        # log_save_interval=80,
        early_stop_callback=early_stop_callback,
        callbacks=callbacks
        )

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model, train_data, val_data)
    if hparams.test:
        trainer.test(test_dataloader)


if __name__ == '__main__':
    main(get_args())   




