import time
import random
import argparse
import logging as log
from matplotlib import pyplot as plt
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
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger
from myutils.utils import pick_edge, Pretrain_Select, find_polyitem, kernel_heatmap


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


class LogCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        """Called when the fit begins."""
        pass
    
    def on_fit_end(self, trainer, pl_module):
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
            log.info('drawing final conv/fc weight heatmap in tensorboard...')
            for name, para in pl_module.named_parameters():
                if 'weight' in name:
                    f = kernel_heatmap(para, name)
                    if f:
                        pl_module.logger.experiment.add_figure('final_'+name, f)


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
    parent_parser.add_argument('--pretrained', default=None, type=str,
                               help='path to the saved modal')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--log-gpu', action='store_true',
                               help='whether to log gpu usage')
    parent_parser.add_argument('--bar', action='store_true',
                               help='if true print progress bar')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
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
    parent_parser.add_argument('--net', default='wide_resnet', type=str, 
                                help='network architecture module to load')
    
    args, unknown = parent_parser.parse_known_args()

    parser = mymodels.__dict__[args.net].add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = mymodels.__dict__[hparams.net](hparams, nn.CrossEntropyLoss())

    if hparams.pretrained:
        state_dict = torch.load(hparams.pretrained)['state_dict']
        for name, para in model.named_parameters():
            if name in state_dict:
                with torch.no_grad():
                    para.copy_(state_dict[name])
                para.requires_grad = False
    
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
            save_top_k = 1,
            verbose=True,
            monitor='val_acc',
            mode='max'
        )
    else:
        checkpoint_callback = None

    if hparams.is_tensorboard:
        tb_logger = TestTubeLogger(
            save_dir=hparams.log_dir,
            name=hparams.tb_dir
            )
    else:
        tb_logger = False
        
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
        use_amp=hparams.use_16bit,
        # print_nan_grads=True,
        checkpoint_callback=checkpoint_callback,
        logger=tb_logger,
        show_progress_bar=hparams.bar,
        # row_log_interval=80,
        # log_save_interval=80,
        early_stop_callback=early_stop_callback,
        callbacks=callbacks
        )

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)
    if hparams.test:
        trainer.test()


if __name__ == '__main__':
    main(get_args())   




