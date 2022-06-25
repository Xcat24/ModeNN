from argparse import ArgumentParser
import comet_ml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, ModelPruning
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
import models
from data.dataset import *

#================================= Read Setting End ===================================
def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--is-early-stop', action='store_true',
                                help='whether to use early stop callback')
    parent_parser.add_argument('--patience', default=50, type=int, 
                                help='the patience set in early stop callback')
    parent_parser.add_argument('--max-epochs', default=10, type=int, 
                                help='the epochs number')
    # parent_parser.add_argument('--gpus', nargs='+', type=int,
    #                            help='use which gpus')
    parent_parser.add_argument('--run-name', type=str,
                               help='the name of this run')
    parent_parser.add_argument('--project-name', type=str,
                               help='the project name of the loggers')
    parent_parser.add_argument('--is-wandb-logger', action='store_true',
                                help='whether to use wandb logger')
    parent_parser.add_argument('--wandb-dir', type=str,
                               help='path to save wandb log')
    parent_parser.add_argument('--is-tb-logger', action='store_true',
                                help='whether to use tensorboard logger')
    parent_parser.add_argument('--tb-dir', type=str,
                               help='path to save tensorboard log')
    parent_parser.add_argument('--is-comet-logger', action='store_true',
                                help='whether to use comet logger')
    parent_parser.add_argument('--comet-dir', type=str,
                               help='path to save comet log')
    parent_parser.add_argument('--is-checkpoint', action='store_true',
                                help='whether to use check point callback')
    parent_parser.add_argument('--model-name', type=str,
                               help='the model name')
    parent_parser.add_argument('--saved-path', metavar='DIR', type=str,
                               help='path to save model')
    parent_parser.add_argument('--dataset', type=str,
                               help='dataset to use')
    parent_parser.add_argument('--data-dir', type=str,
                               help='path to dataset')
    parent_parser.add_argument('--augmentation', action='store_true',
                               help='whether to use data augmentation preprocess, now only availbale for CIFAR10 dataset')
    parent_parser.add_argument('--random-sample-trans', action='store_true',
                                help='whether to use random select from all feature dims transformer to data')
    parent_parser.add_argument('--random-trans-in-dim', type=int, default=784,
                                help='input dim of random sample transformer')
    parent_parser.add_argument('--random-trans-out-dim', type=int, default=316,
                                help='output dim of random sample transformer')
    parent_parser.add_argument('--de-trans', action='store_true',
                                help='whether to use de transform to data')
    parent_parser.add_argument('--randomsample-de-trans', action='store_true',
                                help='whether to use randomsample-de transform to data')
    parent_parser.add_argument('--hog', action='store_true',
                                help='whether to use hog transform to data')
    parent_parser.add_argument('--sample-group-num', type=int, default=64,
                                help='group number of sample de transformer')
    parent_parser.add_argument('--de-trans-order', type=int, default=2,
                                help='order of de transformer')
    parent_parser.add_argument('--svd', action='store_true',
                               help='whether to use svd')
    parent_parser.add_argument('--dct', action='store_true',
                               help='whether to use dct')
    parent_parser.add_argument('--dct-dim', type=int, default=10,
                               help='dct feature dimension')
    parent_parser.add_argument('--save-path', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--pretrained', default=None, type=str,
                               help='path to the saved modal')
    parent_parser.add_argument('--num-workers', type=int, default=4,
                               help='how many cpu kernels to use')
    parent_parser.add_argument('--log-gpu', action='store_true',
                               help='whether to log gpu usage')
    parent_parser.add_argument('--bar', type=int, default=1,
                               help='refresh rate of the progress bar')
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
    parent_parser.add_argument('--loss', default='cross_entropy', type=str, 
                                help='Loss function to use')
    parent_parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                                help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
    
    args, unknown = parent_parser.parse_known_args()

    parser = models.__dict__[args.net].add_model_specific_args(parent_parser)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main(hparams):
    #-----------------------------------------  seed setting -----------------------------------------#
    pl.seed_everything(hparams.seed)

    #-----------------------------------------  data setting -----------------------------------------#
    if hparams.gray_scale:
        train_data = gray_cifar_train_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers)
        val_data = gray_cifar_val_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers)
    else:
        train_data = train_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers, random_sample=hparams.random_sample_trans, random_in_dim=hparams.random_trans_in_dim,
                                     random_out_dim=hparams.random_trans_out_dim, svd=hparams.svd, dct=hparams.dct, dct_dim=hparams.dct_dim, de=hparams.de_trans,
                                     randomsample_de=hparams.randomsample_de_trans, group_num=hparams.sample_group_num, order=hparams.de_trans_order, hog=hparams.hog)
        val_data = val_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers, random_sample=hparams.random_sample_trans, random_in_dim=hparams.random_trans_in_dim,
                                     random_out_dim=hparams.random_trans_out_dim, svd=hparams.svd, dct=hparams.dct, dct_dim=hparams.dct_dim, de=hparams.de_trans,
                                     randomsample_de=hparams.randomsample_de_trans, group_num=hparams.sample_group_num, order=hparams.de_trans_order, hog=hparams.hog)
        test_data = test_dataloader(hparams.dataset, hparams.data_dir, hparams.batch_size, hparams.num_workers, random_sample=hparams.random_sample_trans, random_in_dim=hparams.random_trans_in_dim,
                                     random_out_dim=hparams.random_trans_out_dim, svd=hparams.svd, dct=hparams.dct, dct_dim=hparams.dct_dim, de=hparams.de_trans,
                                     randomsample_de=hparams.randomsample_de_trans, group_num=hparams.sample_group_num, order=hparams.de_trans_order, hog=hparams.hog)

    #----------------------------------------- model setting -----------------------------------------#
    dict_args = vars(hparams)
    model = models.__dict__[hparams.net](**dict_args)
    # model = mymodels.BaseModel(hparams, nn.CrossEntropyLoss())
    print(model)
    if hparams.pretrained:
        model = model.load_from_checkpoint(hparams.pretrained)
        state_dict = torch.load(hparams.pretrained)['state_dict']
        for name, para in model.named_parameters():
            if name in state_dict:
                with torch.no_grad():
                    para.copy_(state_dict[name])
                # para.requires_grad = False
    
    # summary(model, input_size=tuple(hparams.input_size), device='cpu')

    # if hparams.seed is not None:
    #     random.seed(hparams.seed)
    #     torch.manual_seed(hparams.seed)
    #     cudnn.deterministic = True

    #-----------------------------------------callback setting-----------------------------------------#
    callbacks = [ModelSummary(max_depth=-1)]
    if hparams.is_early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=hparams.patience,
            verbose=True,
            mode='auto'
        )
        callbacks.append(early_stop_callback)

    if hparams.is_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=hparams.saved_path,
            filename=hparams.model_name+'_{epoch:03d}_{val_loss:.4f}-{val_acc:.5f}',
            save_top_k = 2,
            verbose=True,
            monitor='val_acc',
            mode='max'
        )
        callbacks.append(checkpoint_callback)
    
    #----------------------------------------- Logger setting -----------------------------------------#
    loggers = []
    if hparams.is_wandb_logger:    
        wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, save_dir=hparams.wandb_dir)#, offline=True)
        loggers.append(wandb_logger)  

    if hparams.is_tb_logger:
        tb_logger = TensorBoardLogger(
            save_dir=hparams.tb_dir,
            name=hparams.run_name
            )
        loggers.append(tb_logger)

    if hparams.is_comet_logger:
        comet_logger = CometLogger(
            api_key="28zlOqMqpf8H0CoWfMWu72FtO",
            workspace='xcat24',  # Optional
            save_dir=hparams.comet_dir,  # Optional
            project_name=hparams.project_name,  # Optional
            experiment_name=hparams.run_name,  # Optional
            )
        loggers.append(comet_logger)
    
    #
    # for i in range(len(loggers)):
    #     loggers[i].experiment.log_parameters(hparams)
    # comet_logger.experiment.log_parameters(dict_args)

    #-----------------------------------------    Training    -----------------------------------------#
    trainer = pl.Trainer.from_argparse_args(hparams, logger=loggers, callbacks=callbacks)
    if hparams.is_wandb_logger: 
        wandb_logger.watch(model) 

    trainer.fit(model, train_data, val_data)

    #-----------------------------------------     Testing    -----------------------------------------#
    if hparams.test:
        trainer.test(test_dataloaders=val_data)


if __name__ == '__main__':
    main(get_args())