import time
import logging as log
from pytorch_lightning.callbacks import Callback


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
