import torch

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

def find_polyitem(dim, order):
    result = ['x{}'.format(_) for _ in range(dim)]
    for i in range(1, order):
        temp = torch.combinations(torch.arange(0,dim), i+1, with_replacement=True)
        for j in range(len(temp)):
            item = ''
            for k in range(len(temp[j])):
                item += 'x{}'.format(temp[j,k])
            result.append(item)
    return result

class ModelHooks(torch.nn.Module):

    def on_sanity_check_start(self):
        """
        Called before starting evaluate
        :return:
        """
        pass

    def on_batch_start(self, batch):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_pre_performance_check(self):
        pass

    def on_post_performance_check(self):
        pass

    def on_before_zero_grad(self, optimizer):
        """
        Called after optimizer.step() and before optimizer.zero_grad()

        for optimizer in optimizers:
            optimizer.step()
            model.on_before_zero_grad(optimizer) # < ---- called here
            optimizer.zero_grad

        :param optimizer:
        :return:
        """
        #logger
        # if self.logger:
        #     layer_names = list(self._modules)
        #     for i in range(len(layer_names)):
        #         mod_para = list(self._modules[layer_names[i]].parameters())
        #         if mod_para:
        #             for j in range(len(mod_para)):
        #                 self.logger.experiment.add_histogram(layer_names[i]+'_'+str(mod_para[j].shape)+'_weight-grad', mod_para[j].grad)
        #log gradient
        if self.log_weight:
            mode_para = self.fc.weight.grad
            grad_dict = {}
            try:
                poly_item = find_polyitem(dim=self.input_size, order=self.order) 
                node_mean = mode_para.mean(dim=0)
                for j in range(len(node_mean)):
                    w = node_mean[j].clone().detach()
                    grad_dict.update({poly_item[j]+'_grad':w.item()})
                self.logger.experiment.add_scalars('weight_grad', grad_dict)
            except TypeError as e:
                pass
        return
    
    def on_after_backward(self):
        """
        Called after loss.backward() and before optimizers do anything
        :return:
        """
        pass
    
    def backward(self, use_amp, loss, optimizer):
        """
        Override backward with your own implementation if you need to
        :param use_amp: Whether amp was requested or not
        :param loss: Loss is already scaled by accumulated grads
        :param optimizer: Current optimizer being used
        :return:
        """
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    
    def data_statics(self, tag, data, verbose=False):
        """
        compute the statics of the tensor
        """
        if not isinstance(data, torch.Tensor):
            x = torch.tensor(data).to(torch.device('cpu'))
        else:
            x = torch.tensor(data).to(torch.device('cpu'))
        max = x.max().item()
        min = x.min().item()
        mean = x.mean().item()
        std = x.std().item()
        pos_count = (torch.gt(x, torch.zeros(x.shape, device=x.device))==True).sum().item()
        neg_count = (torch.lt(x, torch.zeros(x.shape, device=x.device))==True).sum().item()
        zero_count = (torch.eq(x, torch.zeros(x.shape, device=x.device))==True).sum().item()
        if verbose == True:
            print('-------------------------'+ tag +' statics---------------------------')
            print('data shape of: ', x.shape)
            print('max:  ', max)
            print('min:  ', min)
            print('mean: ', mean)
            print('std:  ', std)
            print('number of great than 0: ', pos_count)
            print('number of less than 0:  ', neg_count)
            print('number of equal to 0:   ', zero_count)
        return