from curses import nonl
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial2
from .util import load_variavle, save_variable, compute_mode_dim
from module.mode import DescartesExtension

def func_fitting(act_func, start, end, func, fitting_sample_number, p0):
    # generate fitting data 
    X = np.linspace(start=start, stop=end, num=fitting_sample_number)
    y = np.squeeze(act_func(X))
    # print(X.shape)
    # print(y.shape)
    # plt.plot(X, y, label=r"$f(x) = tanh(x)$", linestyle="dotted")
    # plt.legend()
    # plt.xlabel("$x$")
    # plt.ylabel("$f(x)$")
    # _ = plt.title("True generative process")

    #fitting
    popt, pcov = curve_fit(func, X, y, p0=p0)
    print('params:')
    print(popt)
    print('variance:')
    print(pcov)
    #plot
    # plt.plot(X, y,"b+:",label="data")
    # plt.plot(X, func(X, *popt),"ro:", label= "fit")
    # plt.legend()
    # plt.show()
    return popt, pcov


def make_act_layer_list(model, target_module=(nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear), remove_last_linear=True):
    # creat layer names list
    layer_list = []

    # determine whether it is in the target module, add to layer names list if true
    for name, module in model.named_modules():
        if isinstance(module, target_module):
            layer_list.append(name)
    
    # remove the last linear layer which has no activation
    if remove_last_linear:
        layer_list.pop()
    print(layer_list)

    return layer_list


def compute_range(model, layer_list, data_loader, save_range_path=None, device="cuda"):

    def hook(module, input, output):
        nonlocal idx
        nonlocal global_max
        nonlocal global_min

        if first:
            layers_range[layer_list[idx] + '_min'] = output.cpu().numpy().min(axis=0)
            layers_range[layer_list[idx] + '_max'] = output.cpu().numpy().max(axis=0)
            if layers_range[layer_list[idx] + '_min'].min() < global_min:
                global_min = layers_range[layer_list[idx] + '_min'].min()
            if layers_range[layer_list[idx] + '_max'].max() > global_max:
                global_max = layers_range[layer_list[idx] + '_max'].max()
            idx += 1
        else:
            layers_range[layer_list[idx] + '_min'] = np.minimum(layers_range[layer_list[idx]+'_min'], output.cpu().numpy().min(axis=0))
            layers_range[layer_list[idx] + '_max'] = np.maximum(layers_range[layer_list[idx]+'_max'], output.cpu().numpy().max(axis=0))
            if layers_range[layer_list[idx] + '_min'].min() < global_min:
                global_min = layers_range[layer_list[idx] + '_min'].min()
            if layers_range[layer_list[idx] + '_max'].max() > global_max:
                global_max = layers_range[layer_list[idx] + '_max'].max()
            idx += 1

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"


    # create dict to record max/min value of the conv/linear layer
    layers_range = {}
    hooks = []
    idx = 0
    first = True
    global_max = 0.0
    global_min = 0.0

    # register hook
    for name, module in model.named_modules():
        if name in layer_list:
            hooks.append(module.register_forward_hook(hook))

    # make forward pass
    model = model.to(device)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if first:
                model(data)
                first = False
                idx = 0
            else:
                model(data)
                idx = 0
   
    # remove these hooks
    for h in hooks:
        h.remove()
    
    print('range compute complete!')
    if save_range_path:
        if save_range_path.split('.')[-1] == 'csv':
            for i in range(len(layer_list)):
                temp = []
                temp.append(layers_range[layer_list[i]+'_min'])
                temp.append(layers_range[layer_list[i]+'_max'])
                temp = np.stack(temp).T
                np.savetxt(save_range_path.split('.')[0]+layer_list[i]+'.csv', temp, delimiter=',')
        else:
            save_variable(layers_range, save_range_path)

    return layers_range, global_max, global_min


def compute_hermite_params(layers_range,  # if is_global==True, it is [global_min, global_max], else it is the range dict indicating max/min value for each neuron
                           layers_list,   # the names of the layers to compute range
                           np_activation, # the activation func to fit
                           np_Hermite,    # the fitting Hermite func with params
                           p0,            # init of the params
                           save_params_pth, is_global=True, fitting_sample_number=100000, expand_range=2, progress=False, progress_freq=10):
    
    if is_global:
        start = layers_range[0]
        end = layers_range[1]

        X = np.linspace(start=start-expand_range, stop=end+expand_range, num=fitting_sample_number)
        y = np.squeeze(np_activation(X))
        #fitting
        popt, pcov = curve_fit(np_Hermite, X, y, p0=p0)
        if save_params_pth:
            save_variable(popt, save_params_pth)
            print('global hermite params saved to ', save_params_pth)
        
        return popt

    else:
        all_hermite_params = {}
        for name in layers_list:
            layer_hermite_params = []
            starts = layers_range[name+'_min'].flatten()
            ends = layers_range[name+'_max'].flatten()
            cnt = 0
            for i in range(len(starts)):
                # generate fitting data
                X = np.linspace(start=starts[i]-expand_range, stop=ends[i]+expand_range, num=fitting_sample_number)
                y = np.squeeze(np_activation(X))
                #fitting
                popt, pcov = curve_fit(np_Hermite, X, y, p0=p0)
                layer_hermite_params.append(popt)
                if progress:
                    if cnt % progress_freq == 0:
                        print('[{}/{}]'.format(cnt+1,len(starts)))
                cnt += 1
            
            all_hermite_params[name+'_hermite_params'] = layer_hermite_params

        if save_params_pth:
            save_variable(all_hermite_params, save_params_pth)
            print('perNeuron hermite params saved to ', save_params_pth)

        return all_hermite_params

class Squash_MODENN(nn.Module):
    # 用于深度网络压扁后构建MODENN测试
    def __init__(self, order, input_dim, class_num):
        super(Squash_MODENN, self).__init__()
        de_out_dim = 0
        for i in range(order):
            de_out_dim += compute_mode_dim(input_dim, i+1)

        self.de = [DescartesExtension(order=i) for i in range(order,1,-1)]
        self.fc = nn.Linear(de_out_dim, class_num)

    def forward(self, x):
        origin = torch.flatten(x, 1)
        # 按sympy 中的'grlex'顺序排列，由高阶到低阶(2阶)
        de_out = [self.de[_](x) for _ in range(len(self.de))]
        # 最后加上1阶项
        de_out.append(origin)
        # 整体变为tensor
        de_out = torch.cat(de_out, dim=1)
        # x_de = self.norm_layer(x_de)
        out = self.fc(de_out)
        return out