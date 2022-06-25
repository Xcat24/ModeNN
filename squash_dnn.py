
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import numpy as np
from data.dataset import *
from module.activations import h_poly_init, np_hermite_poly, t_hermite_poly, hermite_poly_act
from utils.squash import *


def test(model, device, test_loader, batch_size):
    model.eval()
    test_loss = 0
    Acc = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

            # print(accuracy(output, target))
            Acc += accuracy(output, target)
            correct += accuracy(output,target)*batch_size

    test_loss /= len(test_loader.dataset)
    Acc /= (len(test_loader.dataset)/batch_size)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

class origin_mlp(torch.nn.Module):
    def __init__(self, dense_nodes, input_size, class_num, norm):
        super(origin_mlp, self).__init__()
        self.dense_nodes = dense_nodes
        self.input_size = input_size
        self.norm = norm
        self.fc = self._make_dense()
        self.out_layer = nn.Linear(self.dense_nodes[-1], class_num)

    def _make_dense(self):

        layers = [nn.Linear(self.input_size, self.dense_nodes[0])]
        layers.append(nn.Tanh())
        if self.norm:
            layers.append(nn.BatchNorm1d(self.dense_nodes[0]))

        if len(self.dense_nodes) > 1:
            for _ in range(1, len(self.dense_nodes)):
                layers.append(nn.Linear(self.dense_nodes[_-1], self.dense_nodes[_]))
                layers.append(nn.Tanh())
                if self.norm:
                    layers.append(nn.BatchNorm1d(self.dense_nodes[_]))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        out = self.fc(x)
        out = self.out_layer(out)
        # out = F.tanh(out)
        # out = F.softmax(out)
        return out


class hermite_tanh_mlp(torch.nn.Module):
    def __init__(self, dense_nodes, input_size, class_num, norm, order, params, layer_list, is_global=False):
        super(hermite_tanh_mlp, self).__init__()
        self.dense_nodes = dense_nodes
        self.input_size = input_size
        self.norm = norm
        self.order = order
        self.params = params
        self.layer_list = layer_list
        self.is_global = is_global

        self.fc = self._make_dense()
        self.out_layer = nn.Linear(self.dense_nodes[-1], class_num)

    def _make_dense(self):
        if self.is_global:
            layers = [nn.Linear(self.input_size, self.dense_nodes[0])]
            layers.append(hermite_poly_act(self.order, self.params, True))
            if self.norm:
                layers.append(nn.BatchNorm1d(self.dense_nodes[0]))
            
            if len(self.dense_nodes) > 1:
                for _ in range(1, len(self.dense_nodes)):
                    layers.append(nn.Linear(self.dense_nodes[_-1], self.dense_nodes[_]))
                    layers.append(hermite_poly_act(self.order, self.params, True))
                    if self.norm:
                        layers.append(nn.BatchNorm1d(self.dense_nodes[_]))
        
        else:
            idx = 0

            layers = [nn.Linear(self.input_size, self.dense_nodes[0])]
            layers.append(hermite_poly_act(self.order, self.params[self.layer_list[idx]+'_hermite_params']))
            idx += 1
            if self.norm:
                layers.append(nn.BatchNorm1d(self.dense_nodes[0]))

            if len(self.dense_nodes) > 1:
                for _ in range(1, len(self.dense_nodes)):
                    layers.append(nn.Linear(self.dense_nodes[_-1], self.dense_nodes[_]))
                    layers.append(hermite_poly_act(self.order, self.params[self.layer_list[idx]+'_hermite_params']))
                    idx += 1
                    if self.norm:
                        layers.append(nn.BatchNorm1d(self.dense_nodes[_]))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        out = self.fc(x)
        out = self.out_layer(out)

        return out



def main():
    #------------------------------ arges ------------------------------#
    # MNIST
    dataset = 'MNIST'
    data_dir = '/disk/Dataset/MNIST'
    dense_nodes = [2500,2000,1500,1000,500]
    input_size = 784
    class_num = 10
    norm = True
    batch_size = 50
    model_ckpt = '/disk/Log/model/MNIST/MLP/2500-2000-1500-1000-500_norm_epoch=026_val_loss=0.0443-val_acc=0.98682.ckpt'
    order = 7
    load_hermite_parmas = False
    hermite_fit_params_path = '/disk/Log/model/MNIST/MLP/5-MLP_tanh_all_7-order_Hermite_params_global_valdata-fit.pkl'
    is_global = True
    fitting_on_train = False
    act = np.tanh

    # Iris
    # dataset = 'NUMPY'
    # data_dir = '/disk/Dataset/Iris/scaled'
    # dense_nodes = [5]
    # input_size = 4
    # class_num = 3
    # norm = False
    # batch_size = 5
    # model_ckpt = '/disk/Log/torch/Iris/saved_model/MLP/mlp-5-lr0.01-linear.ckpt'
    # order = 2
    # load_hermite_parmas = False
    # hermite_fit_params_path = '/disk/Log/torch/Iris/saved_model/MLP/MLP_tanh_2-order_Hermite_params_global_traindata-fit.pkl'
    # is_global = True
    # fitting_on_train = True
    # act = np.tanh

    # Adult
    # dataset = 'NUMPY'
    # data_dir = '/disk/Dataset/UCI-Adult/normalize/'
    # dense_nodes = [30,20]
    # input_size = 14
    # class_num = 2
    # norm = True
    # batch_size = 100
    # model_ckpt = '/disk/Log/torch/UCI-Adult/normalize-MLP-norm-30-20-tanh-lr0.001-linear-_epoch=247_val_loss=0.3198-val_acc=0.85131.ckpt'
    # order = 7
    # load_hermite_parmas = False
    # hermite_fit_params_path = '/disk/Log/torch/UCI-Adult/MLP_tanh_7-order_Hermite_params_global_traindata-fit.pkl'
    # is_global = True
    # fitting_on_train = True
    # act = np.tanh



    #------------------------------ main ------------------------------#
    train_data = train_dataloader(dataset, data_dir, batch_size, 4)
    val_data = val_dataloader(dataset, data_dir, batch_size, 4)

    model = origin_mlp(dense_nodes, input_size, class_num, norm).to('cuda')
    state_dict = torch.load(model_ckpt)['state_dict']
    model.load_state_dict(state_dict)
    print('model loaded')
    print('origin model performance')

    test(model, 'cuda', val_data, batch_size)

    layer_list = make_act_layer_list(model)

    if load_hermite_parmas:
        hermite_fit_params = load_variavle(hermite_fit_params_path)

    else:
        if fitting_on_train:
            layers_range, global_max, global_min = compute_range(model, layer_list, train_data)
        else:
            layers_range, global_max, global_min = compute_range(model, layer_list, val_data)

        if is_global:
            layers_range = (global_min, global_max)

        hermite_fit_params = compute_hermite_params(layers_range, layer_list, act, np_hermite_poly(order), h_poly_init(order), hermite_fit_params_path, is_global=is_global)

    print('creating {} order hermite fitting MLP...'.format(order))
    fit_model = hermite_tanh_mlp(dense_nodes, input_size, class_num, norm, order=order, params=hermite_fit_params, layer_list=layer_list, is_global=is_global).to('cuda')
    state_dict = torch.load(model_ckpt)['state_dict']
    fit_model.load_state_dict(state_dict)
    print('model loaded')

    test(fit_model, 'cuda', val_data, batch_size)


if __name__ == '__main__':
    main()