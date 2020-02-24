import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
from skimage import feature
from skimage.color import rgb2gray
from matplotlib import pyplot as plt


def compute_cnn_out(input_size, kernel_size, padding=(0,0), dilation=(1,1), stride=(1,1), pooling=(2,2)):
    h_out = ((input_size[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)//stride[0] + 1)//pooling[0]
    w_out = ((input_size[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)//stride[1] + 1)//pooling[1]
    return (h_out, w_out)

def compute_5MODE_dim(input_dim):
    temp = 0
    for i in range(1,5):
        temp += len(torch.combinations(torch.rand(input_dim), i+1, with_replacement=True))
    temp += input_dim
    return temp

def compute_mode_dim(input_dim, order=None):
    if isinstance(input_dim, (list,tuple)):
        temp = 0
        for i in range(len(input_dim)):
            temp += len(torch.combinations(torch.rand(input_dim[i]), i+2, with_replacement=True))
        return temp
    else:
        return len(torch.combinations(torch.rand(input_dim), order, with_replacement=True))

def Pretrain_Mask(model_path, weight_name='fc.weight', num=35*9):
    '''
    根据预训练模型的权值，产生用于mask的坐标矩阵
    '''
    weight = torch.load(model_path)['state_dict'][weight_name]
    weight = weight.abs().sum(dim=0)/weight.size()[0]
    return torch.topk(weight, num)[1]

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

def draw_weight_distribute(model, input_dim, order, class_num, out_put):
    weight = torch.load(model)['state_dict']['fc.weight']
    labels = ['node{}'.format(i) for i in range(len(weight))]
    poly_item = find_polyitem(dim=input_dim, order=order)
    fig = plt.figure()
    x = range(len(poly_item))
    for i in range(len(weight)):
        w = weight[i].cpu().numpy()
        plt.bar([j+0.2*i for j in x], w, width=0.2, label=labels[i])
    plt.xticks(x, poly_item, rotation=-45, fontsize='small')
    plt.legend()
    plt.savefig(out_put)

def kernel_weight_to_visual_numpy(state_dict, name):
    '''
    输入为4维torch.tensor
    '''
    w = state_dict[name]
    if len(w.shape) == 4:
        w = torch.sum(w, dim=1)
        return w.reshape((w.shape[0],-1)).to('cpu').numpy().T
    elif len(w.shape) == 2:
        return w.to('cpu').numpy()
    else:
        return None

def kernel_heatmap(state_dict, name, save_path='./weight_heatmap/'):
    x = kernel_weight_to_visual_numpy(state_dict, name)
    if type(x) != np.ndarray:
        return
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(50, 8))
    ax.set_title(name)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(x, cmap=cmap, xticklabels=8, yticklabels=False, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    f.savefig(save_path+name+'.png', dpi=300)
    return

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

class pick_edge(object):
    """transform: detect the edge of the image, return 0-1 torch tensor"""
    def __call__(self, pic):
        #handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        temp = torch.zeros(img.shape[:-1])

        if img.shape[-1] == 3:
            temp = torch.tensor(feature.canny(rgb2gray(img), sigma=1.6))
        else:
            temp = torch.tensor(feature.canny(img.reshape(img.shape[:-1]), sigma=1.6))
        temp = torch.where(temp==True, torch.ones(temp.shape), torch.zeros(temp.shape))
        return temp.numpy()

class Pretrain_Select(object):
    '''
    transform: select dims according to the pretrained model's weight
    following the ToTensor transforms
    return 2D tensor of shape: (N, num)
    '''
    def __init__(self, model_path, weight_name='fc.weight', num=35*9):
        weight = torch.load(model_path)['state_dict'][weight_name]
        weight = weight.abs().sum(dim=0)/weight.size()[0]
        self.topk_index = torch.topk(weight, num)[1]
        del weight
        torch.cuda.empty_cache()

    def __call__(self, x):
        x = torch.flatten(x, 1)
        return torch.index_select(x, 1, self.topk_index.to(torch.device('cpu')))
        


if __name__ == "__main__":
    # x = torchvision.datasets.MNIST(root='/disk/Dataset/', train=True, transform=transforms.Compose([transforms.ToTensor(), Pretrain_Select('/disk/Log/torch/model/NoHiddenBase_MNIST/_ckpt_epoch_69.ckpt')]))
    # print(x.__getitem__(1)[0].shape)
    draw_weight_distribute('/disk/Log/torch/model/3-ModeNN_Iris/_ckpt_epoch_15.ckpt', input_dim=4, order=2, class_num=3, out_put='/disk/test.jpg')
