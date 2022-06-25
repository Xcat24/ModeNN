import math
import numpy as np
import torch
import pickle

def save_variable(v, filename):
    f = open(filename,"wb")
    pickle.dump (v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename,"rb")
    r = pickle.load(f)
    f.close()
    return r

def compute_mode_dim(input_dim, order=None):
    if isinstance(input_dim, (list,tuple)):
        temp = 0
        for i in range(len(input_dim)):
            temp += (math.factorial(input_dim[i] + (i+2) - 1)) // (math.factorial(input_dim[i] - 1) * math.factorial(i+2))
        return temp
    else:
        return (math.factorial(input_dim + order - 1)) // (math.factorial(input_dim - 1) * math.factorial(order))

def find_polyterm(dim, order):
    '''
    input : 
        dim  : the dimension of the input data(int/list)
        order: the DE order(int)
    return: 
        A list containing all the algebraic expression for the corresponding polynomial terms.
        Index begins from 0.
    '''
    if isinstance(dim, list):
        dim = np.product(dim)
    result = ['x{}'.format(_) for _ in range(dim)]
    for i in range(1, order):
        temp = torch.combinations(torch.arange(0,dim), i+1, with_replacement=True)
        for j in range(len(temp)):
            item = ''
            for k in range(len(temp[j])):
                item += 'x{}'.format(temp[j,k])
            result.append(item)
    return result

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    Outputs:
        torch.Tensor representing the patch sequece of shape [B, Patch_length, Patch_dim]
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x