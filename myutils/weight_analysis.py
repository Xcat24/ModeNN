import torch
from utils import data_statics

def load_model_weight(path, weight_name):
    model = torch.load(path)
    return model['state_dict'][weight_name]

if __name__ == '__main__':
    path = '/home/xucong/Log/MNIST/ModeNN/2order/best_98.35.ckpt'
    weight_name = 'fc.weight'
    data_statics('2order-weight', load_model_weight(path, weight_name), verbose=True)
    