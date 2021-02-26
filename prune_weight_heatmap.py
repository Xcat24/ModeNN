import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from layer import DescartesExtension
from mymodels import ModeNN
from myutils.datasets import train_dataloader, val_dataloader, test_dataloader
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

class ModeNN(nn.Module):
    def __init__(self, order=2):
        super(ModeNN, self).__init__()
        self.mode2 = DescartesExtension(2)
        self.norm_layer = nn.BatchNorm1d(308504)
        self.fc = nn.Linear(308504,10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x_de_2 = self.mode2(x)
        x_de = torch.cat([x, x_de_2], dim=1)
        x_de = self.norm_layer(x_de)
        out = self.fc(x_de)
        return out

def all_idx(dim, order):
    result = [ _ for _ in range(dim)]
    for i in range(1, order):
        temp = torch.combinations(torch.arange(0,dim), i+1, with_replacement=True)
        for j in temp:
            result.append(j.tolist())
    return result

test_data = test_dataloader('MNIST', "/disk/Dataset/MNIST", 10, 4, random_sample=False)
model = ModeNN().to('cuda')
state_dict = torch.load("/disk/Log/torch/MNIST/ModeNN/best_98.35.ckpt")['state_dict']

model.load_state_dict(state_dict)
prune.l1_unstructured(model.fc, name="weight", amount=0.9810)
w_prune9810 = model.fc.weight
print(w_prune9810)

idxs = all_idx(784, 2)

sns.set(style="white")
for i in range(10):
    firstorder_w_heatmap = np.zeros(784)
    for j in range(784):
        firstorder_w_heatmap[j] += w_prune9810[i,j]
    sns.heatmap(data=firstorder_w_heatmap.reshape((28,28)), square=True, vmin=-0.25, vmax=0.25, robust=True, cmap=cm.coolwarm, xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig('/home/xcat/1order_{}.pdf'.format(i))

for i in range(10):
    SecOrder_w_heatmap = np.zeros((784,784))
    for j in range(784, 308504):
        x = idxs[j][0]
        y = idxs[j][1]
        if x==y:
            SecOrder_w_heatmap[x,x] += w_prune9810[i,j]
        else:
            SecOrder_w_heatmap[x,y] += w_prune9810[i,j]
            SecOrder_w_heatmap[y,x] += w_prune9810[i,j]
    sns.heatmap(data=SecOrder_w_heatmap, square=True, vmin=-0.02, vmax=0.02, robust=True, cmap=cm.coolwarm, xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig('/home/xcat/2order_{}.pdf'.format(i))