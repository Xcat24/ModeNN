import torch
import torch.nn.functional as F
from torch import nn

class DescartesExtension(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.training = False
        self.order = order

    def forward(self, x):
        if x.dim() == 1:
            return torch.prod(torch.combinations(x, self.order, with_replacement=True), dim=1)
        elif x.dim() == 2:
            return torch.stack([torch.prod(torch.combinations(a, self.order, with_replacement=True), dim=1) for a in x])
        elif x.dim() == 3:
            batch, kernel, _ = x.shape
            x = x.view(-1, x.shape[-1])
            tmp = torch.stack([torch.prod(torch.combinations(a, self.order, with_replacement=True), dim=1) for a in x])
            return tmp.view((batch, -1))
        else:
            raise ValueError("the dimension of input tensor is expected 1, 2 or 3")

class Mode(nn.Module):
    def __init__(self, order=2):
        super(Mode, self).__init__()
        self.de = [DescartesExtension(order=i+2) for i in range(order-1)]

    def forward(self, x):
        de_out = [self.de[_](x) for _ in range(len(self.de))]
        return torch.cat(de_out, dim=-1)

class Select_Mode(nn.Module):
    '''
    对数据进行多阶笛卡尔扩张，用order_dim数组来控制扩张的阶数以及每个阶数扩张的维度，order_dim[i]表示用于阶数为i+2的升阶操作的输入维度
    当对应阶数扩张的维度小于总维度时，则是按输入维度从前往后截取（因此，输入的维度最好为按重要顺序排序的）。
    输入：(N, feature_dim)
    输出：(N, MODE_dim) 其中MODE_dim为各阶扩张结果的拼接
    '''
    def __init__(self, order_dim=[300, 50, 20, 10]):
        super(Select_Mode, self).__init__()
        self.order_dim = order_dim
        self.de = [DescartesExtension(order=i+2) for i in range(len(order_dim))]

    def forward(self, x):
        de_out = [self.de[_](x[:,:self.order_dim[_]]) for _ in range(len(self.de))]
        return torch.cat(de_out, dim=-1)

class Outer_prod(nn.Module):
    def __init__(self, dim):
        super(Outer_prod, self).__init__()
        self.training = False

    def forward(self, x):
        if x.dim() == 1:
            return torch.outer(x,x)
        elif x.dim() == 2:
            return torch.tensor([torch.outer[x[_], x[_]] for _ in range(len(x))])
        else:
            raise ValueError("the dimension of input tensor is expected 1, 2 or 3")