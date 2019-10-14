import torch
import torch.nn.functional as F
from torch import nn


class DescartesExtension(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.training = False
        self.order = order

    def forward(self, x):
        #empty gpu cache
        torch.cuda.empty_cache()
        if x.dim() == 1:
            return torch.prod(torch.combinations(x, self.order, with_replacement=True), dim=1)
        elif x.dim() == 2:
            return torch.stack([torch.prod(torch.combinations(a, self.order, with_replacement=True), dim=1) for a in x])
        else:
            raise ValueError("the dimension of input tensor is expected 1 or 2")

class LocalDE(nn.Module):
    def __init__(self, order=2, kernel_size=(3,3)):
        super(LocalDE, self).__init__()
        self.training = False
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.de = DescartesExtension(order=order)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("the dimension of input tensor is expected 4")
        x = self.unfold(x).transpose(1,2)
        out = x.reshape(-1, x.size(-1))
        out = self.de(out)
        out = out.view(x.size(0), -1)
        return out

class MaskDE(nn.Module):
    def __init__(self, mask, order=2):
        super(MaskDE, self).__init__()
        self.training = False
        self.order = order
        self.mask = mask
        self.de = [DescartesExtension(order=i) for i in range(2, order+1)]

    def forward(self, x):
        if x.dim() == 1:
            x = torch.masked_select(x, self.mask)
            de = torch.cat([self.de[_](x) for _ in range(len(self.de))], dim=-1)
            out = torch.cat([x, de], dim=-1)
            return out
        elif x.dim() == 2:
            masked_dim = self.mask.sum()
            mask = self.mask.repeat(x.shape[0],1)
            x = torch.masked_select(x, mask).reshape((x.shape[0], masked_dim))
            de = torch.cat([self.de[_](x) for _ in range(len(self.de))], dim=-1)
            out = torch.cat([x, de], dim=-1)
            return out
        else:
            raise ValueError("the dimension of input tensor is expected 1 or 2")


class SLConv(nn.Module):
    def __init__(self, in_channel, stride=1, padding=1):
        super(SLConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.sobel_weight_h = torch.tensor([[-1., -2. , -1.], [0., 0., 0.], [1., 2. , 1.]], requires_grad=False)
        self.sobel_weight_w = self.sobel_weight_h.t()
        self.sobel_weight_h = self.sobel_weight_h.expand(self.in_channel, 3, 3)
        self.sobel_weight_w = self.sobel_weight_w.expand(self.in_channel, 3, 3)
        self.laplace_weight = torch.tensor([[0., 1., 0.], [1., -4., 1.],[0. ,1., 0.]], requires_grad=False).expand(self.in_channel, 3, 3)
        self.kernel = torch.stack([self.sobel_weight_h, self.sobel_weight_w, self.laplace_weight], dim=0)
                
    def forward(self, x):
        out = F.conv2d(x, self.kernel.to(x.device), stride=self.stride, padding=self.padding)
        return out
        