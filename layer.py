import torch
import torch.nn.functional as F
from torch import nn
from myutils.utils import compute_mode_dim


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

class RandomDE(nn.Module):
    def __init__(self, order=[2,], input_dim=784, output_dim=[64,]):
        super().__init__()
        self.training = False
        self.order = order
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.idx = [torch.randint(input_dim, (output_dim[_], order[_])) for _ in range(len(order))]
        # print(self.idx)

    def compute_order(self, x, order, input_dim, output_dim, idx):
        if x.dim() == 1:
            if torch.cuda.is_available():
                idx = idx.cuda()
            select_terms = torch.stack([torch.prod(torch.index_select(x,dim=0,index=idx[i])) for i in range(output_dim)])
            return select_terms
        elif x.dim() == 2:
            if torch.cuda.is_available():
                idx = idx.cuda()
            select_terms = torch.stack([torch.prod(torch.index_select(x,dim=1,index=idx[i]), dim=1) for i in range(output_dim)], dim=1)
            return select_terms
        else:
            raise ValueError("the dimension of input tensor is expected 1 or 2")
    
    def forward(self, x):
        de_out = [self.compute_order(x, self.order[i], self.input_dim, self.output_dim[i], self.idx[i]) for i in range(len(self.output_dim))]
        return torch.cat(de_out, dim=-1)

class DE_Conv(nn.Module):
    ##TODO gradient overflow bug to be fixed
    def __init__(self, order=2, input_size=(28,28), kernel_size=(3,3), in_channel=1, out_channel=16):
        super(DE_Conv, self).__init__()
        self.in_channel = in_channel
        self.kernel_dim = kernel_size[0]*kernel_size[1]
        de_dim = compute_mode_dim(torch.prod(torch.tensor(kernel_size)).item(), order)
        # self.weights = nn.Parameter(torch.randn(out_channel, de_dim))
        self.weights = nn.Parameter(torch.randn(out_channel, in_channel*de_dim), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.fold = nn.Fold((input_size[0]-kernel_size[0]+1, input_size[1]-kernel_size[1]+1), (1,1))

        self.de = DescartesExtension(order=order)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("the dimension of input tensor is expected 4")
        x = self.unfold(x).transpose(1,2).view((x.shape[0], -1, self.in_channel, self.kernel_dim))
        out = x.reshape(-1, x.size(-1))
        out = self.de(out)
        out = out.view(x.size(0), x.size(1), -1)
        out = out.matmul(self.weights.t()).transpose(1,2)
        out = self.fold(out)

        # out = x.matmul(self.weights.t()).transpose(1,2)
        return out

class Fast2Order_DE_Conv(nn.Module):
    def __init__(self, input_size=(28,28), kernel_size=(3,3), in_channel=1, out_channel=16):
        super(Fast2Order_DE_Conv, self).__init__()
        self.in_channel = in_channel
        self.kernel_dim = kernel_size[0]*kernel_size[1]
        de_dim = self.kernel_dim*self.kernel_dim
        # self.weights = nn.Parameter(torch.randn(out_channel, de_dim))
        self.weights = nn.Parameter(torch.randn(out_channel, in_channel*de_dim), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.fold = nn.Fold((input_size[0]-kernel_size[0]+1, input_size[1]-kernel_size[1]+1), (1,1))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("the dimension of input tensor is expected 4")
        x = self.unfold(x).transpose(1,2).view((x.shape[0], -1, self.in_channel, self.kernel_dim))
        out = x.reshape(-1, x.size(-1))
        out = torch.matmul(out.unsqueeze(-1),out.unsqueeze(-2))
        out = out.view(x.size(0), x.size(1), -1)
        out = out.matmul(self.weights.t()).transpose(1,2)
        out = self.fold(out)

        # out = x.matmul(self.weights.t()).transpose(1,2)
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

class Pretrain_5MODE(nn.Module):
    def __init__(self, num_classes, bins_size=9, bins_num=35):
        super(Pretrain_5MODE, self).__init__()
        self.bins_size = bins_size
        self.bins_num = bins_num
        self.de2 = DescartesExtension(order=2)
        self.de3 = DescartesExtension(order=3)
        self.de4 = DescartesExtension(order=4)
        self.de5 = DescartesExtension(order=5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        temp = []
        for i in range(self.bins_num):
            origin = x[:,i*self.bins_size:(i+1)*self.bins_size]
            de2_out = self.de2(origin)
            de3_out = self.de3(origin)
            de4_out = self.de4(origin)
            de5_out = self.de5(origin)
            temp.append(torch.cat([origin, de2_out, de3_out, de4_out, de5_out], dim=-1))

        out = torch.cat(temp, dim=-1)
      
        return out

class Mode(nn.Module):
    '''
    对数据进行多阶笛卡尔扩张，用order_dim数组来控制扩张的阶数以及每个阶数扩张的维度，order_dim[i]表示用于阶数为i+2的升阶操作的输入维度
    当对应阶数扩张的维度小于总维度时，则是按输入维度从前往后截取（因此，输入的维度最好为按重要顺序排序的）。
    输入：(N, feature_dim)
    输出：(N, MODE_dim) 其中MODE_dim为各阶扩张结果的拼接
    '''
    def __init__(self, order_dim=[300, 50, 20, 10]):
        super(Mode, self).__init__()
        self.order_dim = order_dim
        self.de = [DescartesExtension(order=i+2) for i in range(len(order_dim))]

    def forward(self, x):
        de_out = [self.de[_](x[:,:self.order_dim[_]]) for _ in range(len(self.de))]
        return torch.cat(de_out, dim=-1)

class MaskLayer(nn.Module):
    '''
    根据mask矩阵（坐标索引，如torch.topk()[1]得到的矩阵），过滤输入，只通过mask中值为True的类型
    输出：（N，feature）
    '''
    def __init__(self,mask):
        super(MaskLayer, self).__init__()
        self.mask = mask
    
    def forward(self, x):
        return torch.index_select(x, 1, self.mask.to(x.device))

class BreakupConv(nn.Module):
    def __init__(self, input_size=(28,28), kernel_size=(3,3), out_channel=16, in_channel=3):
        super(BreakupConv, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size[0]*kernel_size[1]), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.fold = nn.Fold(output_size=(input_size[0]-kernel_size[0]+1, input_size[1]-kernel_size[1]+1), kernel_size=(1,1))

    def forward(self, x):
        inp_unf = self.unfold(x)
        out_unf = inp_unf.transpose(1, 2).matmul(self.weights.view(self.weights.size(0), -1).t()).transpose(1, 2)
        out = self.fold(out_unf)
        return out

if __name__ == "__main__":
    import torchvision
    import torch
    from torch.nn import functional as F

    w = torch.rand((16,3,3,3))
    b = torch.zeros((16,))
    x = torch.rand(2, 16, 13, 13)

    #BreakupConv  Test
    # conv_out = F.conv2d(x,weight=w,bias=b,stride=(1,1))
    # print(conv_out)

    # inp_unf = F.unfold(x,3) #shape: [2,27,676]
    # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2) #shape: [2,16,676]
    # out = F.fold(out_unf,(26,26), (1,1)) #shape: [2,16,26,26]

    # if torch.equal(out, conv_out):
    #     print('it is equal')

    # b_conv = BreakupConv()
    # out = b_conv(x)
    # print(out)

    #DE_Conv Test
    # conv_de = DE_Conv(order=2, input_size=(13,13), kernel_size=(3,3), in_channel=16, out_channel=64)
    # out = conv_de(x)
    # out = F.max_pool2d(out,2)

    #Fast2Order_DE_Conv Test
    conv_de = Fast2Order_DE_Conv(order=2, input_size=(13,13), kernel_size=(3,3), in_channel=16, out_channel=64)
    out = conv_de(x)

    x = torch.arange(32).reshape((2,4,4)).float()
    y = torch.rand((2,1,4,4))
    print(x)
    print(y)
    # m = LocalDE(order=2, kernel_size=(3,3), out_channel=16)
    m = DescartesExtension(2)
    output = m(x)
    output[0,0,0].backward()
    m.weights.grad()
    print(output)
    print(m(y))