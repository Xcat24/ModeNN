# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from numpy.lib.shape_base import expand_dims
import torch
import numpy as np
import math
import os
from torch import device, dtype
from torch.optim import optimizer
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.pyplot as plt
from myutils.datasets import NumpyDataset, train_dataloader
from sympy import expand
from sympy.abc import x, y


# %%
train_dataset = NumpyDataset(root_dir='/disk/Dataset/XOR/tf_playground_like', train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataset = NumpyDataset(root_dir='/disk/Dataset/XOR/tf_playground_like', train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# %%
def load_model_weight(path):
    model = torch.load(path)
    return model, model['state_dict']


# %%
mlp_model, mlp_w = load_model_weight("/disk/Log/torch/XOR/saved_model/MLP/best-playground_like-outnode_with_Tanh-MLP-5-lr0.01-1.00.ckpt")


# %%
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)[0]
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            res = output.squeeze()*target.squeeze()
            correct += torch.ge(res, 0).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# %%
class xor_mlp(torch.nn.Module):
    def __init__(self):
        super(xor_mlp, self).__init__()
        self.fc = torch.nn.Linear(2, 5)
        self.fc_out = torch.nn.Linear(5, 1, bias=False)

    def forward(self, x):
        fc_out = self.fc(x)
        fc_tanh_out = F.tanh(fc_out)
        output = self.fc_out(fc_tanh_out)
        output = F.tanh(output)
        return output, fc_out, fc_tanh_out

def xor_accuracy(out, y):
    res = out.squeeze()*y.squeeze()
    res = torch.ge(res, 0).sum().item()
    return res / len(y)


# %%
class new_mlp(torch.nn.Module):
    def __init__(self):
        super(new_mlp, self).__init__()
        self.fc = torch.nn.Linear(2, 5)
        self.fc_out = torch.nn.Linear(5, 1, bias=False)

    def expansion_tanh(self,x):
        # x = torch.clamp(x, min=-math.pi/2, max=math.pi/2)
        # x = torch.clamp(x, min=-1.2, max=1.2)
        # x = torch.clamp(x, min=-1, max=1)
        return x - 0.3333333333333333*torch.pow(x,3) + 0.13333333333333333*torch.pow(x,5)# - 0.05396825396825397*torch.pow(x,7) + 0.021869488536155203*torch.pow(x,9)

    def forward(self, x):
        fc_out = self.fc(x)
        fc_sig_out = self.expansion_tanh(fc_out)
        output = self.fc_out(fc_sig_out)
        output = F.tanh(output)
        return output, fc_out, fc_sig_out

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

class ModeNN(nn.Module):
    def __init__(self, order=2):
        super(ModeNN, self).__init__()
        self.mode2 = DescartesExtension(2)
        self.mode3 = DescartesExtension(3)
        self.mode4 = DescartesExtension(4)
        self.mode5 = DescartesExtension(5)
        self.fc = nn.Linear(20,1)

    def forward(self, x):
        x_de_2 = self.mode2(x)
        x_de_3 = self.mode3(x)
        x_de_4 = self.mode4(x)
        x_de_5 = self.mode5(x)
        x_de = torch.cat([x, x_de_2, x_de_3, x_de_4, x_de_5], dim=1)
        out = self.fc(x_de)
        out = F.tanh(out)
        return out

# %%
old_model = xor_mlp()
old_model.fc.weight = torch.nn.parameter.Parameter(mlp_w['fc.weight'])
old_model.fc.bias = torch.nn.parameter.Parameter(mlp_w['fc.bias'])
old_model.fc_out.weight = torch.nn.parameter.Parameter(mlp_w['out_layer.weight'])


# %%
test(old_model, 'cuda', val_loader)


# %%
new_exp_model = new_mlp().to('cuda')
new_exp_model.fc.weight = torch.nn.parameter.Parameter(mlp_w['fc.weight'])
new_exp_model.fc.bias = torch.nn.parameter.Parameter(mlp_w['fc.bias'])
new_exp_model.fc_out.weight = torch.nn.parameter.Parameter(mlp_w['out_layer.weight'])

# %%
test(new_exp_model, 'cuda', val_loader)


# %%
old_model.eval()
count = 0
for data, target in val_loader:
    data, target = data.to('cuda'), target.to('cuda')
    # print('data: ', data) # 0~1的小数
    # print('label: ', target) #0~9的数字
    output, fc1_out, fc1_sig_out = old_model(data)
    new_out, new_fc1_out, new_fc1_sig_out = new_exp_model(data)
    res = new_out.squeeze()*target.squeeze()
    correct = torch.ge(res, 0).sum().item()
    
    if not correct and torch.ge(output.squeeze()*target.squeeze(), 0).sum().item():
        print('======================================')
        print('test index: ', count+1)
        print('--------------------------------------')
        print('old model: {}\n predict label: {}'.format(output.tolist(), output.item()))
        print('--------------------------------------')
        print('new model: {}\n predict label: {}'.format(new_out.tolist(), new_out.item()))
    
        # 不同层输出图形显示
        x_ax = np.arange(1, 6)
        y1 = fc1_sig_out.squeeze().cpu().detach().numpy()
        y2 = new_fc1_sig_out.squeeze().cpu().detach().numpy()
        # fig = plt.figure(figsize=(10, 4), dpi=80)
        # ax = fig.add_subplot(1, 1, 1)

        fig, axs = plt.subplots(2, 1, sharey=False, tight_layout=True)
        axs[0].hist(fc1_out.squeeze().cpu().detach().numpy(), 20)
        axs[1].plot(x_ax, y1)
        axs[1].plot(x_ax, y2)
        axs[1].set_xlabel('fc_sig_out')

        plt.show()
        break

    # 测试次数控制
    count += 1
        
    # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    # correct += pred.eq(target.view_as(pred)).sum().item()


# %%
theta = torch.tensor([-0.3333333333333333, 0.13333333333333333])
w1 = mlp_w['fc.weight'] #(5,2)
b1 = mlp_w['fc.bias'] #(5,)
w2 = mlp_w['out_layer.weight'] #(2,5)


# %%
H = []
for i in range(5):
    co1 = w1[i][0].item()
    co2 = w1[i][1].item()
    co3 = b1[i].item()
    h = (co1*x + co2*y +co3) + theta[0].item()*(co1*x + co2*y +co3)**3 + theta[1].item()*(co1*x + co2*y +co3)**5
    H.append(expand(h))
for _ in range(len(H)):
    print(H[_])

# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = 0.0116386007387144*X**5-0.0497741096342584*X**4*Y+0.0236998633590181*X**4+0.0799482838656374*X**3*Y**2-2.86707984267615*X**3*Y+0.737117484083114*X**3-0.144674149163892*X**2*Y**3-0.309310765738997*X**2*Y**2-1.97434327385155*X**2*Y+1.3491982326054*X**2+0.0131844789547463*X*Y**4-3.00979270293206*X*Y**3+1.69564735611823*X*Y**2-14.0691387607622*X*Y+1.55016254040045*X-0.0198165571104951*Y**5-0.13783284493559*Y**4-0.970652021344202*Y**3+0.589221669224032*Y**2-1.33432002269387*Y+1.40309670473631
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %%
O = []
for j in range(1):
    cos = [w2[j][_] for _ in range(5)]
    temp = 0
    for _ in range(5):
        temp += cos[_]*H[_]
    temp += cos[-1]
    O.append(temp)

for _ in range(len(O)):
    print(O[_])

# %% [markdown]
#PlayGround model expansion

# %%
w = torch.tensor([[+1.55016254040045, -1.33432002269387, +1.3491982326054, -14.0691387607622, +0.589221669224032, +0.737117484083114, -1.97434327385155, +1.69564735611823, -0.970652021344202, +0.0236998633590181, -2.86707984267615, -0.309310765738997, -3.00979270293206, -0.13783284493559, 0.0116386007387144, -0.0497741096342584, +0.0799482838656374, -0.144674149163892, +0.0131844789547463, -0.0198165571104951]])
b = torch.tensor([1.40309670473631])
print(w.shape)
print(b.shape)
# %%
mode = ModeNN(5)
mode.fc.weight = torch.nn.parameter.Parameter(-w)
mode.fc.bias = torch.nn.parameter.Parameter(b)

# %%
mode = mode.to('cuda')
mode.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to('cuda'), target.to('cuda')
        output = mode(data)
        test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
        res = output.squeeze()*target.squeeze()
        correct += torch.ge(res, 0).sum().item()

test_loss /= len(val_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    test_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))

# %%
def de_5order(x):
    #计算二维输入(x,y)的5为扩张结果，结果维度顺序为整体阶数的升阶顺序，统一阶次内未x的降幂顺序
    #即(x,y, x2, xy, y2, x3, x2y, xy2, y3, ...)
    res = torch.zeros((len(x), 20), dtype=torch.float32)
    res[:,:2] = x
    res[:,2] = torch.pow(x[:,0], 2)
    res[:,3] = x[:,0]*x[:,1]
    res[:,4] = torch.pow(x[:,1],2)
    res[:,5] = torch.pow(x[:,0], 3)
    res[:,6] = torch.pow(x[:,0], 2)*x[:,1]
    res[:,7] = x[:,0]*torch.pow(x[:,1],2)
    res[:,8] = torch.pow(x[:,1], 3)
    res[:,9] = torch.pow(x[:,0], 4)
    res[:,10] = torch.pow(x[:,0], 3)*x[:,1]
    res[:,11] = torch.pow(x[:,0], 2)*torch.pow(x[:,1], 2)
    res[:,12] = torch.pow(x[:,0], 1)*torch.pow(x[:,1], 3)
    res[:,13] = torch.pow(x[:,1], 4)
    res[:,14] = torch.pow(x[:,0], 5)
    res[:,15] = torch.pow(x[:,0], 4)*torch.pow(x[:,1], 1)
    res[:,16] = torch.pow(x[:,0], 3)*torch.pow(x[:,1], 2)
    res[:,17] = torch.pow(x[:,0], 2)*torch.pow(x[:,1], 3)
    res[:,18] = torch.pow(x[:,0], 1)*torch.pow(x[:,1], 4)
    res[:,19] = torch.pow(x[:,1], 5)
    return res

# %%
# 笨方法算展开后的MODENN识别率
w = w.to('cuda')
b = b.to('cuda')
for data, target in val_loader:
    data, target = data.to('cuda'), target.to('cuda')
    output = de_5order(data).to('cuda')
    output = torch.mm(output, w.T)
    test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
    res = output.squeeze()*target.squeeze()
    correct += torch.ge(res, 0).sum().item()

test_loss /= len(val_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    test_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))


# %%
theta = torch.tensor([-0.3333333333333333, 0.13333333333333333])
play_ground_w1 = torch.tensor([[0.34, -0.48],
                    [0.54, 0.62],
                    [-0.54, -0.54],
                    [0.57, -0.56],
                    [-0.49, 0.45]])
playground_b1 = torch.tensor([-0.42, 1.4, 1.4, 1.6, 1.5]) #(5,)
playground_w2 = torch.tensor([-0.52, -2.0, -2.3, 2.2, 2.0]) #(1,5)


# %%
H = []
for i in range(5):
    co1 = w1[i][0].item()
    co2 = w1[i][1].item()
    co3 = b1[i].item()
    h = (co1*x + co2*y +co3) + theta[0].item()*(co1*x + co2*y +co3)**3 + theta[1].item()*(co1*x + co2*y +co3)**5
    H.append(expand(h))
for _ in range(len(H)):
    print(H[_])

# %%
cos = [w2[_] for _ in range(5)]
O = 0
for _ in range(5):
    O += cos[_]*H[_]
print(O)


# %%
model = xor_mlp()
device = 'cuda'
optimizer = torch.optim.SGD(model.parameters(),lr=0.03)
model.to(device)
model.train()
for epoch in range(200):
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data/10
        optimizer.zero_grad()
        output = model(data)[0]
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += xor_accuracy(output, target)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print('Train Epoch: {} \tLoss: {:.6f} \tTrain_Acc: {:.4f}'.format(epoch, train_loss, train_acc))


# %%
model = xor_mlp()
model.fc.weight = torch.nn.parameter.Parameter(torch.tensor([[0.34, -0.48],
                    [0.54, 0.62],
                    [-0.54, -0.54],
                    [0.57, -0.56],
                    [-0.49, 0.45]]))
model.fc.bias = torch.nn.parameter.Parameter(torch.tensor([-0.42, 1.4, 1.4, 1.6, 1.5]))
model.fc_out.weight = torch.nn.parameter.Parameter(torch.tensor([-0.52, -2.0, -2.3, 2.2, 2.0]))

# %%
model = new_mlp()
model.fc.weight = torch.nn.parameter.Parameter(torch.tensor([[0.34, -0.48],
                    [0.54, 0.62],
                    [-0.54, -0.54],
                    [0.57, -0.56],
                    [-0.49, 0.45]]))
model.fc.bias = torch.nn.parameter.Parameter(torch.tensor([-0.42, 1.4, 1.4, 1.6, 1.5]))
model.fc_out.weight = torch.nn.parameter.Parameter(torch.tensor([-0.52, -2.0, -2.3, 2.2, 2.0]))

# %%
model.to('cuda')
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to('cuda'), target.to('cuda')
        output = model(data)[0]
        test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
        res = output.squeeze()*target.squeeze()
        correct += torch.ge(res, 0).sum().item()

test_loss /= len(val_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    test_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))


# %%
