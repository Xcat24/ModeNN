# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import numpy as np
import math
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# %%
def taylor_series_co(fn, order):
    x = torch.zeros(1, requires_grad=True)
    res = []
    temp = fn(x)
    res.append(temp.item())
    for i in range(order):
        temp = torch.autograd.grad(temp, x, create_graph=True)[0]
        res.append(temp.item()/math.factorial(i+1))
    return res

def Cnm(n,m):
    return math.factorial(n) / (math.factorial(n-m)*math.factorial(m))

def combination_co(idx, order, dim):
    res = 1
    for i in range(dim):
        cnt = idx.count(i)
        res *= Cnm(dim, cnt)
        dim -= cnt
        if dim <= 0:
            break
    return res

def compute_de_co(taylor_co, w, order):
    b = taylor_co[0]
    de_w = []
    for i in range(order):
        idx = torch.combinations(torch.arange(len(w)), i + 1, with_replacement=True).tolist()
#         print(idx)
        for j in range(len(idx)):
            temp = taylor_co[i+1]*combination_co(idx[j], i+1, len(w))*w.index_select(0, torch.tensor(idx[j])).prod()
#             print(temp)
            de_w.append(temp)
    return torch.tensor(de_w)


# %%
w = torch.randn((10,))
input_x = torch.randn((10,))


# %%
net = torch.matmul(input_x, w.t())
out = torch.sigmoid(net)


# %%
r = 3
de_w = compute_de_co(taylor_series_co(torch.sigmoid, r), w, r)
de_x = torch.cat([torch.prod(torch.combinations(input_x, _+1, with_replacement=True), dim=1) for _ in range(r)])
delta = torch.matmul(de_x, de_w.t()) - out
print(delta)


# %%
r = 5
de_w = compute_de_co(taylor_series_co(torch.sigmoid, r), w, r)
de_x = torch.cat([torch.prod(torch.combinations(input_x, _+1, with_replacement=True), dim=1) for _ in range(r)])
delta = torch.matmul(de_x, de_w.t()) - out
print(delta)


# %%
r = 6
de_w = compute_de_co(taylor_series_co(torch.sigmoid, r), w, r)
de_x = torch.cat([torch.prod(torch.combinations(input_x, _+1, with_replacement=True), dim=1) for _ in range(r)])
delta = torch.matmul(de_x, de_w.t()) - out
print(delta)


# %%
r = 7
de_w = compute_de_co(taylor_series_co(torch.sigmoid, r), w, r)
de_x = torch.cat([torch.prod(torch.combinations(input_x, _+1, with_replacement=True), dim=1) for _ in range(r)])
delta = torch.matmul(de_x, de_w.t()) - out
print(delta)


# %%
r = 9
de_w = compute_de_co(taylor_series_co(torch.sigmoid, r), w, r)
de_x = torch.cat([torch.prod(torch.combinations(input_x, _+1, with_replacement=True), dim=1) for _ in range(r)])
delta = torch.matmul(de_x, de_w.t()) - out
print(delta)

# %% [markdown]
# # softplus expansion

# %%
def softplus(x):
    return torch.log(1 + torch.exp(x))


# %%
taylor_series_co(softplus, 10)

# %% [markdown]
# # Load Model & Replace Activation Function
# %% [markdown]
# ## mnist data loader

# %%
dataset = torchvision.datasets.MNIST(root='/mydata/xcat/Data/MNIST', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


# %%
def load_model_weight(path):
    model = torch.load(path)
    return model, model['state_dict']


# %%
model, w = load_model_weight('/mydata/xcat/Log/MNIST/MLP/500-500/epoch149_97.4.ckpt')


# %%
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)[0]
        loss = F.nll_loss(output, target)
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# %%
class old_mlp(torch.nn.Module):
    def __init__(self):
        super(old_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(784, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        self.fc_out = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        fc1_out = self.fc1(x)
        fc1_sig_out = F.sigmoid(fc1_out)
        # print('fc1 sig out: ', fc1_sig_out)
        fc2_out = self.fc2(fc1_sig_out)
        # print('fc2 out: ', fc2_out)
        fc2_sig_out = F.sigmoid(fc2_out)
        # print('fc2 sig out: ', fc2_sig_out)
        output = self.fc_out(fc2_sig_out)
        # output = F.log_softmax(x, dim=1)
        return output, fc1_out, fc1_sig_out, fc2_out, fc2_sig_out

class mlp(torch.nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = torch.nn.Linear(784, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        self.fc_out = torch.nn.Linear(500, 10)
    def expansion_sigmoid(self, x):
        return 0.5 + (1/4)*x - (1/48)*torch.pow(x,3) + (1/480)*torch.pow(x,5) - (17/80640)*torch.pow(x,7)

    def forward(self, x):
        x = torch.flatten(x, 1)
        fc1_out = self.fc1(x)
        fc1_sig_out = self.expansion_sigmoid(fc1_out)
        # print('fc1 sig expansion out: ', fc1_sig_out)
        fc2_out = self.fc2(fc1_sig_out)
        # print('fc2 out: ', fc2_out)
        fc2_sig_out = self.expansion_sigmoid(fc2_out)
        # print('fc2 sig expansion out: ', fc2_sig_out)
        output = self.fc_out(fc2_sig_out)
        # output = F.log_softmax(x, dim=1)
        return output, fc1_out, fc1_sig_out, fc2_out, fc2_sig_out


# %%
class new_mlp(torch.nn.Module):
    def __init__(self):
        super(new_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(784, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        self.fc_out = torch.nn.Linear(500, 10)
    def expansion_sigmoid(self, x):
        temp = 0.5 + (1/4)*x - (1/48)*torch.pow(x,3) + (1/480)*torch.pow(x,5) - (17/80640)*torch.pow(x,7)
        temp = torch.clamp(temp, min=-1, max=1)
        return temp 

    def forward(self, x):
        x = torch.flatten(x, 1)
        fc1_out = self.fc1(x)
        fc1_sig_out = self.expansion_sigmoid(fc1_out)
        # print('fc1 sig expansion out: ', fc1_sig_out)
        fc2_out = self.fc2(fc1_sig_out)
        # print('fc2 out: ', fc2_out)
        fc2_sig_out = self.expansion_sigmoid(fc2_out)
        # print('fc2 sig expansion out: ', fc2_sig_out)
        output = self.fc_out(fc2_sig_out)
        # output = F.log_softmax(x, dim=1)
        return output, fc1_out, fc1_sig_out, fc2_out, fc2_sig_out

# %%
old_model = old_mlp()
old_model.fc1.weight = torch.nn.parameter.Parameter(w['fc.0.weight'])
old_model.fc1.bias = torch.nn.parameter.Parameter(w['fc.0.bias'])
old_model.fc2.weight = torch.nn.parameter.Parameter(w['fc.2.weight'])
old_model.fc2.bias = torch.nn.parameter.Parameter(w['fc.2.bias'])
old_model.fc_out.weight = torch.nn.parameter.Parameter(w['out_layer.weight'])
old_model.fc_out.bias = torch.nn.parameter.Parameter(w['out_layer.bias'])


# %%
test(old_model, 'cuda', data_loader)


# %%
new_model = mlp()
new_model.fc1.weight = torch.nn.parameter.Parameter(w['fc.0.weight'])
new_model.fc1.bias = torch.nn.parameter.Parameter(w['fc.0.bias'])
new_model.fc2.weight = torch.nn.parameter.Parameter(w['fc.2.weight'])
new_model.fc2.bias = torch.nn.parameter.Parameter(w['fc.2.bias'])
new_model.fc_out.weight = torch.nn.parameter.Parameter(w['out_layer.weight'])
new_model.fc_out.bias = torch.nn.parameter.Parameter(w['out_layer.bias'])

# %%
new_exp_model = new_mlp()
new_exp_model.fc1.weight = torch.nn.parameter.Parameter(w['fc.0.weight'])
new_exp_model.fc1.bias = torch.nn.parameter.Parameter(w['fc.0.bias'])
new_exp_model.fc2.weight = torch.nn.parameter.Parameter(w['fc.2.weight'])
new_exp_model.fc2.bias = torch.nn.parameter.Parameter(w['fc.2.bias'])
new_exp_model.fc_out.weight = torch.nn.parameter.Parameter(w['out_layer.weight'])
new_exp_model.fc_out.bias = torch.nn.parameter.Parameter(w['out_layer.bias'])

# %%
test(new_exp_model, 'cuda', data_loader)


# %%
old_model.eval()
count = 0
for data, target in data_loader:
    data, target = data.to('cuda'), target.to('cuda')
    # print('data: ', data) # 0~1的小数
    # print('label: ', target) #0~9的数字
    print('======================================')
    print('test index: ', count+1)
    print('--------------------------------------')
    output, fc1_out, fc1_sig_out, fc2_out, fc2_sig_out = old_model(data)
    print('old model: {}\n predict label: {}'.format(output.tolist(), output.argmax(dim=1, keepdim=True).item()))
    print('--------------------------------------')
    new_out, new_fc1_out, new_fc1_sig_out, new_fc2_out, new_fc2_sig_out = new_exp_model(data)
    print('new model: {}\n predict label: {}'.format(new_out.tolist(), new_out.argmax(dim=1, keepdim=True).item()))
    
    # 不同层输出图形显示
    x_ax = np.arange(1, 501)
    y1 = fc1_sig_out.squeeze().cpu().detach().numpy()
    y2 = new_fc1_sig_out.squeeze().cpu().detach().numpy()
    y3 = fc2_out.squeeze().cpu().detach().numpy()
    y4 = new_fc2_out.squeeze().cpu().detach().numpy()
    y5 = fc2_sig_out.squeeze().cpu().detach().numpy()
    y6 = new_fc2_sig_out.squeeze().cpu().detach().numpy()
    # fig = plt.figure(figsize=(10, 4), dpi=80)
    # ax = fig.add_subplot(1, 1, 1)

    fig, axs = plt.subplots(2, 2, sharey=False, tight_layout=True)
    axs[0][0].hist(fc1_out.squeeze().cpu().detach().numpy(), 20)
    axs[0][1].plot(x_ax, y1)
    axs[0][1].plot(x_ax, y2)
    axs[0][1].set_xlabel('fc1_sig_out')
    axs[1][0].plot(x_ax, y3)
    axs[1][0].plot(x_ax, y4)
    axs[1][0].set_xlabel('fc2_net')
    axs[1][1].plot(x_ax, y5)
    axs[1][1].plot(x_ax, y6)
    axs[1][1].set_xlabel('fc2_sig_out')
    plt.show()

    # 测试次数控制
    count += 1
    if count == 2:
        break
    # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    # correct += pred.eq(target.view_as(pred)).sum().item()


# %%



