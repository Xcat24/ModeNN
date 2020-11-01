import torch
import numpy as np
import math
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

def load_model_weight(path):
    model = torch.load(path)
    return model, model['state_dict']

class old_mlp(torch.nn.Module):
    def __init__(self):
        super(old_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(784, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        self.fc_out = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        output = self.fc_out(x)
        # output = F.log_softmax(x, dim=1)
        return output

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
        x = self.fc1(x)
        x = self.expansion_sigmoid(x)
        x = self.fc2(x)
        x = self.expansion_sigmoid(x)
        output = self.fc_out(x)
        # output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    #data
    dataset = torchvision.datasets.MNIST(root='/mydata/xcat/Data/MNIST', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    #load model and weight
    model, w = load_model_weight('/mydata/xcat/Log/MNIST/MLP/500-500/epoch149_97.4.ckpt')

    old_model = old_mlp()
    old_model.fc1.weight = torch.nn.parameter.Parameter(w['fc.0.weight'])
    old_model.fc1.bias = torch.nn.parameter.Parameter(w['fc.0.bias'])
    old_model.fc2.weight = torch.nn.parameter.Parameter(w['fc.2.weight'])
    old_model.fc2.bias = torch.nn.parameter.Parameter(w['fc.2.bias'])
    old_model.fc_out.weight = torch.nn.parameter.Parameter(w['out_layer.weight'])
    old_model.fc_out.bias = torch.nn.parameter.Parameter(w['out_layer.bias'])

    new_model = mlp()
    new_model.fc1.weight = torch.nn.parameter.Parameter(w['fc.0.weight'])
    new_model.fc1.bias = torch.nn.parameter.Parameter(w['fc.0.bias'])
    new_model.fc2.weight = torch.nn.parameter.Parameter(w['fc.2.weight'])
    new_model.fc2.bias = torch.nn.parameter.Parameter(w['fc.2.bias'])
    new_model.fc_out.weight = torch.nn.parameter.Parameter(w['out_layer.weight'])
    new_model.fc_out.bias = torch.nn.parameter.Parameter(w['out_layer.bias'])

    old_model.eval()
    count = 0
    for data, target in data_loader:
        data, target = data.to('cuda'), target.to('cuda')
        # print('data: ', data) # 0~1的小数
        # print('label: ', target) #0~9的数字
        output = old_model(data)
        new_out = new_model(data)
        print('======================================')
        print('test index: ', count+1)
        print('old model: {}\n predict label: {}'.format(output.tolist(), output.argmax(dim=1,keepdim=True).item()))
        print('new model: {}\n predict label: {}'.format(new_out.tolist(), new_out.argmax(dim=1, keepdim=True).item()))
        count += 1
        if count == 10:
            break