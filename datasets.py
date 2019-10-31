import torch
import torchvision
import torchvision.transforms as transforms
import layer
from myutils.datasets import ORLdataset, NumpyDataset
from sklearn import datasets
import numpy as np
import math

#produce circle_square numpy data
def gen_point():
    bound = math.sqrt(math.pi/8)
    x = np.random.uniform(-bound, bound)
    y = np.random.uniform(-bound, bound)
    if math.sqrt(x**2 + y**2) > 0.5:
        label = np.array([0,])
    else:
        label = np.array([1,])
    
    return np.array([[x,y]]), label

train_data, train_label = gen_point()
for i in range(999):
    data, label = gen_point()
    train_data = np.vstack([train_data, data])
    train_label = np.concatenate([train_label, label])
print(np.sum(train_label))

val_data, val_label = gen_point()
for i in range(399):
    data, label = gen_point()
    val_data = np.vstack([val_data, data])
    val_label = np.concatenate([val_label, label])
print(np.sum(val_label))

np.save('/disk/Dataset/circle_square/pytorch_numpy_data/train_data.npy', train_data)
np.save('/disk/Dataset/circle_square/pytorch_numpy_data/val_data.npy', val_data)
np.save('/disk/Dataset/circle_square/pytorch_numpy_data/train_label.npy', train_label)
np.save('/disk/Dataset/circle_square/pytorch_numpy_data/val_label.npy', val_label)

#produce T-C numpy data
def gen_pixel():
    return 0.25*np.random.random() + 0.25

def produce_t():
    t_1 = np.array([[1., 1., 1.],[-1., 1., -1.],[-1., 1., -1.]])
    t_2 = np.array([[-1., 1., -1.],[-1., 1., -1.],[1., 1., 1.]])
    for t in [t_1, t_2, t_1.T, t_2.T]:
        for i in range(9):
            t[i//3][i%3] = gen_pixel()*t[i//3][i%3]
    return np.array([t_1, t_2, t_1.T, t_2.T])

def produce_c():
    c_1 = np.array([[-1., 1., 1.],[-1., 1., -1.],[-1., 1., 1.]])
    c_2 = np.array([[1., 1., -1.],[-1., 1., -1.],[1., 1., -1.]])
    for c in [c_1, c_2, c_1.T, c_2.T]:
        for i in range(9):
            c[i//3][i%3] = gen_pixel()*c[i//3][i%3]
    return np.array([c_1, c_2, c_1.T, c_2.T])

train_data = produce_t()
for _ in range(124):
    train_data = np.vstack([train_data, produce_t()])

for _ in range(125):
    train_data = np.vstack([train_data, produce_c()])

train_label = np.concatenate((np.zeros(500, dtype=np.int), np.ones(500, dtype=np.int)))


val_data = produce_t()
for _ in range(49):
    val_data = np.vstack([val_data, produce_t()])

for _ in range(50):
    val_data = np.vstack([val_data, produce_c()])

val_label = np.concatenate((np.zeros(200, dtype=np.int), np.ones(200, dtype=np.int)))

np.save('/disk/Dataset/T-C/pytorch_numpy_data/train_data.npy', train_data)
np.save('/disk/Dataset/T-C/pytorch_numpy_data/val_data.npy', val_data)
np.save('/disk/Dataset/T-C/pytorch_numpy_data/train_label.npy', train_label)
np.save('/disk/Dataset/T-C/pytorch_numpy_data/val_label.npy', val_label)

#produce XOR numpy data
num = 500
train_pos = np.vstack((np.stack((np.random.random(250), np.random.random(250))).T, np.stack((-np.random.random(250), -np.random.random(250))).T))
train_neg = np.vstack((np.stack((-np.random.random(250), np.random.random(250))).T, np.stack((np.random.random(250), -np.random.random(250))).T))
train_data = np.vstack((train_pos, train_neg))
train_label = np.concatenate((np.zeros(500, dtype=np.int), np.ones(500, dtype=np.int)))

val_pos = np.vstack((np.stack((np.random.random(100), np.random.random(100))).T, np.stack((-np.random.random(100), -np.random.random(100))).T))
val_neg = np.vstack((np.stack((-np.random.random(100), np.random.random(100))).T, np.stack((np.random.random(100), -np.random.random(100))).T))
val_data = np.vstack((val_pos, val_neg))
val_label = np.concatenate((np.zeros(200, dtype=np.int), np.ones(200, dtype=np.int)))
np.save('/disk/Dataset/XOR/pytorch_numpy_data/train_data.npy', train_data)
np.save('/disk/Dataset/XOR/pytorch_numpy_data/val_data.npy', val_data)
np.save('/disk/Dataset/XOR/pytorch_numpy_data/train_label.npy', train_label)
np.save('/disk/Dataset/XOR/pytorch_numpy_data/val_label.npy', val_label)

#produce Iris numpy data
iris=datasets.load_iris()
irisFeature=iris.data
irisTarget=iris.target
val_data = np.vstack((iris.data[35:50], iris.data[85:100], iris.data[135:150]))
train_data = np.vstack((iris.data[0:35], iris.data[50:85], iris.data[100:135]))
train_label = np.concatenate((iris.target[0:35], iris.target[50:85], iris.target[100:135]))
val_label = np.concatenate((iris.target[35:50], iris.target[85:100], iris.target[135:150]))

np.save('/disk/Dataset/Iris/pytorch_numpy_data/train_data.npy', train_data)
np.save('/disk/Dataset/Iris/pytorch_numpy_data/val_data.npy', val_data)
np.save('/disk/Dataset/Iris/pytorch_numpy_data/train_label.npy', train_label)
np.save('/disk/Dataset/Iris/pytorch_numpy_data/val_label.npy', val_label)


mnist_train_data = torchvision.datasets.MNIST(root='/disk/Dataset/', train=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_val_data = torchvision.datasets.MNIST(root='/disk/Dataset/', train=False, transform=transforms.Compose([transforms.ToTensor()]))

temp = torch.utils.data.DataLoader(dataset=mnist_train_data, batch_size=60000, shuffle=False)
for i , j in enumerate(temp):
    index, train_data = i, j

temp = torch.utils.data.DataLoader(dataset=mnist_val_data, batch_size=10000, shuffle=False)
for i , j in enumerate(temp):
    index, val_data = i, j

slconv = layer.SLConv(in_channel=1, stride=1, padding=1)
pool = torch.nn.MaxPool2d(4)

# x = mnist_train_data.__getitem__(0)[0].unsqueeze(1)
out = slconv(val_data[0])


root = '/disk/Dataset'
# cifar_train = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=True, download=True)
# cifar_test = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=False, download=True)

imagenet_train = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='train', download=True)
imagenet_val = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='val', download=True)